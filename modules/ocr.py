import os
import sys
from typing import Dict, Any, List
from paddleocr import PaddleOCR
import logging

# Suppress PaddleOCR warnings/infos
logging.getLogger("ppocr").setLevel(logging.ERROR)

# Lazily load OCR model to avoid multiple instantiations
_ocr_engine = None
_page_ocr_cache = {}

def cache_page_ocr(image_path: str, boxes: list):
    _page_ocr_cache[image_path] = boxes

def get_cached_ocr_boxes(image_path: str) -> list:
    return _page_ocr_cache.get(image_path, None)

def get_ocr_engine() -> PaddleOCR:
    global _ocr_engine
    if _ocr_engine is None:
        # Isolated configuration of PaddleOCR
        _ocr_engine = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
        try:
            from ppocr.utils.logging import get_logger
            get_logger().setLevel(logging.ERROR)
        except Exception:
            pass
    return _ocr_engine

def extract_text(image_path: str) -> Dict[str, Any]:
    """
    Runs PaddleOCR on the specified image and returns a dictionary containing
    the reconstructed text (sorted by lines, top-to-bottom, left-to-right)
    and the average confidence score.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found for OCR: {image_path}")
        
    ocr_engine = get_ocr_engine()
    
    try:
        # result is a list of lists of [box, (text, confidence)]
        result = ocr_engine.ocr(image_path, cls=False)
    except Exception as e:
        raise RuntimeError(f"PaddleOCR processing error: {str(e)}")
        
    if not result or result[0] is None:
        return {
            "text": "",
            "confidence": 0.0
        }
        
    # Extract boxes, texts, and confidences
    lines_info = []
    for line in result[0]:
        box = line[0]
        text, conf = line[1]
        
        # Calculate bounding coordinates
        x_min = min(pt[0] for pt in box)
        x_max = max(pt[0] for pt in box)
        y_min = min(pt[1] for pt in box)
        y_max = max(pt[1] for pt in box)
        center_y = (y_min + y_max) / 2
        
        lines_info.append({
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "center_y": center_y,
            "text": text,
            "confidence": conf
        })
        
    if not lines_info:
        return {
            "text": "",
            "confidence": 0.0
        }
        
    # Group boxes into lines based on vertical overlap ratio
    # Sort boxes by center_y first
    lines_info.sort(key=lambda item: item["center_y"])
    
    grouped_lines = []
    for box in lines_info:
        found_line = False
        box_ymin = box["y_min"]
        box_ymax = box["y_max"]
        box_height = box_ymax - box_ymin
        
        # Check existing lines from most recent to oldest
        for line in reversed(grouped_lines):
            line_ymin = min(item["y_min"] for item in line)
            line_ymax = max(item["y_max"] for item in line)
            line_height = line_ymax - line_ymin
            
            overlap = min(box_ymax, line_ymax) - max(box_ymin, line_ymin)
            min_h = min(box_height, line_height)
            
            if min_h > 0:
                overlap_ratio = overlap / min_h
                if overlap_ratio >= 0.5:
                    line.append(box)
                    found_line = True
                    break
                    
        if not found_line:
            grouped_lines.append([box])
        
    # Sort boxes within each line by x_min (left-to-right)
    final_text_lines = []
    confidences = []
    
    for line in grouped_lines:
        line.sort(key=lambda item: item["x_min"])
        line_text = " ".join(item["text"] for item in line)
        final_text_lines.append(line_text)
        confidences.extend(item["confidence"] for item in line)
        
    merged_text = "\n".join(final_text_lines)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    return {
        "text": merged_text,
        "confidence": avg_confidence
    }

if __name__ == '__main__':
    # Simple self-test using the cropped test image from the previous phase
    test_img = "temp/merged_bills/test_self_merge.png"
    if os.path.exists(test_img):
        print("Running OCR self-test...")
        res = extract_text(test_img)
        print("Confidence:", res["confidence"])
        print("OCR Text:\n", res["text"])
    else:
        print("OCR self-test skipped: test input image not found.")
