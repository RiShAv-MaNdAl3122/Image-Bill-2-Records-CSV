import os
import re
import numpy as np
from PIL import Image
from typing import Dict, Any, List
from utils.logger import logger
from modules.ocr import get_cached_ocr_boxes

def snap_to_whitespace(img, left, right, y, search_range=12):
    """
    img: PIL Image object
    left, right: horizontal bounds of column
    y: vertical coordinate to snap
    search_range: vertical search window
    """
    h = img.height
    y_min = max(0, y - search_range)
    y_max = min(h, y + search_range)
    
    if y_max <= y_min:
        return y
        
    # Crop the search strip
    strip = img.crop((left, y_min, right, y_max))
    # Convert to grayscale numpy array
    strip_gray = np.array(strip.convert("L"))
    
    # Threshold: anything darker than 240 is considered text (foreground)
    non_white = (strip_gray < 240).astype(np.uint8)
    row_sums = np.sum(non_white, axis=1)
    
    # Smoothed sums to avoid single-row noise
    if len(row_sums) >= 3:
        smoothed = np.convolve(row_sums, np.ones(3)/3, mode='same')
    else:
        smoothed = row_sums
    
    best_idx = y - y_min
    min_score = float('inf')
    center_idx = y - y_min
    
    for i, val in enumerate(smoothed):
        dist = abs(i - center_idx)
        # We penalize distance from center to prefer staying close to initial estimate
        score = val + dist * 8.0
        if score < min_score:
            min_score = score
            best_idx = i
            
    return y_min + best_idx

def get_snapped_boundaries(img, left, right, top, bottom, img_path):
    """
    Retrieves cached OCR boxes for the page, identifies column-specific Record Numbers
    and 'Available' markers, and snaps crop top/bottom to the exact text boxes if found.
    Falls back to whitespace valley snapping.
    """
    h = img.height
    scale_y = h / 1000.0
    scale_x = img.width / 750.0
    
    search_range = int(12 * scale_y)
    
    boxes = get_cached_ocr_boxes(img_path)
    
    snapped_top = None
    snapped_bottom = None
    
    if boxes:
        col_boxes = []
        for box_info in boxes:
            box = box_info[0]
            text, conf = box_info[1]
            x_min = min(pt[0] for pt in box)
            x_max = max(pt[0] for pt in box)
            y_min = min(pt[1] for pt in box)
            y_max = max(pt[1] for pt in box)
            center_x = (x_min + x_max) / 2
            
            # Check if box is inside the column bounds
            if left - int(10 * scale_x) <= center_x <= right + int(10 * scale_x):
                col_boxes.append((y_min, y_max, x_min, x_max, text))
                
        # Find Record Number boxes (5-digit number on the left side of the column)
        rec_boxes = []
        # Find 'Available' boxes (fuzzy matches)
        avail_boxes = []
        
        for y_min, y_max, x_min, x_max, text in col_boxes:
            text_clean = text.strip()
            # Check for 5-digit Record Number anchor
            if re.search(r'\b\d{5}\b', text_clean) or (len(text_clean) == 5 and text_clean.isdigit()):
                if x_min - left <= int(50 * scale_x):
                    rec_boxes.append((y_min, y_max, text_clean))
            
            # Check for Available marker
            text_lower = text_clean.lower()
            if any(kw in text_lower for kw in ['avail', 'avai', 'valable', 'lable']):
                avail_boxes.append((y_min, y_max, text_clean))
                
        # Match RecNum near top
        rec_boxes.sort(key=lambda x: x[0])
        for y_min, y_max, text_clean in rec_boxes:
            if abs(y_min - top) <= int(50 * scale_y):
                snapped_top = int(y_min)
                logger.debug(f"  Snapping crop top to RecNum ({text_clean}) box: {top} -> {snapped_top}")
                break
                
        # Match Available near bottom
        avail_boxes.sort(key=lambda x: x[0], reverse=True)
        for y_min, y_max, text_clean in avail_boxes:
            if abs(y_max - bottom) <= int(50 * scale_y):
                snapped_bottom = int(y_max + int(2 * scale_y))
                logger.debug(f"  Snapping crop bottom to 'Available' ({text_clean}) box: {bottom} -> {snapped_bottom}")
                break

    if snapped_top is None:
        snapped_top = snap_to_whitespace(img, left, right, top, search_range)
        logger.debug(f"  Whitespace-snapped crop top: {top} -> {snapped_top}")
        
    if snapped_bottom is None:
        snapped_bottom = snap_to_whitespace(img, left, right, bottom, search_range)
        logger.debug(f"  Whitespace-snapped crop bottom: {bottom} -> {snapped_bottom}")
        
    # Fail-safe: ensure bottom > top and within image bounds
    if snapped_bottom <= snapped_top:
        logger.warning(f"  Snapping resulted in invalid bounds (top={snapped_top}, bottom={snapped_bottom}). Reverting to unsnapped coordinates.")
        snapped_top = top
        snapped_bottom = max(top + 5, bottom)
        
    return max(0, snapped_top), min(h, snapped_bottom)

def get_segment_anchors(seg: Dict[str, Any]) -> tuple:
    """
    Analyzes OCR text boxes in the segment's column to detect if the segment contains:
    - A Record Number (5-digit number) near its top boundary.
    - An 'Available' marker near its bottom boundary or within the segment.
    Returns (has_rec, has_avail).
    """
    img_path = seg["image_path"]
    bbox = seg["bbox"]
    if not os.path.exists(img_path):
        return False, False
        
    with Image.open(img_path) as img:
        w, h = img.size
    scale_x = w / 750.0
    scale_y = h / 1000.0
    
    cols = [int(77 * scale_x), int(271 * scale_x), int(475 * scale_x), int(670 * scale_x)]
    
    # Robustly find column index
    dists = [abs(bbox[0] - cols[idx]) for idx in range(4)]
    c = int(np.argmin(dists[:3]))
    
    boxes = get_cached_ocr_boxes(img_path)
    col_recs = []
    col_avails = []
    
    if boxes:
        for box_info in boxes:
            box = box_info[0]
            text, conf = box_info[1]
            x_min = min(pt[0] for pt in box)
            x_max = max(pt[0] for pt in box)
            y_min = min(pt[1] for pt in box)
            y_max = max(pt[1] for pt in box)
            center_x = (x_min + x_max) / 2
            
            # Check if text box falls within this column's horizontal range
            if cols[c] - int(15 * scale_x) <= center_x <= cols[c+1] + int(15 * scale_x):
                text_clean = text.strip()
                # Check for 5-digit Record Number
                if re.search(r'\b\d{5}\b', text_clean) or (len(text_clean) == 5 and text_clean.isdigit()):
                    col_recs.append(y_min)
                # Check for Available marker
                text_lower = text_clean.lower()
                if any(kw in text_lower for kw in ['avail', 'avai', 'valable', 'lable']):
                    col_avails.append(y_max)
                    
    # Check if a Record No anchor is near the top of the segment bbox
    has_rec = any(abs(y - bbox[1]) <= int(60 * scale_y) for y in col_recs)
    
    # Check if an Available anchor is near the bottom of the segment bbox,
    # or inside the segment box (with a bit of buffer)
    has_avail = any(
        abs(y - bbox[3]) <= int(60 * scale_y) or 
        (y > bbox[1] + int(100 * scale_y) and y <= bbox[3] + int(10 * scale_y))
        for y in col_avails
    )
    
    return has_rec, has_avail

def crop_and_merge_record(record: Dict[str, Any], output_dir: str, ignore_anchors: bool = False) -> tuple:
    """
    Crops specified bounding box segments for a record, merges them by stacking
    them vertically, and saves the resulting image.
    
    bbox format: [x_min, y_min, x_max, y_max]
    """
    os.makedirs(output_dir, exist_ok=True)
    record_no = record["record_no"]
    segments = record["segments"]
    
    filtered_segments = []
    
    if len(segments) == 2:
        seg1, seg2 = segments[0], segments[1]
        has_rec1, has_avail1 = get_segment_anchors(seg1)
        
        if has_rec1 and has_avail1:
            logger.info(f"  Record {record_no}: Segment 1 is complete within column. Skipping Segment 2 merge.")
            record["merge_type"] = "none"
            filtered_segments = [seg1]
        else:
            if not has_rec1:
                if not ignore_anchors:
                    raise ValueError(f"Record {record_no}: missing Record Number anchor in Segment 1.")
                else:
                    logger.warning(f"Record {record_no}: missing Record Number anchor in Segment 1. Ignoring anchor check.")
                
            has_rec2, has_avail2 = get_segment_anchors(seg2)
            if not has_avail2:
                if not ignore_anchors:
                    raise ValueError(f"Record {record_no}: missing 'Available' anchor in Segment 2.")
                else:
                    logger.warning(f"Record {record_no}: missing 'Available' anchor in Segment 2. Ignoring anchor check.")
                
            filtered_segments = [seg1, seg2]
    elif len(segments) == 1:
        seg1 = segments[0]
        has_rec1, has_avail1 = get_segment_anchors(seg1)
        if not has_rec1:
            if not ignore_anchors:
                raise ValueError(f"Record {record_no}: missing Record Number anchor.")
            else:
                logger.warning(f"Record {record_no}: missing Record Number anchor. Ignoring anchor check.")
        if not has_avail1:
            if not ignore_anchors:
                raise ValueError(f"Record {record_no}: missing 'Available' anchor.")
            else:
                logger.warning(f"Record {record_no}: missing 'Available' anchor. Ignoring anchor check.")
        filtered_segments = [seg1]
    else:
        filtered_segments = segments

    if not filtered_segments:
        raise ValueError(f"Record {record_no} has no valid segments to crop.")
        
    cropped_images = []
    
    try:
        for seg in filtered_segments:
            img_path = seg["image_path"]
            bbox = seg["bbox"]
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Source image {img_path} not found.")
                
            with Image.open(img_path) as img:
                # Pillow crop box is (left, upper, right, lower)
                # Apply snapped top/bottom boundaries
                left = bbox[0]
                right = bbox[2]
                top, bottom = get_snapped_boundaries(img, left, right, bbox[1], bbox[3], img_path)
                cropped_img = img.crop((left, top, right, bottom))
                
                # Check segment crop height
                seg_width = right - left
                seg_height = bottom - top
                logger.debug(
                    f"  Crop [{record_no}] bbox=({left},{top},{right},{bottom}) "
                    f"→ {seg_width}x{seg_height}px from {os.path.basename(img_path)}"
                )
                if seg_height < 40:
                    logger.warning(f"Record {record_no} has a suspiciously small segment crop height: {seg_height}px")
                
                # Convert to RGB to ensure uniform mode
                cropped_images.append(cropped_img.convert("RGB"))

                
        # Trim white borders for each cropped segment to limit white blanks
        trimmed_images = []
        for cropped in cropped_images:
            # Convert to grayscale to detect white levels
            gray = cropped.convert("L")
            # Anything brighter than 240 is treated as white border
            bw = gray.point(lambda x: 0 if x > 240 else 255)
            bbox = bw.getbbox()
            if bbox:
                # Use small padding (5px horiz, 2px vert) to keep crop tight and avoid adjacent bills
                pad_x = 5
                pad_y = 2
                left = max(0, bbox[0] - pad_x)
                top = max(0, bbox[1] - pad_y)
                right = min(cropped.width, bbox[2] + pad_x)
                bottom = min(cropped.height, bbox[3] + pad_y)
                trimmed = cropped.crop((left, top, right, bottom))
            else:
                trimmed = cropped
            trimmed_images.append(trimmed)

        # Stack vertically
        widths, heights = zip(*(i.size for i in trimmed_images))
        max_width = max(widths)
        
        resized_images = []
        for img in trimmed_images:
            if img.width != max_width and img.width > 0:
                new_height = int(img.height * (max_width / img.width))
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
            resized_images.append(img)
            
        total_height = sum(img.height for img in resized_images)
        
        # Create blank canvas
        merged_img = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))
        
        y_offset = 0
        for img in resized_images:
            merged_img.paste(img, (0, y_offset))
            y_offset += img.height
            
        output_path = os.path.join(output_dir, f"{record_no}.png")
        merged_img.save(output_path, "PNG")
        merged_width, merged_height = merged_img.size
        return output_path, merged_height, merged_width
        
    except Exception as e:
        raise RuntimeError(f"Failed to crop and merge record {record_no}: {str(e)}")

if __name__ == '__main__':
    # Simple self-test
    test_record = {
        "record_no": "test_self_merge",
        "segments": [
            {
                "image_path": "input/media__1780801696663.jpg",
                "bbox": [77, 144, 271, 422]
            }
        ],
        "merge_type": "none"
    }
    
    # Check if input file exists before running test
    if os.path.exists(test_record["segments"][0]["image_path"]):
        out_path = crop_and_merge_record(test_record, "temp/merged_bills")
        print(f"Self-test success. Image saved to: {out_path}")
    else:
        print("Self-test skipped: test input image not found.")
