import os
import cv2
import numpy as np
from typing import List, Dict, Any
import config
import re
from modules.ocr import get_ocr_engine

def get_sorted_images(input_dir: str) -> List[str]:
    """
    Finds and returns sorted paths of valid bill images from the input directory.
    Filters out directories and files that do not meet size thresholds (e.g., thin slices).
    Sorts files using natural numeric order so different naming conventions with incremented
    numbers (like page_1.png, page_10.png) sort correctly.
    """
    import re
    valid_extensions = ('.png', '.jpg', '.jpeg')
    images = []
    
    if not os.path.exists(input_dir):
        return []
        
    for file in os.listdir(input_dir):
        if file.lower().endswith(valid_extensions):
            full_path = os.path.join(input_dir, file)
            try:
                img = cv2.imread(full_path)
                if img is not None:
                    h, w = img.shape[:2]
                    # Expected bill page has width/height > 500
                    if w > 500 and h > 500:
                        images.append(full_path)
            except Exception:
                pass
                
    def natural_sort_key(s: str) -> list:
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    images.sort(key=natural_sort_key)
    return images

def detect_dividers_for_image(image_path: str) -> Dict[int, List[int]]:
    """
    Dynamically detects the y-coordinates of horizontal divider lines for each column.
    Returns a dictionary mapping column index (0, 1, 2) to a list of [y1, y2, y3] dividers.
    Uses OCR text boxes to locate 5-digit Record Number anchors, snaps them to Hough lines,
    and uses robust interpolation for any missing/undetected anchors.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {0: [144, 422, 700], 1: [144, 422, 700], 2: [144, 422, 700]}
        
    h, w = img.shape[:2]
    scale_x = w / 750.0
    scale_y = h / 1000.0

    cols = [int(77 * scale_x), int(271 * scale_x), int(475 * scale_x), int(670 * scale_x)]
    
    # 1. Run full-page OCR
    try:
        ocr = get_ocr_engine()
        result = ocr.ocr(image_path, cls=False)
        if result and result[0]:
            from modules.ocr import cache_page_ocr
            cache_page_ocr(image_path, result[0])
    except Exception as e:
        # If OCR fails, fall back to standard scaled defaults
        from utils.logger import logger
        logger.warning(f"Full-page OCR layout scan failed: {e}. Falling back to default layout.")
        return {
            0: [int(144 * scale_y), int(422 * scale_y), int(700 * scale_y)],
            1: [int(144 * scale_y), int(422 * scale_y), int(700 * scale_y)],
            2: [int(144 * scale_y), int(422 * scale_y), int(700 * scale_y)]
        }
        
    if not result or not result[0]:
        return {
            0: [int(144 * scale_y), int(422 * scale_y), int(700 * scale_y)],
            1: [int(144 * scale_y), int(422 * scale_y), int(700 * scale_y)],
            2: [int(144 * scale_y), int(422 * scale_y), int(700 * scale_y)]
        }
        
    boxes = result[0]
    
    # Group OCR boxes by column
    col_boxes = {0: [], 1: [], 2: []}
    for box_info in boxes:
        box = box_info[0]
        text, conf = box_info[1]
        x_min = min(pt[0] for pt in box)
        x_max = max(pt[0] for pt in box)
        y_min = min(pt[1] for pt in box)
        y_max = max(pt[1] for pt in box)
        center_x = (x_min + x_max) / 2
        
        for c in range(3):
            if cols[c] <= center_x <= cols[c+1]:
                col_boxes[c].append((y_min, y_max, x_min, x_max, text))
                break
                
    col_dividers = {}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    for c in range(3):
        x_start = cols[c]
        x_end = cols[c+1]
        col_thresh = thresh[:, x_start:x_end]
        
        # Hough lines parameters tuned for horizontal grid lines
        h_lines = cv2.HoughLinesP(col_thresh, 1, np.pi/180, threshold=30, minLineLength=int(40 * scale_x), maxLineGap=15)
        y_lines = []
        if h_lines is not None:
            for line in h_lines:
                x1, y1, x2, y2 = line[0]
                if abs(y1 - y2) <= 8:
                    y_lines.append(int((y1 + y2)/2))
        y_lines = sorted(list(set(y_lines)))
        
        grouped_lines = []
        if y_lines:
            curr = y_lines[0]
            group = [curr]
            for y in y_lines[1:]:
                if y - curr <= int(10 * scale_y):
                    group.append(y)
                else:
                    grouped_lines.append(int(np.mean(group)))
                    group = [y]
                curr = y
            grouped_lines.append(int(np.mean(group)))
            
        # Define vertical search ranges for row anchors (Ratios of height)
        ranges = [
            (0.10, 0.35),  # Row 2 (y1)
            (0.35, 0.65),  # Row 3 (y2)
            (0.60, 0.90)   # Row 4 (y3)
        ]
        
        div_y_coords = []
        
        for idx, (y_min_r, y_max_r) in enumerate(ranges):
            y_start = y_min_r * h
            y_end = y_max_r * h
            
            candidates = []
            for item in col_boxes[c]:
                item_y_min, item_y_max, item_x_min, item_x_max, text = item
                if y_start <= item_y_min <= y_end:
                    # Look for 5-digit number at the beginning of the text
                    if re.match(r'^\s*\|?\s*\d{5}', text):
                        # Ensure the box is on the left side of the column (Record Number position)
                        if item_x_min - cols[c] <= int(40 * scale_x):
                            candidates.append((item_y_min, text))
                            
            anchor_y = None
            if candidates:
                candidates.sort(key=lambda x: x[0])
                anchor_y = candidates[0][0]
                
            if anchor_y is not None:
                candidate_divider = anchor_y - int(8 * scale_y)
                
                # Snap to closest Hough line in vicinity
                snapped_y = None
                closest_dist = float('inf')
                for hl in grouped_lines:
                    dist = abs(hl - candidate_divider)
                    if dist <= int(25 * scale_y) and dist < closest_dist:
                        snapped_y = hl
                        closest_dist = dist
                        
                if snapped_y is not None:
                    div_y_coords.append(int(snapped_y))
                else:
                    div_y_coords.append(int(candidate_divider))
            else:
                div_y_coords.append(None)
                
        # Interpolate missing dividers using neighboring row heights
        # Case A: y1, y2 present, y3 missing -> y3 = y2 + (y2 - y1)
        if div_y_coords[2] is None and div_y_coords[1] is not None and div_y_coords[0] is not None:
            div_y_coords[2] = div_y_coords[1] + (div_y_coords[1] - div_y_coords[0])
        # Case B: y2, y3 present, y1 missing -> y1 = y2 - (y3 - y2)
        elif div_y_coords[0] is None and div_y_coords[1] is not None and div_y_coords[2] is not None:
            div_y_coords[0] = div_y_coords[1] - (div_y_coords[2] - div_y_coords[1])
        # Case C: y1, y3 present, y2 missing -> y2 = (div_y_coords[0] + div_y_coords[2]) // 2
        elif div_y_coords[1] is None and div_y_coords[0] is not None and div_y_coords[2] is not None:
            div_y_coords[1] = int((div_y_coords[0] + div_y_coords[2]) / 2)
            
        # Fill remaining None with shifted defaults
        defaults = [int(144 * scale_y), int(422 * scale_y), int(700 * scale_y)]
        for idx in range(3):
            if div_y_coords[idx] is None:
                valid_idx = next((i for i in range(3) if div_y_coords[i] is not None), None)
                if valid_idx is not None:
                    diff_default = defaults[idx] - defaults[valid_idx]
                    div_y_coords[idx] = div_y_coords[valid_idx] + diff_default
                else:
                    div_y_coords[idx] = defaults[idx]
                    
        col_dividers[c] = div_y_coords
        
    return col_dividers


def detect_horizontal_dividers_opencv(image_path: str, canny_thresh1=50, canny_thresh2=150) -> List[int]:
    """Detect strong horizontal divider lines across the page using Canny + Hough.
    Returns list of Y coordinates (image-space) for detected horizontal lines.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use adaptive equalization to improve edge response on varied scans
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # Canny edges
    edges = cv2.Canny(gray, canny_thresh1, canny_thresh2, apertureSize=3)

    # Morphological close to join broken horizontal edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # HoughLinesP to find line segments
    lines = cv2.HoughLinesP(closed, 1, np.pi/180, threshold=100, minLineLength=80, maxLineGap=20)
    y_coords = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            if abs(y1 - y2) <= 6:  # roughly horizontal
                y_coords.append(int((y1 + y2) / 2))
    # Cluster nearby y coords
    y_coords = sorted(list(set(y_coords)))
    clustered = []
    if y_coords:
        group = [y_coords[0]]
        for y in y_coords[1:]:
            if y - group[-1] <= 12:
                group.append(y)
            else:
                clustered.append(int(np.mean(group)))
                group = [y]
        clustered.append(int(np.mean(group)))
    return clustered


def whitespace_projection_valleys(image_path: str, thresh_ratio=0.03, min_valley_height=10) -> List[int]:
    """Compute horizontal whitespace projection and return candidate valley midpoints (y-coords).
    `thresh_ratio` is fraction of page width below which a row is considered whitespace.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    h, w = img.shape[:2]
    # Binarize using Otsu (white background -> foreground 255)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Consider white pixels as 255; compute row-wise fraction of white pixels
    row_white_frac = np.sum(bw == 255, axis=1) / float(w)

    # Rows with white fraction above (1 - thresh_ratio) are very white
    white_rows = (row_white_frac >= (1.0 - thresh_ratio)).astype(np.uint8)

    # Find contiguous white regions and treat midpoints as valleys
    valleys = []
    start = None
    for i, val in enumerate(white_rows):
        if val == 1 and start is None:
            start = i
        elif val == 0 and start is not None:
            end = i - 1
            if end - start + 1 >= min_valley_height:
                valleys.append((start, end))
            start = None
    if start is not None:
        end = len(white_rows) - 1
        if end - start + 1 >= min_valley_height:
            valleys.append((start, end))

    midpoints = [int((s + e) / 2) for s, e in valleys]
    return midpoints


def detect_columns_from_projection(image_path: str, threshold_ratio=0.15) -> List[int]:
    """Detect column dividers (x-coordinates) using vertical projection.
    Returns list of x positions indicating column boundaries (including 0 and width).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [0, 750]
    h, w = img.shape[:2]
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    col_white_frac = np.sum(bw == 255, axis=0) / float(h)

    # Identify vertical separators where white_frac is high (i.e., empty vertical strips)
    sep_mask = col_white_frac >= (1.0 - threshold_ratio)
    boundaries = []
    start = None
    for x, val in enumerate(sep_mask):
        if val and start is None:
            start = x
        elif not val and start is not None:
            end = x - 1
            boundaries.append((start, end))
            start = None
    if start is not None:
        boundaries.append((start, w-1))

    # Convert boundaries to cut positions between columns (take midpoints)
    cuts = [0]
    for s, e in boundaries:
        cuts.append(int((s + e) / 2))
    cuts.append(w)

    # If cuts are too many/too few, fallback to simple equal thirds
    if len(cuts) < 3:
        return [0, int(w/3), int(2*w/3), w]
    # Ensure sorted unique
    cuts = sorted(list(dict.fromkeys(cuts)))
    return cuts


def validate_crop_against_layout(record_plan: Dict[str, Any], page_layout: Dict[str, Any], page_shape: tuple, crop_height: int, crop_width: int, word_count: int) -> Dict[str, Any]:
    """Validate the crop using layout info and return a small confidence report.
    record_plan: the record spec with segments and bboxes (page coordinates).
    page_layout: dict containing 'h_lines', 'wh_valleys', 'cols' (x coords list)
    page_shape: (h, w)
    """
    h, w = page_shape
    # Compute simple metrics
    min_height = max(40, int(0.04 * h))
    min_words = 4
    height_score = min(1.0, crop_height / max(min_height, crop_height)) if crop_height > 0 else 0.0
    width_score = min(1.0, crop_width / w)
    word_score = min(1.0, word_count / max(min_words, word_count)) if word_count > 0 else 0.0

    # Determine nearest horizontal divider distance for each segment bbox
    max_allowed_gap = int(0.08 * h)
    seg_misalign_penalty = 0
    middle_lines = [y for y in page_layout.get('h_lines', []) if int(0.2 * h) <= y <= int(0.8 * h)]
    
    for seg in record_plan.get('segments', []):
        bbox = seg['bbox']
        # use bottom y of bbox to compare to nearest h_line
        y_bottom = bbox[3]
        is_middle_divider = int(0.2 * h) <= y_bottom <= int(0.8 * h)
        if is_middle_divider and middle_lines:
            dists = [abs(y_bottom - y) for y in middle_lines]
            if dists:
                nearest = min(dists)
                if nearest > max_allowed_gap:
                    seg_misalign_penalty += 1
        elif not is_middle_divider and page_layout.get('h_lines'):
            dists = [abs(y_bottom - y) for y in page_layout['h_lines']]
            if dists:
                nearest = min(dists)
                if nearest > max_allowed_gap:
                    seg_misalign_penalty += 1

    # Combine scores (simple weighted average)
    base = 0.4 * height_score + 0.3 * width_score + 0.3 * word_score
    # reduce confidence per misaligned segment
    penalty = max(0.0, seg_misalign_penalty * 0.15)
    confidence = max(0.0, base - penalty)

    suspicious = False
    reasons = []
    if crop_height < min_height:
        suspicious = True
        reasons.append('too_short')
    if word_count < min_words:
        suspicious = True
        reasons.append('low_word_count')
    if seg_misalign_penalty > 0:
        suspicious = True
        reasons.append('misaligned_with_dividers')

    return {
        'record_no': record_plan.get('record_no'),
        'height': crop_height,
        'width': crop_width,
        'word_count': word_count,
        'confidence': round(confidence, 3),
        'suspicious': suspicious,
        'reasons': reasons
    }


def save_debug_overlay(image_path: str, out_path: str, h_lines: List[int]=None, v_cuts: List[int]=None, crops: List[Dict[str, Any]]=None):
    """Save a visualization overlay showing horizontal lines, vertical cuts, and crop rects.
    crops: list of {'bbox': [x1,y1,x2,y2], 'record_no': str}
    """
    img = cv2.imread(image_path)
    if img is None:
        return
    overlay = img.copy()
    h, w = img.shape[:2]
    # Draw horizontal lines
    if h_lines:
        for y in h_lines:
            cv2.line(overlay, (0, y), (w, y), (0, 0, 255), 2)
    # Draw vertical cuts
    if v_cuts:
        for x in v_cuts:
            cv2.line(overlay, (x, 0), (x, h), (255, 0, 0), 2)
    # Draw crops
    if crops:
        for c in crops:
            bbox = c.get('bbox')
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(overlay, c.get('record_no', ''), (x1+4, y1+14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # Blend with original for visibility
    blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, blended)


def analyze_layout(image_path: str) -> Dict[str, Any]:
    """Top-level convenience function to return a page layout analysis dict.
    Contains horizontal lines, whitespace valleys, and detected column cuts.
    """
    h_lines = detect_horizontal_dividers_opencv(image_path)
    wh_valleys = whitespace_projection_valleys(image_path)
    v_cuts = detect_columns_from_projection(image_path)
    return {
        'h_lines': h_lines,
        'wh_valleys': wh_valleys,
        'v_cuts': v_cuts
    }

def generate_layout_plan(image_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Generates a structured list of record segments and merge types from sorted image pages.
    Coordinates are dynamically adjusted for each page and column using Hough line detection.
    """
    records = []
    num_pages = len(image_paths)
    if num_pages == 0:
        return records
        
    # Pre-detect dividers for all pages
    page_dividers = {}
    for p, img_path in enumerate(image_paths):
        page_dividers[p] = detect_dividers_for_image(img_path)
        
    # Handle page 1 Col 1 Row 1 orphan segment
    p1_img = cv2.imread(image_paths[0])
    p1_h, p1_w = p1_img.shape[:2] if p1_img is not None else (1000, 750)
    p1_scale_x = p1_w / 750.0
    p1_scale_y = p1_h / 1000.0
    
    p1_cols = [int(77 * p1_scale_x), int(271 * p1_scale_x), int(475 * p1_scale_x), int(670 * p1_scale_x)]
    p1_c1_y1 = page_dividers[0][0][0]
    
    records.append({
        "record_no": "rec_000_orphan",
        "segments": [
            {
                "image_path": image_paths[0],
                "bbox": [p1_cols[0], int(80 * p1_scale_y), p1_cols[1], p1_c1_y1]
            }
        ],
        "merge_type": "none"
    })
    
    for p in range(num_pages):
        img_path = image_paths[p]
        img = cv2.imread(img_path)
        h, w = img.shape[:2] if img is not None else (1000, 750)
        scale_x = w / 750.0
        scale_y = h / 1000.0
        
        cols = [int(77 * scale_x), int(271 * scale_x), int(475 * scale_x), int(670 * scale_x)]
        y_top = int(80 * scale_y)
        y_bottom = int(980 * scale_y)
        
        divs = page_dividers[p]
        next_img_path = image_paths[p+1] if p < num_pages - 1 else None
        next_divs = page_dividers[p+1] if p < num_pages - 1 else None
        
        for c in range(3):
            y1, y2, y3 = divs[c]
            
            # Row 2
            records.append({
                "record_no": f"rec_{p+1}_{c+1}_row2",
                "segments": [{"image_path": img_path, "bbox": [cols[c], y1, cols[c+1], y2]}],
                "merge_type": "none"
            })
            
            # Row 3
            records.append({
                "record_no": f"rec_{p+1}_{c+1}_row3",
                "segments": [{"image_path": img_path, "bbox": [cols[c], y2, cols[c+1], y3]}],
                "merge_type": "none"
            })
            
            # Row 4
            if c < 2:
                next_col_y1 = divs[c+1][0]
                records.append({
                    "record_no": f"rec_{p+1}_{c+1}_row4_horiz",
                    "segments": [
                        {"image_path": img_path, "bbox": [cols[c], y3, cols[c+1], y_bottom]},
                        {"image_path": img_path, "bbox": [cols[c+1], y_top, cols[c+2], next_col_y1]}
                    ],
                    "merge_type": "horizontal"
                })
            else:
                if next_img_path and next_divs:
                    next_page_c1_y1 = next_divs[0][0]
                    next_img = cv2.imread(next_img_path)
                    next_w = next_img.shape[1] if next_img is not None else 750
                    next_scale_x = next_w / 750.0
                    next_cols = [int(77 * next_scale_x), int(271 * next_scale_x)]
                    next_h = next_img.shape[0] if next_img is not None else 1000
                    next_scale_y = next_h / 1000.0
                    
                    records.append({
                        "record_no": f"rec_{p+1}_{c+1}_row4_vert",
                        "segments": [
                            {"image_path": img_path, "bbox": [cols[c], y3, cols[c+1], y_bottom]},
                            {"image_path": next_img_path, "bbox": [next_cols[0], int(80 * next_scale_y), next_cols[1], next_page_c1_y1]}
                        ],
                        "merge_type": "vertical"
                    })
                else:
                    records.append({
                        "record_no": f"rec_{p+1}_{c+1}_row4_tail",
                        "segments": [{"image_path": img_path, "bbox": [cols[c], y3, cols[c+1], y_bottom]}],
                        "merge_type": "none"
                    })
    return records
                    
    return records

if __name__ == '__main__':
    # Simple self-test
    input_folder = os.path.join("..", config.INPUT_FOLDER) if os.path.exists(os.path.join("..", config.INPUT_FOLDER)) else config.INPUT_FOLDER
    imgs = get_sorted_images(input_folder)
    print(f"Found {len(imgs)} valid images:")
    plan = generate_layout_plan(imgs)
    print(f"Generated {len(plan)} record segments.")
