import os
import sys
import shutil
import json
import subprocess
import time
import requests
from typing import List, Dict, Any
from utils.logger import logger
import config
from modules.layout import get_sorted_images, detect_dividers_for_image, analyze_layout, validate_crop_against_layout, save_debug_overlay
from modules.cropper import crop_and_merge_record
from modules.ocr import extract_text
from modules.extractor import extract_fields, fill_missing_fields_with_raw_ocr
from modules.validator import validate_record, MANDATORY_FIELDS
from modules.exporter import (
    append_extracted_record, append_review_required, append_processing_log
)
import re
from modules.csv_schema import CSV_COLUMNS
from modules.recovery_engine import recover_record
from modules.normalizer import normalize_text


def sanitize_filename(name: str) -> str:
    """Removes or replaces characters that are illegal in file names."""
    return re.sub(r'[\\/*?:"<>|]', '_', name)


def check_field_validity(record: Dict[str, Any], validation: Dict[str, Any]) -> Dict[str, bool]:
    """Helper to check if each field in the record passed validation (is present and formatted)."""
    validity = {}
    missing = set(validation.get("missing_fields", []))
    fmt_errs = set(validation.get("format_errors", []))
    for field in CSV_COLUMNS:
        if field == "IMAGE NAME":
            continue
        val_entry = record.get(field, "")
        if isinstance(val_entry, dict) and "value" in val_entry:
            val = val_entry["value"]
        else:
            val = str(val_entry)
        val = val.strip()
        is_valid = bool(val) and (field not in missing) and (field not in fmt_errs)
        validity[field] = is_valid
    return validity


def run_pipeline() -> None:
    """
    Main orchestration function for the ImageBillExtractor pipeline.
    Processes pages sequentially (page-by-page streaming) to support 100+ images efficiently.
    """
    logger.info("Starting ImageBillExtractor pipeline...")

    # Ollama LLM setup has been removed from automatic batch extraction.

    # Clear temp directory at startup
    if os.path.exists(config.TEMP_FOLDER):
        logger.info(f"Clearing temp directory: {config.TEMP_FOLDER}")
        for filename in os.listdir(config.TEMP_FOLDER):
            file_path = os.path.join(config.TEMP_FOLDER, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {file_path}: {e}")

    # Clear output files and NeedReview folder at startup
    if os.path.exists(config.OUTPUT_FOLDER):
        logger.info(f"Clearing output CSVs and NeedReview folder in: {config.OUTPUT_FOLDER}")
        for csv_name in ["extracted_records.csv", "failed_records.csv", "processing_log.csv"]:
            file_path = os.path.join(config.OUTPUT_FOLDER, csv_name)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete output file {file_path}: {e}")
        
        need_review_dir = os.path.join(config.OUTPUT_FOLDER, "NeedReview")
        if os.path.exists(need_review_dir):
            try:
                shutil.rmtree(need_review_dir)
                logger.info("Cleared NeedReview directory.")
            except Exception as e:
                logger.warning(f"Failed to delete NeedReview folder: {e}")

    # 1. Fetch input images
    input_dir = config.INPUT_FOLDER
    logger.info(f"Scanning input directory: {os.path.abspath(input_dir)}")
    image_paths = get_sorted_images(input_dir)

    if not image_paths:
        logger.error("No valid page images found in the input folder.")
        return

    if hasattr(config, 'LIMIT_PAGES') and config.LIMIT_PAGES is not None:
        image_paths = image_paths[:config.LIMIT_PAGES]
        logger.info(f"Limiting execution to the first {config.LIMIT_PAGES} page(s) for testing.")

    # Optional explicit page range (1-based, inclusive)
    if hasattr(config, 'START_PAGE') and hasattr(config, 'END_PAGE') and config.START_PAGE is not None and config.END_PAGE is not None:
        start_idx = max(0, int(config.START_PAGE) - 1)
        end_idx = int(config.END_PAGE)
        image_paths = image_paths[start_idx:end_idx]
        logger.info(f"Limiting execution to pages {config.START_PAGE} through {config.END_PAGE} for testing.")

    logger.info(f"Found {len(image_paths)} valid page images to process.")

    # Pre-define output CSV paths
    extracted_csv_path = os.path.join(config.OUTPUT_FOLDER, "extracted_records.csv")
    review_csv_path = os.path.join(config.OUTPUT_FOLDER, "failed_records.csv")
    log_csv_path = os.path.join(config.OUTPUT_FOLDER, "processing_log.csv")

    # Initialize global recovery metrics
    global_recovery_metrics = {
        field: {
            "extracted": 0,
            "recovered_by_recovery_engine": 0,
            "recovered_by_llm": 0,
            "failed_completely": 0
        }
        for field in CSV_COLUMNS if field != "IMAGE NAME"
    }

    # 2. Process each page sequentially
    deferred_vert_segment = None  # Stores details of vertical continuation from previous page
    cols = [77, 271, 475, 670]

    total_records_processed = 0
    successful_count = 0
    failed_count = 0
    llm_assisted_count = 0

    # Initialize timing arrays
    ocr_times = []
    extraction_times = []
    recovery_times = []
    validation_times = []

    # Initialize error tracking for review analytics
    review_analytics = {
        "Missing blood group": 0,
        "Missing shipper name": 0,
        "Missing tablets": 0,
        "Date errors": 0,
        "Address errors": 0
    }

    t_start_pipeline = time.perf_counter()

    num_pages = len(image_paths)
    for p_idx, img_path in enumerate(image_paths):
        logger.info(f"--- Processing Page {p_idx + 1}/{num_pages}: {os.path.basename(img_path)} ---")

        if getattr(config, 'LAST_PAGE_ONLY', False) and p_idx != num_pages - 1:
            logger.info(f"Skipping Page {p_idx + 1} (LAST_PAGE_ONLY is enabled).")
            continue

        # Load page to find dimension-based scale factors
        import cv2
        img_temp = cv2.imread(img_path)
        if img_temp is None:
            logger.error(f"Failed to read image: {img_path}")
            continue
        h, w = img_temp.shape[:2]
        scale_x = w / 750.0
        scale_y = h / 1000.0

        # Calculate dynamic columns and limits for this page
        cols = [int(77 * scale_x), int(271 * scale_x), int(475 * scale_x), int(670 * scale_x)]
        y_top = int(80 * scale_y)
        y_bottom = int(980 * scale_y)

        # Page-specific temp folder to organize pictures
        page_temp_dir = os.path.join(config.TEMP_FOLDER, f"page_{p_idx + 1}")
        os.makedirs(page_temp_dir, exist_ok=True)

        # Detect dividers for current page (existing heuristic)
        divs = detect_dividers_for_image(img_path)
        # Augmented layout analysis (Hough horizontal dividers, whitespace valleys, dynamic columns)
        page_layout_info = analyze_layout(img_path)
        first_segment_img = os.path.basename(img_path)

        next_img_path = image_paths[p_idx + 1] if p_idx < num_pages - 1 else None

        # List of record tasks to execute for this page
        page_tasks = []

        # A. Handle the deferred vertical segment from the previous page
        if deferred_vert_segment is not None:
            c1_y1 = divs[0][0]
            record_plan = {
                "record_no": deferred_vert_segment["record_no"],
                "segments": [
                    {
                        "image_path": deferred_vert_segment["image_path"],
                        "bbox": deferred_vert_segment["bbox"]
                    },
                    {
                        "image_path": img_path,
                        "bbox": [cols[0], y_top, cols[1], c1_y1]
                    }
                ],
                "merge_type": "vertical"
            }
            page_tasks.append(record_plan)
            deferred_vert_segment = None

        # B. Define other tasks for the current page columns
        for c in range(3):
            y1, y2, y3 = divs[c]

            page_tasks.append({
                "record_no": f"rec_{p_idx+1}_{c+1}_row2",
                "segments": [
                    {
                        "image_path": img_path,
                        "bbox": [cols[c], y1, cols[c+1], y2]
                    }
                ],
                "merge_type": "none"
            })

            page_tasks.append({
                "record_no": f"rec_{p_idx+1}_{c+1}_row3",
                "segments": [
                    {
                        "image_path": img_path,
                        "bbox": [cols[c], y2, cols[c+1], y3]
                    }
                ],
                "merge_type": "none"
            })

            if c < 2:
                next_col_y1 = divs[c+1][0]
                page_tasks.append({
                    "record_no": f"rec_{p_idx+1}_{c+1}_row4_horiz",
                    "segments": [
                        {
                            "image_path": img_path,
                            "bbox": [cols[c], y3, cols[c+1], y_bottom]
                        },
                        {
                            "image_path": img_path,
                            "bbox": [cols[c+1], y_top, cols[c+2], next_col_y1]
                        }
                    ],
                    "merge_type": "horizontal"
                })
            else:
                if next_img_path:
                    deferred_vert_segment = {
                        "record_no": f"rec_{p_idx+1}_{c+1}_row4_vert",
                        "image_path": img_path,
                        "bbox": [cols[c], y3, cols[c+1], y_bottom]
                    }
                else:
                    page_tasks.append({
                        "record_no": f"rec_{p_idx+1}_{c+1}_row4_tail",
                        "segments": [
                            {
                                "image_path": img_path,
                                "bbox": [cols[c], y3, cols[c+1], y_bottom]
                            }
                        ],
                        "merge_type": "none"
                    })

        # C. Execute all page tasks (process one record at a time)
        for record_plan in page_tasks:
            record_placeholder = record_plan["record_no"]
            merge_type = record_plan["merge_type"]
            first_segment_img = os.path.basename(record_plan["segments"][0]["image_path"])

            logger.info(f"  Processing segment: {record_placeholder} (Merge: {merge_type})")
            total_records_processed += 1

            try:
                merged_img_path, crop_height, crop_width = crop_and_merge_record(record_plan, page_temp_dir)
                merge_type = record_plan["merge_type"]

                # OCR extraction
                t0_ocr = time.perf_counter()
                ocr_result = extract_text(merged_img_path)
                t1_ocr = time.perf_counter()
                ocr_times.append(t1_ocr - t0_ocr)

                ocr_text = ocr_result["text"]
                ocr_conf = ocr_result["confidence"]

                # Compute word count for audit
                word_count = len([w for w in ocr_text.split() if w])

                # Validate crop against detected layout to log warnings for suspicious crops
                try:
                    page_img = img_temp
                    page_h, page_w = page_img.shape[:2]
                    layout_check = validate_crop_against_layout(record_plan, page_layout_info, (page_h, page_w), crop_height, crop_width, word_count)
                    if layout_check.get('suspicious'):
                        logger.warning(f"    Suspicious crop detected for {record_plan['record_no']}.")
                except Exception:
                    pass

                # Write audit CSV is disabled

                # Short-circuit: skip records where the crop produced too little text
                # (fewer than 3 non-empty lines = almost certainly a boundary/layout artifact)
                non_empty_lines = [l for l in ocr_text.splitlines() if l.strip()]
                if len(non_empty_lines) < 3:
                    logger.warning(
                        f"    Skipping segment {record_placeholder}: OCR returned only "
                        f"{len(non_empty_lines)} non-empty line(s) — likely a layout artifact or empty crop."
                    )
                    total_records_processed -= 1  # don't count this toward the totals
                    continue

                logger.debug(f"    OCR Completed. Confidence: {ocr_conf:.2f}")

                # Stage checkpoints for recovery metrics
                initial_validity = {}
                recovery_validity = {}
                final_validity = {}

                # Step B.2: Normalize OCR text
                t0_ext = time.perf_counter()
                normalized_ocr_text = normalize_text(ocr_text)

                # Step C: Always run regex extractor first
                record_data = extract_fields(normalized_ocr_text, image_name=first_segment_img, ocr_confidence=ocr_conf)
                t1_ext = time.perf_counter()
                extraction_times.append(t1_ext - t0_ext)

                # Validation timer
                rec_val_time = 0.0

                t0_val = time.perf_counter()
                validation = validate_record(record_data)
                t1_val = time.perf_counter()
                rec_val_time += (t1_val - t0_val)
                
                # Check initial validity
                initial_status = validation["status"]
                initial_validity = check_field_validity(record_data, validation)

                # Log pipeline state
                logger.info(f"--- Pipeline Flow for Segment: {record_placeholder} ---")
                logger.info(f"  [1] RAW OCR TEXT:\n{ocr_text.strip()}\n")
                logger.info(f"  [2] NORMALIZED OCR TEXT:\n{normalized_ocr_text.strip()}\n")
                logger.info(f"  [3] EXTRACTED FIELDS (REGEX):\n{json.dumps({k: v for k, v in record_data.items() if v}, indent=2)}\n")
                logger.info(f"  [4] VALIDATION RESULT (INITIAL): Status={initial_status}, Errors={validation['errors']}\n")

                # Step D: Recovery Engine (First Run)
                t_recov_elapsed = 0.0
                if validation["status"] != "PASS":
                    logger.info("    Status is not PASS. Running Recovery Engine heuristics...")
                    t0_recov = time.perf_counter()
                    record_data = recover_record(
                        record_data, 
                        normalized_ocr_text, 
                        validation["missing_fields"], 
                        validation["format_errors"]
                    )
                    t1_recov = time.perf_counter()
                    t_recov_elapsed = t1_recov - t0_recov

                    t0_val = time.perf_counter()
                    validation = validate_record(record_data)
                    t1_val = time.perf_counter()
                    rec_val_time += (t1_val - t0_val)
                    logger.info(f"  [5] VALIDATION RESULT (POST-RECOVERY): Status={validation['status']}, Errors={validation['errors']}\n")
                
                recovery_times.append(t_recov_elapsed)
                
                # Check recovery validity
                recovery_validity = check_field_validity(record_data, validation)

                # Step E: LLM OCR Text Repair (Bypassed)

                # Step F: Fallback to raw OCR text if there are still validation errors
                if validation["status"] != "PASS":
                    logger.info("    Status is still not PASS. Filling missing fields with raw OCR fallbacks...")
                    t0_fb = time.perf_counter()
                    record_data = fill_missing_fields_with_raw_ocr(record_data, normalized_ocr_text)
                    t1_fb = time.perf_counter()
                    extraction_times[-1] += (t1_fb - t0_fb)

                    t0_val = time.perf_counter()
                    validation = validate_record(record_data)
                    t1_val = time.perf_counter()
                    rec_val_time += (t1_val - t0_val)
                    logger.info(f"  [7] VALIDATION RESULT (POST-FALLBACK): Status={validation['status']}, Errors={validation['errors']}\n")

                validation_times.append(rec_val_time)

                # Final validity check
                final_validity = check_field_validity(record_data, validation)

                # Record metrics for this segment
                for field in CSV_COLUMNS:
                    if field == "IMAGE NAME":
                        continue
                    if not final_validity.get(field, False):
                        status_category = "failed_completely"
                    elif initial_validity.get(field, False):
                        status_category = "extracted"
                    elif recovery_validity.get(field, False):
                        status_category = "recovered_by_recovery_engine"
                    else:
                        status_category = "failed_completely"
                    
                    global_recovery_metrics[field][status_category] += 1

                logger.info("--------------------------------------------------")

                # Step G: Handle validation result / output writing
                status = validation["status"]

                # Track errors in review analytics
                if status in ("REVIEW", "FAIL"):
                    missing_fields = validation.get("missing_fields", [])
                    format_errors = validation.get("format_errors", [])
                    
                    # Missing blood group
                    if "BLOOD GP" in missing_fields or "BLOOD GP" in format_errors:
                        review_analytics["Missing blood group"] += 1
                    
                    # Missing shipper name
                    if "SHIPPER NAME" in missing_fields:
                        review_analytics["Missing shipper name"] += 1
                        
                    # Missing tablets
                    if "TABLETS" in missing_fields:
                        review_analytics["Missing tablets"] += 1
                        
                    # Date errors
                    if any(f in missing_fields or f in format_errors for f in ("D BIRTH", "D_B_LIFE_ASSURE", "DOB")):
                        review_analytics["Date errors"] += 1
                        
                    # Address errors
                    if any(f in missing_fields or f in format_errors for f in ("RES_ADDRESS", "CITY_1", "STATE_1", "ZIP_1", "CITY_2", "STATE_2", "ZIP_2")):
                        review_analytics["Address errors"] += 1

                rec_no_entry = record_data.get("RECORD NO", "")
                if isinstance(rec_no_entry, dict) and "value" in rec_no_entry:
                    extracted_rec_no = rec_no_entry["value"]
                else:
                    extracted_rec_no = str(rec_no_entry)
                extracted_rec_no = extracted_rec_no.strip()
                
                display_rec_no = extracted_rec_no if extracted_rec_no else record_placeholder

                if status == "PASS":
                    logger.info(f"    Validation PASS for record: {display_rec_no}")

                    safe_rec_no = sanitize_filename(display_rec_no)
                    final_crop_name = f"{safe_rec_no}Merged.png" if merge_type != "none" else f"{safe_rec_no}.png"
                    final_crop_path = os.path.join(page_temp_dir, final_crop_name)
                    if os.path.exists(merged_img_path) and not os.path.exists(final_crop_path):
                        try:
                            os.rename(merged_img_path, final_crop_path)
                            logger.debug(f"    Renamed crop to: {final_crop_name}")
                        except Exception as rename_err:
                            logger.warning(f"    Failed to rename crop to {final_crop_name}: {rename_err}")

                    append_extracted_record(record_data, extracted_csv_path)

                    append_processing_log({
                        "record_no": display_rec_no,
                        "image_name": first_segment_img,
                        "status": status,
                        "errors": []
                    }, log_csv_path)
                    successful_count += 1
                else:
                    logger.warning(
                        f"    Validation {status} for record: {display_rec_no}. Errors: {validation['errors']}"
                    )

                    safe_rec_no = sanitize_filename(display_rec_no)
                    final_crop_name = f"{safe_rec_no}Merged-review.png" if merge_type != "none" else f"{safe_rec_no}-review.png"
                    final_crop_path = os.path.join(page_temp_dir, final_crop_name)
                    if os.path.exists(merged_img_path) and not os.path.exists(final_crop_path):
                        try:
                            os.rename(merged_img_path, final_crop_path)
                            logger.debug(f"    Renamed crop to: {final_crop_name}")
                        except Exception as rename_err:
                            logger.warning(f"    Failed to rename crop to {final_crop_name}: {rename_err}")

                    append_review_required(record_data, validation["errors"], review_csv_path)
                    append_processing_log({
                        "record_no": display_rec_no,
                        "image_name": first_segment_img,
                        "status": status,
                        "errors": validation["errors"]
                    }, log_csv_path)
                    failed_count += 1

                if status in ("REVIEW", "FAIL"):
                    need_review_dir = os.path.join(config.OUTPUT_FOLDER, "NeedReview")
                    os.makedirs(need_review_dir, exist_ok=True)
                    
                    crop_src = final_crop_path if 'final_crop_path' in locals() and os.path.exists(final_crop_path) else merged_img_path
                    if os.path.exists(crop_src):
                        try:
                            dest_name = os.path.basename(crop_src)
                            crop_dest = os.path.join(need_review_dir, dest_name)
                            shutil.copy(crop_src, crop_dest)
                            logger.info(f"    [Manual Review Needed] Copied crop image to: {crop_dest}")
                        except Exception as copy_err:
                            logger.warning(f"    Failed to copy crop image to NeedReview folder: {copy_err}")

            except Exception as e:
                err_msg = str(e)
                logger.error(f"    Pipeline error processing record {record_placeholder}: {err_msg}")
                
                # Check if a crop was already successfully generated before the exception occurred.
                # If so, save it to the NeedReview folder and do not delete/lose it.
                crop_saved = False
                if 'merged_img_path' in locals() and merged_img_path and os.path.exists(merged_img_path):
                    try:
                        safe_rec_no = sanitize_filename(record_placeholder)
                        final_crop_name = f"{safe_rec_no}Merged-review.png" if merge_type != "none" else f"{safe_rec_no}-review.png"
                        need_review_dir = os.path.join(config.OUTPUT_FOLDER, "NeedReview")
                        os.makedirs(need_review_dir, exist_ok=True)
                        crop_dest = os.path.join(need_review_dir, final_crop_name)
                        shutil.copy(merged_img_path, crop_dest)
                        logger.info(f"    [Manual Review Needed - Terminated] Saved existing crop to: {crop_dest}")
                        crop_saved = True
                    except Exception as copy_err:
                        logger.warning(f"    Failed to save existing crop on error: {copy_err}")
                
                if not crop_saved and "anchor" in err_msg.lower():
                    try:
                        logger.info(f"    Attempting anchor-bypass fallback crop for {record_placeholder}...")
                        fallback_img_path, fb_h, fb_w = crop_and_merge_record(record_plan, page_temp_dir, ignore_anchors=True)
                        
                        fallback_ocr = extract_text(fallback_img_path)
                        from modules.extractor import find_record_no
                        rec_no = find_record_no(fallback_ocr["text"])
                        if not rec_no:
                            rec_no = record_placeholder
                            
                        safe_rec_no = sanitize_filename(rec_no)
                        final_crop_name = f"{safe_rec_no}Merged-review.png" if merge_type != "none" else f"{safe_rec_no}-review.png"
                        final_crop_path = os.path.join(page_temp_dir, final_crop_name)
                        
                        if os.path.exists(fallback_img_path):
                            if not os.path.exists(final_crop_path):
                                os.rename(fallback_img_path, final_crop_path)
                            
                            need_review_dir = os.path.join(config.OUTPUT_FOLDER, "NeedReview")
                            os.makedirs(need_review_dir, exist_ok=True)
                            crop_dest = os.path.join(need_review_dir, final_crop_name)
                            shutil.copy(final_crop_path if os.path.exists(final_crop_path) else fallback_img_path, crop_dest)
                            logger.info(f"    [Manual Review Needed - Anchor Fallback] Saved fallback crop to: {crop_dest}")
                    except Exception as fallback_err:
                        logger.warning(f"    Failed to generate fallback crop for {record_placeholder}: {fallback_err}")
                
                append_review_required({col: "" for col in CSV_COLUMNS}, [err_msg], review_csv_path)
                append_processing_log({
                    "record_no": record_placeholder,
                    "image_name": first_segment_img,
                    "status": "ERROR",
                    "errors": [err_msg]
                }, log_csv_path)
                failed_count += 1

    logger.info("--- Pipeline Execution Summary ---")
    logger.info(f"  Total Records Processed:            {total_records_processed}")
    logger.info(f"  Successful Records (PASS):           {successful_count}")
    logger.info(f"  Failed Records (Review Required):    {failed_count}")
    logger.info(f"  Reports saved to: {config.OUTPUT_FOLDER}/")

    t_end_pipeline = time.perf_counter()
    total_elapsed = t_end_pipeline - t_start_pipeline

    # Log summary table
    logger.info("--- Recovery Metrics Summary ---")
    logger.info(f"{'Field Name':<20} | {'Extracted':<10} | {'Recov Eng':<10} | {'Failed':<10}")
    logger.info("-" * 50)
    for field, counts in global_recovery_metrics.items():
        logger.info(f"{field:<20} | {counts['extracted']:<10} | {counts['recovered_by_recovery_engine']:<10} | {counts['failed_completely']:<10}")

    # Write stats to stats.csv
    import csv
    stats_csv_path = os.path.join(config.OUTPUT_FOLDER, "stats.csv")
    try:
        with open(stats_csv_path, mode='w', encoding='utf-8', newline='') as sf:
            writer = csv.writer(sf, delimiter='|')
            writer.writerow(["METRIC_TYPE", "NAME", "EXTRACTED_OR_VALUE", "RECOVERED_BY_ENGINE", "FAILED"])
            writer.writerow(["SUMMARY", "Total Records Processed", str(total_records_processed), "N/A", "N/A"])
            writer.writerow(["SUMMARY", "Successful Records (PASS)", str(successful_count), "N/A", "N/A"])
            writer.writerow(["SUMMARY", "Failed Records (Review Required)", str(failed_count), "N/A", "N/A"])
            for field, counts in global_recovery_metrics.items():
                writer.writerow([
                    "RECOVERY", 
                    field, 
                    str(counts['extracted']), 
                    str(counts['recovered_by_recovery_engine']), 
                    str(counts['failed_completely'])
                ])
        logger.info(f"Stats saved to: {stats_csv_path}")
    except Exception as stats_err:
        logger.warning(f"Failed to write stats CSV: {stats_err}")

    logger.info("Pipeline completed successfully.")


if __name__ == '__main__':
    run_pipeline()
