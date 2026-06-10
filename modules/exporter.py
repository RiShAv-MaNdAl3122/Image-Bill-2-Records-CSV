import os
import csv
from datetime import datetime
from typing import List, Dict, Any
from modules.csv_schema import CSV_COLUMNS

def unpack_record(record: Dict[str, Any]) -> Dict[str, str]:
    """Helper to extract clean string values from value-confidence dictionaries."""
    unpacked = {}
    for k, v in record.items():
        if isinstance(v, dict) and "value" in v:
            unpacked[k] = v["value"]
        else:
            unpacked[k] = str(v)
    return unpacked

def export_extracted_records(records: List[Dict[str, Any]], filepath: str) -> None:
    """
    Exports successfully extracted and validated records to extracted_records.csv.
    Uses the centralized CSV schema.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        with open(filepath, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, delimiter='|')
            writer.writeheader()
            for rec in records:
                unpacked = unpack_record(rec)
                row = {col: unpacked.get(col, "") for col in CSV_COLUMNS}
                writer.writerow(row)
    except Exception as e:
        print(f"Failed to export extracted records: {str(e)}")
        raise

def export_review_required(records_with_errors: List[Dict[str, Any]], filepath: str) -> None:
    """
    Exports failed records to review_required.csv, including a 'VALIDATION_ERRORS' column.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = CSV_COLUMNS + ["VALIDATION_ERRORS"]
    
    try:
        with open(filepath, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='|')
            writer.writeheader()
            for item in records_with_errors:
                record = unpack_record(item["record"])
                errors = item["errors"]
                row = {col: record.get(col, "") for col in CSV_COLUMNS}
                row["VALIDATION_ERRORS"] = "; ".join(errors)
                writer.writerow(row)
    except Exception as e:
        print(f"Failed to export review required records: {str(e)}")
        raise

def export_processing_log(log_entries: List[Dict[str, Any]], filepath: str) -> None:
    """
    Writes processing results to processing_log.csv.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = ["TIMESTAMP", "RECORD_NO", "IMAGE_NAME", "STATUS", "ERRORS"]
    
    try:
        file_exists = os.path.exists(filepath)
        with open(filepath, mode='a' if file_exists else 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='|')
            if not file_exists:
                writer.writeheader()
                
            for entry in log_entries:
                writer.writerow({
                    "TIMESTAMP": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "RECORD_NO": entry.get("record_no", ""),
                    "IMAGE_NAME": entry.get("image_name", ""),
                    "STATUS": entry.get("status", ""),
                    "ERRORS": "; ".join(entry.get("errors", []))
                })
    except Exception as e:
        print(f"Failed to export processing log: {str(e)}")
        raise

def append_extracted_record(record: Dict[str, Any], filepath: str) -> None:
    """
    Appends a single validated record incrementally to the CSV file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.exists(filepath)
    with open(filepath, mode='a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, delimiter='|')
        if not file_exists:
            writer.writeheader()
        unpacked = unpack_record(record)
        row = {col: unpacked.get(col, "") for col in CSV_COLUMNS}
        writer.writerow(row)

def append_review_required(record: Dict[str, Any], errors: List[str], filepath: str) -> None:
    """
    Appends a single validation-failed record incrementally to the review CSV.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.exists(filepath)
    fieldnames = CSV_COLUMNS + ["VALIDATION_ERRORS"]
    with open(filepath, mode='a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='|')
        if not file_exists:
            writer.writeheader()
        unpacked = unpack_record(record)
        row = {col: unpacked.get(col, "") for col in CSV_COLUMNS}
        row["VALIDATION_ERRORS"] = "; ".join(errors)
        writer.writerow(row)

def append_processing_log(entry: Dict[str, Any], filepath: str) -> None:
    """
    Appends a single processing log entry incrementally to the log CSV.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.exists(filepath)
    fieldnames = ["TIMESTAMP", "RECORD_NO", "IMAGE_NAME", "STATUS", "ERRORS"]
    with open(filepath, mode='a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='|')
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "TIMESTAMP": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "RECORD_NO": entry.get("record_no", ""),
            "IMAGE_NAME": entry.get("image_name", ""),
            "STATUS": entry.get("status", ""),
            "ERRORS": "; ".join(entry.get("errors", []))
        })
