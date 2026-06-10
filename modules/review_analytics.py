import csv
import json
from collections import Counter
from typing import Dict

KEYWORD_MAP = [
    ("ZIP", "zip_missing"),
    ("ZIP_2", "zip_2_missing"),
    ("BLOOD", "blood_group_error"),
    ("SHIPPER", "shipper_name_corruption"),
    ("POLICY", "policy_holder_parse"),
    ("DOB", "date_parse"),
    ("D BIRTH", "date_parse"),
    ("EMAIL", "email_parse"),
    ("PH_NO", "phone_parse"),
    ("PHONE", "phone_parse"),
]


def analyze_review_csv(filepath: str) -> Dict[str, int]:
    """Reads a review-required CSV (pipe-delimited) and returns grouped root-cause counts.

    The function uses simple heuristics on the VALIDATION_ERRORS column to group causes.
    """
    counter = Counter()
    try:
        with open(filepath, encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='|')
            for row in reader:
                errs = row.get('VALIDATION_ERRORS', '') or ''
                key = None
                up = errs.upper()
                for kw, root in KEYWORD_MAP:
                    if kw in up:
                        key = root
                        break
                if not key:
                    # fallback heuristics
                    if any(ch.isdigit() for ch in up) and ('SR_' in up or 'SR' in up or 'POLICY' in up):
                        key = 'policy_holder_parse'
                    else:
                        key = 'other'
                counter[key] += 1
    except FileNotFoundError:
        return {}

    # Return sorted dict
    return dict(counter.most_common())


def write_analytics_json(counts: Dict[str, int], outpath: str) -> None:
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(counts, f, indent=2)
