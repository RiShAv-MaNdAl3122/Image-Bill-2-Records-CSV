import re
from typing import Dict, Any, List, Tuple
from utils.regex_patterns import EMAIL_PATTERN
from modules.csv_schema import CSV_COLUMNS

# All CSV columns are mandatory — IMAGE NAME is always auto-filled from filename
_NON_MANDATORY = {"IMAGE NAME"}
MANDATORY_FIELDS = [col for col in CSV_COLUMNS if col not in _NON_MANDATORY]

# Fields where format errors should NOT block export (soft format checks)
# These are accepted even if they don't exactly match the strict format
_SOFT_FORMAT_FIELDS = {"SHIPPING CO", "CARD NAME", "STM NAME", "STM CODE", "MEDICINE", "DOSAGE"}


def validate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates a single record dictionary against formatting and mandatory presence rules.

    Returns a dict with:
        'valid'           : bool — True if record can be exported as-is
        'errors'          : list of str — All errors found
        'missing_fields'  : list of str — Mandatory fields that are empty
        'format_errors'   : list of str — Fields present but incorrectly formatted
    """
    # Unpack value/confidence structures to clean string representation for validation
    unpacked_record = {}
    for k, v in record.items():
        if isinstance(v, dict) and "value" in v:
            unpacked_record[k] = v["value"]
        else:
            unpacked_record[k] = str(v)
            
    errors = []
    missing_fields = []
    format_errors = []

    # 1. Check Mandatory Fields — ALL fields are required
    for field in MANDATORY_FIELDS:
        val = unpacked_record.get(field, "").strip()
        if not val:
            msg = f"Mandatory field '{field}' is missing or empty."
            errors.append(msg)
            missing_fields.append(field)

    # 2. Validate RECORD NO (must be exactly 5 digits)
    rec_no = unpacked_record.get("RECORD NO", "").strip()
    if rec_no and not re.match(r'^\d{5}$', rec_no):
        msg = f"Invalid Record Number format '{rec_no}': must be exactly 5 digits."
        errors.append(msg)
        format_errors.append("RECORD NO")

    # 3. Validate EMAIL ADDRESS (must match email pattern and not contain arbitrary text/spaces)
    email = unpacked_record.get("EMAIL ADDRESS", "").strip()
    if email:
        if not EMAIL_PATTERN.match(email) or " " in email:
            msg = f"Invalid Email format '{email}'."
            errors.append(msg)
            format_errors.append("EMAIL ADDRESS")

    # 4. Validate ZIP_1 and ZIP_2 (must be exactly 5 digits)
    for zip_field in ("ZIP_1", "ZIP_2"):
        z_val = unpacked_record.get(zip_field, "").strip()
        if z_val and not re.match(r'^\d{5}$', z_val):
            msg = f"Invalid {zip_field} format '{z_val}': must be exactly 5 digits."
            errors.append(msg)
            format_errors.append(zip_field)

    # 5. Validate Phone 1 (must contain at least 7 digits, no arbitrary letters)
    ph = unpacked_record.get("PH_NO1", "").strip()
    if ph:
        if re.search(r'[a-zA-Z]', ph):
            msg = f"Invalid PH_NO1 '{ph}': contains alphabetical characters."
            errors.append(msg)
            format_errors.append("PH_NO1")
        else:
            digits = re.sub(r'\D', '', ph)
            if len(digits) < 7:
                msg = f"Invalid PH_NO1 '{ph}': must contain at least 7 digits."
                errors.append(msg)
                format_errors.append("PH_NO1")

    # 6. Validate Dates: D BIRTH, D_B_LIFE_ASSURE, DOB (must match MM/DD/YY or MM/DD/YYYY)
    for date_field in ("D BIRTH", "D_B_LIFE_ASSURE", "DOB"):
        dt = unpacked_record.get(date_field, "").strip()
        if dt and not re.match(r'^\d{1,2}/\d{1,2}/\d{2,4}$', dt):
            msg = f"Invalid date format for {date_field} '{dt}': must match MM/DD/YY(YY)."
            errors.append(msg)
            format_errors.append(date_field)

    # 7. Validate Sex fields: SEX_1, SEX_2 (must be MALE or FEMALE)
    for sex_field in ("SEX_1", "SEX_2"):
        sx = unpacked_record.get(sex_field, "").strip().upper()
        if sx and sx not in ("MALE", "FEMALE"):
            msg = f"Invalid sex format for {sex_field} '{sx}': must be MALE or FEMALE."
            errors.append(msg)
            format_errors.append(sex_field)

    # 8. Validate Blood Group: BLOOD GP (must be a valid blood group)
    bg = unpacked_record.get("BLOOD GP", "").strip().upper()
    if bg and bg not in ("A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"):
        msg = f"Invalid blood group '{bg}': must be A/B/AB/O with +/-."
        errors.append(msg)
        format_errors.append("BLOOD GP")

    # 8b. Validate Medicine Name: MEDICINE (must be a known medicine)
    med = unpacked_record.get("MEDICINE", "").strip().upper()
    if med:
        from modules.section_splitter import MED_TRIGGERS
        if med not in MED_TRIGGERS:
            msg = f"Invalid medicine name '{med}'."
            errors.append(msg)
            format_errors.append("MEDICINE")

    # 9. Validate Numeric/Int Fields: HEIGHT, WEIGHT, TABLETS
    for num_field in ("HEIGHT", "WEIGHT", "TABLETS"):
        num_val = unpacked_record.get(num_field, "").strip()
        if num_val and not re.match(r'^\d+$', num_val):
            msg = f"Invalid integer format for {num_field} '{num_val}'."
            errors.append(msg)
            format_errors.append(num_field)

    # 10. Validate Currency Fields: PILL RATE, COST, TOTAL AMT
    for curr_field in ("PILL RATE", "COST", "TOTAL AMT"):
        curr_val = unpacked_record.get(curr_field, "").strip()
        if curr_val and not re.match(r'^\$?\d+(\.\d{1,2})?$', curr_val.replace(",", "")):
            msg = f"Invalid currency format for {curr_field} '{curr_val}'."
            errors.append(msg)
            format_errors.append(curr_field)

    # 11. Validate Name Fields for arbitrary/corrupted values
    for name_field in ("CUSTOMER NAME", "BILLER NAME", "SHIPPER NAME", "NAME_P_HOLDER"):
        name_val = unpacked_record.get(name_field, "").strip()
        if name_val:
            # Hard noise tokens that always indicate corruption
            hard_noise = any(noise in name_val for noise in ("$", "|", "Not Available", "Available", "Availabie", "Not", ".png", ".jpg"))
            if hard_noise:
                msg = f"Arbitrary value in {name_field} '{name_val}'."
                errors.append(msg)
                format_errors.append(name_field)
                continue

            # Count digits in the name value
            digit_count = sum(1 for c in name_val if c.isdigit())
            alpha_words = [w for w in re.split(r'[\s\-_,.]+', name_val) if any(c.isalpha() for c in w)]

            # Allow up to 2 digits (common OCR misread: l→1, o→0, etc.) if name has ≥2 alpha words
            if digit_count > 2 or (digit_count > 0 and len(alpha_words) < 2):
                msg = f"Arbitrary value in {name_field} '{name_val}'."
                errors.append(msg)
                format_errors.append(name_field)


    # 12. Validate Address Fields for arbitrary/corrupted values
    for addr_field in ("RES_ADDRESS", "CITY_1", "CITY_2"):
        addr_val = unpacked_record.get(addr_field, "").strip()
        if addr_val:
            if any(noise in addr_val for noise in ("Not Available", "Available", "Availabie", "Not", ".png", ".jpg")) or "/" in addr_val:
                msg = f"Arbitrary value in {addr_field} '{addr_val}'."
                errors.append(msg)
                format_errors.append(addr_field)

    # 13. Check for raw OCR fallback fields (confidence 0.3)
    for field, val_struct in record.items():
        if isinstance(val_struct, dict) and val_struct.get("confidence") == 0.3:
            msg = f"Field '{field}' populated by raw OCR fallback for manual review."
            errors.append(msg)

    # Decide status: FAIL only when RECORD NO or CUSTOMER NAME is missing/empty.
    # Otherwise, if there are any validation errors, it is REVIEW. Otherwise, PASS.
    rec_no = unpacked_record.get("RECORD NO", "").strip()
    cust_name = unpacked_record.get("CUSTOMER NAME", "").strip()
    
    if not rec_no or not cust_name:
        status = "FAIL"
    elif len(errors) > 0:
        status = "REVIEW"
    else:
        status = "PASS"

    return {
        "status": status,
        "valid": status == "PASS",
        "errors": errors,
        "missing_fields": missing_fields,
        "format_errors": format_errors,
    }
