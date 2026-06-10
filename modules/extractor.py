import os
import re
from typing import Dict, Any, List
from modules.csv_schema import CSV_COLUMNS
from utils.regex_patterns import (
    RECORD_NO_PATTERN, EMAIL_PATTERN, ZIP_PATTERN, PHONE_PATTERN,
    DATE_PATTERN, CURRENCY_PATTERN, STAMP_PATTERN
)
from modules.normalizer import normalize_text

MULTI_WORD_CITIES = {"south jordan", "west jordan", "west valley city", "salt lake city", "san benito", "round rock"}

US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN",
    "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV",
    "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
    "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
}

def get_name_words(name: str) -> set:
    if not name:
        return set()
    words = set()
    for w in re.split(r'[\s_.\-@]+', name):
        if not w:
            continue
        words.add(w.lower().strip(".,()[]-"))
        camel_parts = re.findall(r'[A-Z][a-z]*', w)
        if len(camel_parts) > 1:
            for cp in camel_parts:
                words.add(cp.lower().strip(".,()[]-"))
    return {w for w in words if w}

def clean_ocr_text(text: str) -> List[str]:
    """
    Cleans raw OCR text, merges split email lines, and filters out headers/footers.
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    cleaned_lines = []
    for line in lines:
        lower_line = line.lower()
        if lower_line in ("available", "not available", "notavailable", "availabe", "avalabe", "avalable"):
            continue
        cleaned_lines.append(line)
        
    # Merge split emails
    i = 0
    final_lines = []
    while i < len(cleaned_lines):
        curr_line = cleaned_lines[i]
        if '@' in curr_line and not EMAIL_PATTERN.search(curr_line) and not any(curr_line.lower().endswith(ext) for ext in ('.com', '.net', '.org', '.co', '.net ', '.com ')):
            if i + 1 < len(cleaned_lines):
                next_line = cleaned_lines[i + 1]
                next_words = next_line.split()
                if next_words:
                    first_word = next_words[0]
                    combined = curr_line + first_word
                    if EMAIL_PATTERN.search(combined):
                        final_lines.append(combined)
                        remaining = " ".join(next_words[1:])
                        if remaining:
                            cleaned_lines[i + 1] = remaining
                        else:
                            i += 1
                        i += 1
                        continue
        final_lines.append(curr_line)
        i += 1
        
    return final_lines

def correct_customer_email(email_line: str) -> str:
    email_line = email_line.strip()
    if not email_line:
        return ""
        
    # Complete missing suffixes for known domains
    email_line = re.sub(r'@aol$', '@aol.com', email_line, flags=re.I)
    email_line = re.sub(r'@alltel$', '@alltel.net', email_line, flags=re.I)
    email_line = re.sub(r'@hotmail$', '@hotmail.com', email_line, flags=re.I)
    email_line = re.sub(r'@gmail$', '@gmail.com', email_line, flags=re.I)
    email_line = re.sub(r'@netzero$', '@netzero.net', email_line, flags=re.I)
    email_line = re.sub(r'@usa$', '@usa.net', email_line, flags=re.I)
        
    email_line_lower = email_line.lower()
    
    known_domains = {
        "netzero.net": "netzero.net",
        "usa.net": "usa.net",
        "aol.com": "aol.com",
        "acl.com": "aol.com",   # OCR: 'o' misread as 'c'
        "ac1.com": "aol.com",   # OCR: 'o' misread as '1'
        "a0l.com": "aol.com",   # OCR: 'o' misread as '0'
        "hotmail.com": "hotmail.com",
        "attgloba.net": "attgloba.net",
        "comcast.net": "comcast.net",
        "gmail.com": "gmail.com",
        "bellatlantic.net": "bellatlantic.net",
        "visto.com": "visto.com",
    }
    
    for dom_raw, dom_correct in known_domains.items():
        if dom_raw in email_line_lower:
            idx = email_line_lower.find(dom_raw)
            username = email_line[:idx].strip(" \t@_.-")
            if username:
                return f"{username}@{dom_correct}"
                
    return email_line

def find_record_no(text: str) -> str:
    """
    Scans all 5-digit candidates in text, scoring them to avoid state/ZIP code confusion.
    """
    joined_text = " ".join([l.strip() for l in text.split('\n') if l.strip()])
    
    candidates = re.findall(r'(?<!\d)\d{5}(?!\d)', text)
    if not candidates:
        return ""
    
    if len(candidates) == 1:
        return candidates[0]
        
    scored_candidates = []
    
    for cand in candidates:
        score = 100
        idx = joined_text.find(cand)
        if idx != -1:
            prefix = joined_text[:idx].strip()
            prefix_end = prefix[-15:] if len(prefix) > 15 else prefix
            
            # Check if preceded by a state code
            state_match = re.search(r'\b([A-Z]{2})\b\s*$', prefix_end, re.IGNORECASE)
            if state_match and state_match.group(1).upper() in US_STATES:
                score -= 80
                
            # Check if preceded by stamp prefix
            if re.search(r'[BSbs5]aX[\s_.\-]*[a-zA-Z0-9_]*\s*$', prefix_end, re.IGNORECASE):
                score -= 50
                
        # Find position in raw lines
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        for line in lines:
            if cand in line:
                l_idx = line.find(cand)
                if l_idx == 0:
                    score += 20
                l_suffix = line[l_idx+len(cand):].strip()
                if l_suffix and not re.match(r'^\d', l_suffix) and not any(k in l_suffix.upper() for k in ("@ ", "INC", "CO", "LANE", "DRIVE", "ROAD", "STREET")):
                    score += 10
                    
        scored_candidates.append((cand, score))
        
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return scored_candidates[0][0]

def extract_gender(text: str) -> str:
    """
    Returns 'FEMALE' or 'MALE' based on keywords.
    """
    text_upper = text.upper()
    if re.search(r'\b(FEMALE|FEMAL|FEMELE|FEMEL)\b', text_upper) or "FEMALE" in text_upper:
        return "FEMALE"
    if re.search(r'\b(MALE|MAL)\b', text_upper) or "MALE" in text_upper:
        return "MALE"
    return ""

def combat_ocr_phone_noise(phone_str: str) -> str:
    """
    Cleans area codes from common OCR errors.
    """
    if phone_str.startswith("(S"):
        phone_str = "(8" + phone_str[2:]
    return phone_str

def fill_missing_fields_with_raw_ocr(record: Dict[str, Any], ocr_text: str) -> Dict[str, Any]:
    """
    Fallback function to fill missing (empty) fields with the exact corresponding raw OCR line
    of their section. Assigns a confidence score of 0.3 to trigger review status.
    """
    updated = {k: dict(v) for k, v in record.items()}
    cust_name = updated.get("CUSTOMER NAME", {}).get("value", "")
    
    from modules.section_splitter import split_into_sections
    sections = split_into_sections(ocr_text, cust_name)
    
    cust_lines = [l.strip() for l in sections.get("customer", "").split('\n') if l.strip()]
    ship_lines = [l.strip() for l in sections.get("shipper", "").split('\n') if l.strip()]
    policy_lines = [l.strip() for l in sections.get("policy", "").split('\n') if l.strip()]
    med_lines = [l.strip() for l in sections.get("medication", "").split('\n') if l.strip()]
    
    field_mapping = {
        "RECORD NO": (cust_lines, 0),
        "CUSTOMER NAME": (cust_lines, 0),
        "BILLER NAME": (cust_lines, 0),
        "EMAIL ADDRESS": (cust_lines, 1),
        "RES_ADDRESS": (cust_lines, 2),
        "CITY_1": (cust_lines, 3),
        "STATE_1": (cust_lines, 3),
        "ZIP_1": (cust_lines, 3),
        "PH_NO1": (cust_lines, 3),
        "COUNTRY_1": (cust_lines, 4),
        "SEX_1": (cust_lines, 4),
        "D BIRTH": (cust_lines, 4),
        "HEIGHT": (cust_lines, 5),
        "WEIGHT": (cust_lines, 5),
        "BLOOD GP": (cust_lines, 5),
        
        "SHIPPER NAME": (ship_lines, 0),
        "CITY_2": (ship_lines, 1),
        "STATE_2": (ship_lines, 1),
        "ZIP_2": (ship_lines, 1),
        "COUNTRY_2": (ship_lines, 1),
        "PH_NO2": (ship_lines, 2),
        "ALCOHOLIC": (ship_lines, 2),
        "SMOKER": (ship_lines, 2),
        "PAST SURG": (ship_lines, 2),
        "DIABETIC": (ship_lines, 2),
        "ALLERGIESED": (ship_lines, 2),
        
        "POLICY NO": (policy_lines, 0),
        "D_B_LIFE_ASSURE": (policy_lines, 0),
        "P_INST": (policy_lines, 0),
        "NAME_P_HOLDER": (policy_lines, 1),
        "STM NAME": (policy_lines, 1),
        "STM CODE": (policy_lines, 1),
        "DOB": (policy_lines, 2),
        "SEX_2": (policy_lines, 2),
        
        "CARD NAME": (med_lines, 0),
        "MEDICINE": (med_lines, 0),
        "DOSAGE": (med_lines, 0),
        "TABLETS": (med_lines, 0),
        "PILL RATE": (med_lines, 1),
        "COST": (med_lines, 1),
        "SHIPPING CO": (med_lines, 1),
        "TOTAL AMT": (med_lines, 1)
    }
    
    for field, (lines_list, line_idx) in field_mapping.items():
        val = updated.get(field, {}).get("value", "").strip()
        if val:
            continue
        if not lines_list:
            continue
            
        actual_idx = min(line_idx, len(lines_list) - 1)
        raw_val = lines_list[actual_idx]
        
        if any(x in raw_val.lower() for x in ("not available", "available", "notavailable")):
            continue
            
        if len(raw_val) > 100:
            raw_val = raw_val[:100]
            
        # Type-specific fallback filters to avoid garbage formats
        if field in ("ZIP_1", "ZIP_2"):
            zip_match = ZIP_PATTERN.search(raw_val)
            if zip_match:
                raw_val = zip_match.group(1)
            else:
                continue
        elif field in ("D BIRTH", "DOB", "D_B_LIFE_ASSURE"):
            date_match = DATE_PATTERN.search(raw_val)
            if date_match:
                raw_val = date_match.group(1)
            else:
                continue
        elif field in ("SEX_1", "SEX_2"):
            raw_upper = raw_val.upper()
            if "FEMALE" in raw_upper:
                raw_val = "FEMALE"
            elif "MALE" in raw_upper:
                raw_val = "MALE"
            else:
                continue
        elif field == "EMAIL ADDRESS":
            email_match = EMAIL_PATTERN.search(raw_val)
            if email_match:
                raw_val = email_match.group(0)
            else:
                continue
        elif field == "NAME_P_HOLDER":
            raw_val = re.sub(r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b', '', raw_val)
            raw_val = re.sub(r'\$?\s*\b\d+\.\d{2}\b', '', raw_val)
            raw_val = re.sub(r'\b(?:SR|5R|sR|Sr)[_0-9a-zA-Z-]*\b', '', raw_val)
            raw_val = re.sub(r'\b[BSbs5]aX[\s_.\-]*[a-zA-Z0-9_\-.]*?\s*[a-zA-Z0-9]{5}\b', '', raw_val, flags=re.I)
            raw_val = re.sub(r'\S+@\S+|\S+(?:HOTMAIL|GMAIL|AOL|YAHOO|NETZERO|COMCAST)\S*', '', raw_val, flags=re.IGNORECASE)
            raw_val = re.sub(r'\S+\.(?:COM|NET|ORG|c[o0]m|ne[t1])\b', '', raw_val, flags=re.IGNORECASE)
            raw_val = re.sub(r'\d+', '', raw_val)
            raw_val = raw_val.strip(" ,.-_")
            if not raw_val or len([c for c in raw_val if c.isalpha()]) < 3:
                continue
            
        updated[field] = {"value": raw_val, "confidence": 0.3}
        
    return updated

def extract_fields(text: str, image_name: str = "", ocr_confidence: float = 0.90) -> Dict[str, Any]:
    """
    Extracts structured fields from the OCR text using a section-aware strategy
    and returns a dictionary of {"field": {"value": "...", "confidence": ...}}.
    """
    record = {col: {"value": "", "confidence": 0.0} for col in CSV_COLUMNS}
    record["IMAGE NAME"] = {"value": image_name, "confidence": 1.0}
    
    cleaned_lines = clean_ocr_text(text)
    if not cleaned_lines:
        return record
        
    # Apply generic, pattern-based OCR normalizations
    normalized_text, norm_factor = normalize_text("\n".join(cleaned_lines))
    normalized_lines = [l for l in normalized_text.split('\n') if l.strip()]
    joined_text = " ".join(normalized_lines)
    all_words = joined_text.split()
    
    def set_field(field_name: str, val: str, conf: float):
        # conf is the raw confidence already incorporating OCR-certainty in callers.
        # Apply normalization factor to reflect any corrections applied.
        final_conf = round(min(1.0, conf * (norm_factor if 'norm_factor' in locals() else 1.0)), 2)
        record[field_name] = {"value": val.strip(), "confidence": final_conf}
        
    # 1. RECORD NO & CUSTOMER NAME
    rec_no = find_record_no(text)
    set_field("RECORD NO", rec_no, 1.0 * ocr_confidence if rec_no else 0.0)
    
    # Preliminary name detection to split sections
    prelim_cust_name = ""
    if rec_no:
        for line in normalized_lines[:3]:
            if rec_no in line:
                prelim_cust_name = line.replace(rec_no, "").strip(" \t,-_")
                break
    if not prelim_cust_name and normalized_lines:
        prelim_cust_name = normalized_lines[0].strip()
        
    # Split into sections
    from modules.section_splitter import split_into_sections, edit_distance
    sections = split_into_sections(normalized_text, prelim_cust_name)
    
    cust_txt = sections["customer"]
    ship_txt = sections["shipper"]
    policy_txt = sections["policy"]
    med_txt = sections["medication"]
    
    cust_lines = [l.strip() for l in cust_txt.split('\n') if l.strip()]
    ship_lines = [l.strip() for l in ship_txt.split('\n') if l.strip()]
    policy_lines = [l.strip() for l in policy_txt.split('\n') if l.strip()]
    med_lines = [l.strip() for l in med_txt.split('\n') if l.strip()]
    
    cust_joined = " ".join(normalized_lines[:8])
    ship_joined = " ".join(ship_lines)
    policy_joined = " ".join(policy_lines)
    med_joined = " ".join(med_lines)
    
    # Customer Name
    cust_name = ""
    cust_name_conf = 0.0
    if rec_no:
        for line in cust_lines:
            if rec_no in line:
                cust_name = line.replace(rec_no, "").strip(" \t,-_")
                cust_name_conf = 1.0 * ocr_confidence
                break
    if not cust_name and cust_lines:
        cust_name = cust_lines[0]
        cust_name_conf = 0.8 * ocr_confidence
    if not cust_name and normalized_lines:
        cust_name = normalized_lines[0]
        cust_name_conf = 0.6 * ocr_confidence
        
    set_field("CUSTOMER NAME", cust_name, cust_name_conf)
    set_field("BILLER NAME", cust_name, cust_name_conf)
    
    # 2. EMAIL ADDRESS
    email = ""
    email_conf = 0.0
    emails = EMAIL_PATTERN.findall(cust_joined)
    if emails:
        email = correct_customer_email(emails[0])
        email_conf = 1.0 * ocr_confidence
    set_field("EMAIL ADDRESS", email, email_conf)
    
    # 3. Customer Address Fields: RES_ADDRESS, CITY_1, STATE_1, ZIP_1, COUNTRY_1, PH_NO1
    zip1 = ""
    zip1_conf = 0.0
    cust_joined_no_email = re.sub(r'\S+@\S+', '', cust_joined)
    zips_c = ZIP_PATTERN.findall(cust_joined_no_email)
    zips_c = [z for z in zips_c if z != rec_no]
    if zips_c:
        zip1 = zips_c[0]
        zip1_conf = 1.0 * ocr_confidence
    set_field("ZIP_1", zip1, zip1_conf)
    
    # Extract state, city, address preceding zip1
    state1 = ""
    state1_conf = 0.0
    city1 = ""
    city1_conf = 0.0
    res_addr = ""
    res_addr_conf = 0.0
    
    if zip1:
        words = cust_joined.split()
        try:
            zip1_idx = words.index(zip1)
            idx = zip1_idx - 1
            if idx >= 0:
                word = words[idx].strip("(),. ")
                word_clean = word.upper().replace('4', 'A').replace('1', 'A').replace('0', 'O').replace('2', 'Z').replace('5', 'S').replace('8', 'B')
                if word_clean == "ON":
                    word_clean = "OH"
                if len(word_clean) == 2 and word_clean in US_STATES:
                    state1 = word_clean
                    state1_conf = 0.9 * ocr_confidence
                    idx -= 1
                    
            # Check for multi-word city first
            city_parts = []
            is_multi = False
            for length in (3, 2):
                if idx - length + 1 >= 0:
                    cand_words = words[idx - length + 1:idx + 1]
                    cand_city = " ".join(cand_words).lower().strip(" ,.()")
                    if cand_city in MULTI_WORD_CITIES:
                        city_parts = cand_words
                        idx -= length
                        is_multi = True
                        break
            if not is_multi and idx >= 0:
                city_parts = [words[idx]]
                idx -= 1
                
            if city_parts:
                city1 = " ".join(city_parts).strip(" ,.()")
                city1_conf = 0.85 * ocr_confidence
                
            start_idx = 0
            email_found = False
            if email:
                for i in range(idx + 1):
                    if email in words[i] or '@' in words[i]:
                        start_idx = i + 1
                        email_found = True
                        break
            if not email_found and cust_name:
                cust_name_words = [w.lower().strip(".,()[]-") for w in cust_name.split() if w.strip()]
                last_cust_idx = -1
                for i in range(idx + 1):
                    w_clean = words[i].lower().strip(".,()[]-")
                    if w_clean in cust_name_words:
                        last_cust_idx = i
                if last_cust_idx != -1:
                    start_idx = last_cust_idx + 1
                    
            addr_parts = words[start_idx:idx + 1]
            if addr_parts and addr_parts[0] == rec_no:
                addr_parts = addr_parts[1:]
                
            if addr_parts:
                res_addr = " ".join(addr_parts).strip(" ,.()")
                res_addr_conf = 0.85 * ocr_confidence
        except ValueError:
            pass
            
    if not res_addr and len(cust_lines) >= 3:
        res_addr = cust_lines[1]
        res_addr_conf = 0.8 * ocr_confidence
        line2 = cust_lines[2]
        z_match = ZIP_PATTERN.search(line2)
        if z_match and not zip1:
            zip1 = z_match.group(1)
            set_field("ZIP_1", zip1, 0.8 * ocr_confidence)
        st_match = re.search(r'\b([A-Z]{2})\b', line2)
        if st_match and not state1:
            state1 = st_match.group(1)
            state1_conf = 0.8 * ocr_confidence
        if state1 and state1 in line2:
            city1 = line2.split(state1)[0].strip(" ,.()")
            city1_conf = 0.8 * ocr_confidence
            
    set_field("RES_ADDRESS", res_addr, res_addr_conf)
    set_field("CITY_1", city1, city1_conf)
    set_field("STATE_1", state1, state1_conf)
    
    # Phone 1
    ph1 = ""
    ph1_conf = 0.0
    ph1_m = PHONE_PATTERN.search(cust_joined)
    if ph1_m:
        ph1 = combat_ocr_phone_noise(ph1_m.group(0))
        ph1_conf = 1.0 * ocr_confidence
    set_field("PH_NO1", ph1, ph1_conf)
    
    # Country 1
    country1 = ""
    country1_conf = 0.0
    country1_match = re.search(r'(?:\b|[\d-])(US|USA)\b', cust_joined, re.IGNORECASE)
    if country1_match:
        country1 = country1_match.group(1).upper()
        country1_conf = 1.0 * ocr_confidence
    elif zip1:
        country1 = "US"
        country1_conf = 0.6 * ocr_confidence
    set_field("COUNTRY_1", country1, country1_conf)
    
    # Sex 1 & Dates
    sex1 = extract_gender(cust_joined)
    set_field("SEX_1", sex1, 1.0 * ocr_confidence if sex1 else 0.0)
    
    # Dates
    dates_found = DATE_PATTERN.findall(cust_joined)
    d_birth = dates_found[0] if dates_found else ""
    set_field("D BIRTH", d_birth, 1.0 * ocr_confidence if d_birth else 0.0)
    
    # Height, Weight, Blood GP
    bg = ""
    bg_conf = 0.0
    bg_line = ""
    # Find the line containing the blood group in first 8 lines
    for line in normalized_lines[:8]:
        bg_search = re.search(r'(?<!\w)(AB\s*[+\-]|[ABO0]\s*[+\-])(?!\w)', line, re.IGNORECASE)
        if bg_search:
            bg_raw = bg_search.group(1).upper().replace(' ', '')
            bg = re.sub(r'^0([+-])$', r'O\1', bg_raw)
            bg_conf = 1.0 * ocr_confidence
            bg_line = line
            break
            
    if not bg:
        # Standalone check: check for lone + or - or lone A/B/O/AB next to weight/height
        for line in normalized_lines[:8]:
            words = line.split()
            for idx, word in enumerate(words):
                word_upper = word.upper()
                if word_upper in ("+", "-"):
                    if idx > 0 and words[idx-1].isdigit() and 100 <= int(words[idx-1]) <= 250:
                        bg = "B" + word_upper  # Default B for + (e.g. B+)
                        bg_conf = 0.8 * ocr_confidence
                        bg_line = line
                        break
                elif word_upper in ("A", "B", "O", "AB"):
                    if idx > 0 and words[idx-1].isdigit() and 100 <= int(words[idx-1]) <= 250:
                        bg = word_upper + "+"  # Default positive sign
                        bg_conf = 0.8 * ocr_confidence
                        bg_line = line
                        break
            if bg:
                break
            
    bg_nums = []
    if bg_line:
        for word in bg_line.split():
            cleaned_word = re.sub(r'\D', '', word)
            if len(cleaned_word) == 3:
                num = int(cleaned_word)
                if 100 <= num <= 250:
                    bg_nums.append(str(num))
                    
    height = ""
    weight = ""
    if len(bg_nums) == 2:
        height = bg_nums[0]
        weight = bg_nums[1]
    elif len(bg_nums) == 1:
        weight = bg_nums[0]
        try:
            bg_idx = normalized_lines.index(bg_line)
            if bg_idx > 0:
                prev_line = normalized_lines[bg_idx - 1]
                prev_nums = []
                for word in prev_line.split():
                    cleaned_word = re.sub(r'\D', '', word)
                    if len(cleaned_word) == 3:
                        num = int(cleaned_word)
                        if 100 <= num <= 250:
                            prev_nums.append(str(num))
                if prev_nums:
                    height = prev_nums[-1]
        except ValueError:
            pass
            
    if not height or not weight:
        three_digit_nums = []
        for line in normalized_lines[:8]:
            if '@' in line or any(suffix in line.lower() for suffix in ["lane", "ln", "drive", "dr", "street", "st", "road", "rd", "court", "ct"]):
                continue
            for word in line.split():
                cleaned_word = re.sub(r'\D', '', word)
                if len(cleaned_word) == 3:
                    num = int(cleaned_word)
                    if 100 <= num <= 250 and str(num) != zip1:
                        three_digit_nums.append(str(num))
        if not height and three_digit_nums:
            height = three_digit_nums[0]
        if not weight and len(three_digit_nums) >= 2:
            weight = three_digit_nums[1]
            
    set_field("HEIGHT", height, 1.0 * ocr_confidence if height else 0.0)
    set_field("WEIGHT", weight, 1.0 * ocr_confidence if weight else 0.0)
    set_field("BLOOD GP", bg, bg_conf)
    
    # --- Shipper Section ---
    # Zip 2
    zip2 = ""
    zip2_conf = 0.0
    ship_joined_no_email = re.sub(r'\S+@\S+', '', ship_joined)
    zips_s = ZIP_PATTERN.findall(ship_joined_no_email)
    zips_s_filtered = [z for z in zips_s if z != zip1 and z != rec_no]
    if zips_s_filtered:
        zip2 = zips_s_filtered[0]
        zip2_conf = 1.0 * ocr_confidence
    set_field("ZIP_2", zip2, zip2_conf)
    
    # State 2, City 2, Shipper Name
    state2 = ""
    state2_conf = 0.0
    city2 = ""
    city2_conf = 0.0
    shipper_name = ""
    shipper_name_conf = 0.0
    
    if zip2:
        words_s = ship_joined.split()
        try:
            zip2_idx = words_s.index(zip2)
            idx = zip2_idx - 1
            if idx >= 0:
                word = words_s[idx].strip("(),. ")
                word_clean = word.upper().replace('4', 'A').replace('1', 'A').replace('0', 'O').replace('2', 'Z').replace('5', 'S').replace('8', 'B')
                if word_clean == "ON":
                    word_clean = "OH"
                if len(word_clean) == 2 and word_clean in US_STATES:
                    state2 = word_clean
                    state2_conf = 0.9 * ocr_confidence
                    idx -= 1
                    
            pre_words = words_s[:idx + 1]
            city_parts_s = []
            is_multi = False
            n_words = len(pre_words)
            for length in (3, 2):
                if n_words >= length:
                    cand_words = pre_words[n_words - length:]
                    cand_city = " ".join(cand_words).lower().strip(" ,.()")
                    if cand_city in MULTI_WORD_CITIES:
                        city_parts_s = cand_words
                        pre_words = pre_words[:n_words - length]
                        is_multi = True
                        break
            if not is_multi and n_words >= 1:
                city_parts_s = [pre_words[-1]]
                pre_words = pre_words[:-1]
                
            if city_parts_s:
                city2 = " ".join(city_parts_s).strip(" ,.()")
                city2_conf = 0.85 * ocr_confidence
                
            cust_name_words = get_name_words(cust_name) if cust_name else set()
            city2_words = {w.lower().strip(".,()[]-") for w in re.split(r'[\s_.\-@]+', city2) if w.strip()} if city2 else set()
            state2_words = {state2.lower().strip()} if state2 else set()
            shipper_parts = []
            for w in pre_words:
                w_clean = w.strip(".,()[]-")
                w_lower = w_clean.lower()
                if w_lower in cust_name_words or (len(w_lower) >= 4 and any(edit_distance(w_lower, cw) <= 1 for cw in cust_name_words)):
                    continue
                if w_lower in city2_words or w_lower in state2_words:
                    continue
                if w_clean.isdigit() and len(w_clean) == 3:
                    continue
                if re.match(r'^(AB[+-]|[ABO0][+-])$', w_clean, re.I) or w_clean in ("+", "-"):
                    continue
                shipper_parts.append(w)
                
            if shipper_parts:
                shipper_name = " ".join(shipper_parts).strip(" ,.()")
                shipper_name_conf = 0.9 * ocr_confidence
        except ValueError:
            pass
            
    if not shipper_name and len(ship_lines) >= 1:
        if cust_name and len(ship_lines[0]) >= 3 and (ship_lines[0].lower() in cust_name.lower() or cust_name.lower() in ship_lines[0].lower()):
            if len(ship_lines) >= 2:
                shipper_name = ship_lines[1]
                shipper_name_conf = 0.8 * ocr_confidence
        else:
            shipper_name = ship_lines[0]
            shipper_name_conf = 0.9 * ocr_confidence
            
    set_field("SHIPPER NAME", shipper_name, shipper_name_conf)
    set_field("CITY_2", city2, city2_conf)
    set_field("STATE_2", state2, state2_conf)
    
    # Country 2
    country2 = ""
    country2_conf = 0.0
    country2_match = re.search(r'(?:\b|[\d-])(US|USA)\b', ship_joined, re.IGNORECASE)
    if country2_match:
        country2 = country2_match.group(1).upper()
        country2_conf = 1.0 * ocr_confidence
    elif zip2:
        country2 = "US"
        country2_conf = 0.6 * ocr_confidence
    set_field("COUNTRY_2", country2, country2_conf)
    
    # Phone 2
    ph2 = ""
    ph2_conf = 0.0
    ph2_m = PHONE_PATTERN.search(ship_joined)
    if ph2_m:
        ph2 = combat_ocr_phone_noise(ph2_m.group(0))
        ph2_conf = 1.0 * ocr_confidence
    set_field("PH_NO2", ph2, ph2_conf)
    
    # Medical flags
    yn_list = []
    lines_to_scan = list(ship_lines)
    if policy_lines:
        lines_to_scan.append(policy_lines[0])
        
    for line in lines_to_scan:
        line_up = line.upper()
        for word in line_up.split():
            w_clean = re.sub(r'[^A-Z0-9]', '', word)
            if w_clean in ("YES", "YE5", "Y1S"):
                yn_list.append("YES")
            elif w_clean in ("NO", "N0", "NOI", "INO", "N0I"):
                yn_list.append("NO")
                
    alcoholic = yn_list[0] if len(yn_list) >= 1 else ""
    smoker = yn_list[1] if len(yn_list) >= 2 else ""
    past_surg = yn_list[2] if len(yn_list) >= 3 else ""
    diabetic = yn_list[3] if len(yn_list) >= 4 else ""
    allergies = yn_list[4] if len(yn_list) >= 5 else ""
    
    set_field("ALCOHOLIC", alcoholic, 1.0 * ocr_confidence if alcoholic else 0.0)
    set_field("SMOKER", smoker, 1.0 * ocr_confidence if smoker else 0.0)
    set_field("PAST SURG", past_surg, 1.0 * ocr_confidence if past_surg else 0.0)
    set_field("DIABETIC", diabetic, 1.0 * ocr_confidence if diabetic else 0.0)
    set_field("ALLERGIESED", allergies, 1.0 * ocr_confidence if allergies else 0.0)
    
    # --- Policy Section ---
    # POLICY NO
    policy_no = ""
    policy_no_conf = 0.0
    pol_m = re.search(r'\b(SR|5R|sR|Sr)[_0-9a-zA-Z-]*\b', policy_joined)
    if pol_m:
        policy_no = pol_m.group(0).strip("(),.-_")
        policy_no_conf = 1.0 * ocr_confidence
    elif pol_m := re.search(r'\b(SR|5R|sR|Sr)[_0-9a-zA-Z-]*\b', joined_text):
        policy_no = pol_m.group(0).strip("(),.-_")
        policy_no_conf = 0.7 * ocr_confidence
    set_field("POLICY NO", policy_no, policy_no_conf)
    
    # Policy dates
    pol_dates = DATE_PATTERN.findall(policy_joined)
    db_life = pol_dates[0] if pol_dates else ""
    dob = pol_dates[1] if len(pol_dates) >= 2 else (pol_dates[0] if pol_dates else "")
    
    set_field("D_B_LIFE_ASSURE", db_life, 1.0 * ocr_confidence if db_life else 0.0)
    set_field("DOB", dob, 1.0 * ocr_confidence if dob else 0.0)
    
    # Premium Installment
    p_inst = ""
    p_inst_conf = 0.0
    pol_curr = CURRENCY_PATTERN.findall(policy_joined)
    if pol_curr:
        p_inst = f"${float(pol_curr[0].replace('$', '').replace(',', '')):.2f}"
        p_inst_conf = 1.0 * ocr_confidence
    set_field("P_INST", p_inst, p_inst_conf)
    
    # Stamp details
    stm_name = ""
    stm_code = ""
    stamp_conf = 0.0
    stamp_match = STAMP_PATTERN.search(policy_joined)
    if not stamp_match:
        stamp_match = STAMP_PATTERN.search(joined_text)
    if stamp_match:
        full_stamp = stamp_match.group(0)
        stm_name = full_stamp
        stamp_conf = 1.0 * ocr_confidence
        raw_code = stamp_match.group(3)
        trans_table = str.maketrans({
            'I': '1', 'l': '1', 'i': '1', 'O': '0', 'o': '0', 'S': '5', 's': '5',
            'Z': '2', 'z': '2', 'B': '8', 'b': '8', 'G': '6', 'g': '6', 'T': '7', 't': '7',
            'A': '4', 'a': '4'
        })
        stm_code = raw_code.translate(trans_table)
    set_field("STM NAME", stm_name, stamp_conf)
    set_field("STM CODE", stm_code, stamp_conf)
    
    # Policy holder name
    p_holder = ""
    p_holder_conf = 0.0
    if stamp_match:
        prefix = stamp_match.group(1)
        name_part = stamp_match.group(2)
        code_part = stamp_match.group(3)
        asp_val = p_inst.replace("$", "")
        asp_escaped = re.escape(asp_val)
        stamp_escaped = rf'{re.escape(prefix)}[\s_.\-]*{re.escape(name_part)}[\s_.\-]*{re.escape(code_part)}'
        
        name_match = None
        if asp_val:
            name_match = re.search(rf'{asp_escaped}\s+(.*?)\s+(?:\S+@\S+\s+)?{stamp_escaped}', policy_joined, re.IGNORECASE | re.DOTALL)
        if not name_match and db_life:
            dbl_escaped = re.escape(db_life)
            name_match = re.search(rf'{dbl_escaped}\s+(.*?)\s+(?:\S+@\S+\s+)?{stamp_escaped}', policy_joined, re.IGNORECASE | re.DOTALL)
        if not name_match:
            if asp_val:
                name_match = re.search(rf'{asp_escaped}\s+(.*?)\s+(?:\S+@\S+\s+)?{stamp_escaped}', joined_text, re.IGNORECASE | re.DOTALL)
            if not name_match and db_life:
                dbl_escaped = re.escape(db_life)
                name_match = re.search(rf'{dbl_escaped}\s+(.*?)\s+(?:\S+@\S+\s+)?{stamp_escaped}', joined_text, re.IGNORECASE | re.DOTALL)
                
        if name_match:
            raw_name = name_match.group(1).strip()
            # Clean up dates, currencies, stamps, emails
            raw_name = re.sub(r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b', '', raw_name).strip()
            raw_name = re.sub(r'\$?\s*\b\d+\.\d{2}\b', '', raw_name).strip()
            raw_name = re.sub(r'\b(?:SR|5R|sR|Sr)[_0-9a-zA-Z-]*\b', '', raw_name).strip()
            raw_name = re.sub(r'\b[BSbs5]aX[\s_.\-]*[a-zA-Z0-9_\-.]*?\s*[a-zA-Z0-9]{5}\b', '', raw_name, flags=re.I).strip()
            
            raw_name = re.sub(r'\s+[A-Z]{2,3}$', '', raw_name).strip()
            emails2 = EMAIL_PATTERN.findall(raw_name)
            if emails2:
                raw_name = raw_name.replace(emails2[0], "").strip()
            raw_name = re.sub(r'\S+@\S+|\S+(?:HOTMAIL|GMAIL|AOL|YAHOO|NETZERO|COMCAST)\S*', '', raw_name, flags=re.IGNORECASE).strip()
            raw_name = re.sub(r'\S+\.(?:COM|NET|ORG|c[o0]m|ne[t1])\b', '', raw_name, flags=re.IGNORECASE).strip()
            name_words = raw_name.split()
            if len(name_words) > 4:
                name_words = name_words[:4]
            raw_name = " ".join(name_words).strip()
            raw_name = re.sub(r'\s+', ' ', raw_name)
            raw_name = raw_name.strip(" ,.-_$")
            
            if raw_name and len([c for c in raw_name if c.isalpha()]) >= 3 and not re.match(r'^\d', raw_name):
                p_holder = raw_name
                p_holder_conf = 1.0 * ocr_confidence if name_match.string == policy_joined else 0.7 * ocr_confidence
                
    set_field("NAME_P_HOLDER", p_holder, p_holder_conf)
    
    # Sex 2
    sex2 = ""
    sex2_conf = 0.0
    sex2_found = False
    if stamp_match:
        full_stamp = stamp_match.group(0)
        stamp_pos = policy_joined.find(full_stamp)
        if stamp_pos != -1:
            _pst = policy_joined[stamp_pos + len(full_stamp):].strip()
            sex2_match_ps = re.search(r'(\d{2}/\d{2}/\d{2,4})\s+(FEMALE|MALE)\b', _pst, re.IGNORECASE)
            if sex2_match_ps:
                sex2 = sex2_match_ps.group(2).upper()
                sex2_conf = 1.0 * ocr_confidence
                sex2_found = True
    if not sex2_found and dob:
        dob_escaped = re.escape(dob)
        sex2_match = re.search(rf'{dob_escaped}\s+(FEMALE|MALE)\b', policy_joined, re.IGNORECASE)
        if sex2_match:
            sex2 = sex2_match.group(1).upper()
            sex2_conf = 1.0 * ocr_confidence
            sex2_found = True
    if not sex2_found:
        if "FEMALE" in policy_joined.upper():
            sex2 = "FEMALE"
            sex2_conf = 0.8 * ocr_confidence
        elif "MALE" in policy_joined.upper():
            sex2 = "MALE"
            sex2_conf = 0.8 * ocr_confidence
    set_field("SEX_2", sex2, sex2_conf)
    
    # --- Medication Section ---
    # Card Name
    card_name = ""
    card_conf = 0.0
    card_patterns = [
        ("American Express", r'\bAmerican\s*Express(?:\b|(?=[A-Z]))'),
        ("Visa",            r'\b(?:Visa|V1sa|Viga|Vi5a)\b'),
        ("MasterCard",      r'\bMaster\s*(?:Card|C4rd|Cerd)\b'),
        ("Discover",        r'\b(?:Dis\s*cover|Disc\s*0ver|Die\s*cover)\b'),
        ("Retail",          r'\b(?:Retail|Retall|Retal)\b'),
        ("Gas",             r'\bGas\b'),
        ("Bank",            r'\bBank\b'),
        ("Other",           r'\b(?:Others?|0thers?|Othar|Othe[r]?|0the[r]?|Othes)\b'),
    ]
    for card, pat in card_patterns:
        if re.search(pat, med_joined, re.IGNORECASE):
            card_name = card
            card_conf = 1.0 * ocr_confidence
            break
    if not card_name:
        for card, pat in card_patterns:
            if re.search(pat, joined_text, re.IGNORECASE):
                card_name = card
                card_conf = 0.7 * ocr_confidence
                break
    set_field("CARD NAME", card_name, card_conf)
    
    # Medicine
    KNOWN_MEDICINES = [
        "PHENTERMINE", "DIAZEPAM", "XANAX", "VALIUM", "ADIPEX", "MERIDIA",
        "BONTRIL", "IONAMIN", "FASTIN", "PONDIMIN", "REDUX", "ZANTREX",
        "AMBIEN", "AMBIENR", "LUNESTA", "HALCION", "RESTORIL", "SONATA",
        "CELEBREX", "VIOXX", "MORPHINE", "OXYCONTIN", "PERCOCET", "VICODIN",
        "TYLENOL", "IBUPROFEN", "ASPIRIN", "NAPROXEN", "DIDREX", "ZOLOFT",
        "CLONAZEPAM", "CLONAZAPAM",
    ]
    med_text_processed = med_joined
    for med in KNOWN_MEDICINES:
        med_text_processed = re.sub(rf'({med})(\d+(?:\.\d+)?\s*MG)', r'\1 \2', med_text_processed, flags=re.IGNORECASE)
    med_text_processed = re.sub(r'(\$\d+\.\d{2})(\$)', r'\1 \2', med_text_processed)
    
    medication = ""
    med_conf = 0.0
    if card_name:
        card_escaped = re.escape(card_name)
        med_match = re.search(rf'{card_escaped}\s+(?:[A-Za-z]{{1,15}}\s+){{0,2}}([A-Za-z]{{4,20}})\b', med_text_processed, re.IGNORECASE)
        if med_match:
            candidate = med_match.group(1).upper()
            if candidate in KNOWN_MEDICINES or re.match(r'^[A-Z]{4,}$', candidate):
                medication = candidate
                med_conf = 1.0 * ocr_confidence
    if not medication:
        ps_upper = med_text_processed.upper()
        for med in KNOWN_MEDICINES:
            if med in ps_upper:
                medication = med
                med_conf = 1.0 * ocr_confidence
                break
    if not medication:
        ps_upper_g = joined_text.upper()
        for med in KNOWN_MEDICINES:
            if med in ps_upper_g:
                medication = med
                med_conf = 0.7 * ocr_confidence
                break
    set_field("MEDICINE", medication, med_conf)
    
    # Dosage
    dosage = ""
    dosage_conf = 0.0
    dosage_match = re.search(r'\b(\d*\.?\d+\s*MG)\b', med_text_processed, re.IGNORECASE)
    if dosage_match:
        dosage_raw = dosage_match.group(1).strip()
        dosage_raw = re.sub(r'\s+', ' ', dosage_raw).upper()
        if dosage_raw.startswith('.'):
            dosage_raw = '0' + dosage_raw
        dosage = dosage_raw
        dosage_conf = 1.0 * ocr_confidence
    else:
        partial_mg = re.search(r'([\d.]+)\s*(?=MG\b)', med_text_processed, re.IGNORECASE)
        if partial_mg:
            val = partial_mg.group(1).lstrip('.')
            if val.startswith('.'):
                val = '0' + val
            dosage = f"{val}MG"
            dosage_conf = 0.9 * ocr_confidence
            
    if not dosage:
        dosage_match_g = re.search(r'\b(\d*\.?\d+\s*MG)\b', joined_text, re.IGNORECASE)
        if dosage_match_g:
            dosage_raw = dosage_match_g.group(1).strip()
            dosage_raw = re.sub(r'\s+', ' ', dosage_raw).upper()
            if dosage_raw.startswith('.'):
                dosage_raw = '0' + dosage_raw
            dosage = dosage_raw
            dosage_conf = 0.7 * ocr_confidence
    set_field("DOSAGE", dosage, dosage_conf)
    
    # Tablets
    tablets = ""
    tablets_conf = 0.0
    tablets_match = re.search(r'(?:\b|(?<=MG))(\d{2,3})\b', med_text_processed, re.IGNORECASE)
    if tablets_match:
        tablets = tablets_match.group(1)
        tablets_conf = 1.0 * ocr_confidence
    else:
        tablets_match_g = re.search(r'(?:\b|(?<=MG))(\d{2,3})\b', joined_text, re.IGNORECASE)
        if tablets_match_g:
            tablets = tablets_match_g.group(1)
            tablets_conf = 0.7 * ocr_confidence
    set_field("TABLETS", tablets, tablets_conf)
    
    # Prices
    pill_rate = ""
    cost = ""
    shipping = ""
    total = ""
    prices_conf = 0.0
    post_stamp_cleaned = re.sub(r'\b\d+(?:\.\d+)?\s*MG\b', '', med_text_processed, flags=re.IGNORECASE)
    post_currencies = CURRENCY_PATTERN.findall(post_stamp_cleaned)
    curr_vals = []
    for c in post_currencies:
        try:
            curr_vals.append(float(c.replace("$", "").replace(",", "")))
        except ValueError:
            pass
            
    if len(curr_vals) >= 4:
        pill_rate = f"${curr_vals[0]:.2f}"
        cost = f"${curr_vals[1]:.2f}"
        shipping = f"${curr_vals[2]:.2f}"
        total = f"${curr_vals[3]:.2f}"
        prices_conf = 1.0 * ocr_confidence
    elif len(curr_vals) == 3:
        pill_rate = f"${curr_vals[0]:.2f}"
        cost = f"${curr_vals[1]:.2f}"
        shipping = "$20.00"
        total = f"${curr_vals[2]:.2f}"
        prices_conf = 0.8 * ocr_confidence
    elif len(curr_vals) == 2:
        cost = f"${curr_vals[0]:.2f}"
        shipping = "$20.00"
        total = f"${curr_vals[1]:.2f}"
        prices_conf = 0.8 * ocr_confidence
    elif len(curr_vals) == 1:
        total = f"${curr_vals[0]:.2f}"
        prices_conf = 0.7 * ocr_confidence
        
    set_field("PILL RATE", pill_rate, prices_conf)
    set_field("COST", cost, prices_conf)
    set_field("SHIPPING CO", shipping, prices_conf)
    set_field("TOTAL AMT", total, prices_conf)
    
    return record
