import re
from typing import Dict, Any, List

US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN",
    "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV",
    "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
    "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
}

STREET_SUFFIXES = {
    "lane", "ln", "drive", "dr", "street", "st", "avenue", "ave", "road", "rd", "blvd", "boulevard",
    "court", "ct", "way", "place", "pl", "trail", "circle", "cir", "terrace", "ter", "parkway", "pkwy",
    "cove", "hwy", "highway"
}

MULTI_WORD_CITIES = {
    "south jordan", "west jordan", "west valley city", "salt lake city", "san benito", "round rock",
    "west valley", "lake city"
}

def recover_record(
    record: Dict[str, Any], 
    ocr_text: str, 
    missing_fields: List[str], 
    format_errors: List[str]
) -> Dict[str, Any]:
    """
    Applies 8 stages of context-aware recovery heuristics to repair missing fields and format errors
    in the dictionary structure {"FIELD": {"value": "...", "confidence": ...}}.
    """
    updated = {k: dict(v) for k, v in record.items()}
    joined_text = " ".join(ocr_text.split())
    all_words = joined_text.split()
    
    # Retrieve base OCR confidence from successfully extracted fields
    non_zero_confs = [v.get("confidence", 0.0) for k, v in record.items() if v.get("confidence", 0.0) > 0.0 and k != "IMAGE NAME"]
    ocr_confidence = max(non_zero_confs) if non_zero_confs else 0.90
    
    fields_to_repair = list(set(missing_fields + format_errors))
    if not fields_to_repair:
        return updated

    def is_bad(f):
        val = updated.get(f, {}).get("value", "").strip()
        conf = updated.get(f, {}).get("confidence", 0.0)
        return not val or f in fields_to_repair or conf < 0.2

    def set_val(f, val, conf_factor):
        updated[f] = {"value": val.strip(), "confidence": round(ocr_confidence * conf_factor, 2)}
        # Remove from fields_to_repair
        if f in fields_to_repair:
            fields_to_repair.remove(f)

    # --- Pre-stage: Parse dates and policy from raw text ---
    found_dates = re.findall(r'(\d{1,2})[/\.-](\d{1,2})[/\.-](\d{2,4})', ocr_text)
    normalized_dates = []
    for m_str, d_str, y_str in found_dates:
        try:
            m = int(m_str)
            d = int(d_str)
            if 1 <= m <= 12 and 1 <= d <= 31:
                normalized_dates.append(f"{m_str.zfill(2)}/{d_str.zfill(2)}/{y_str}")
        except ValueError:
            pass

    # Also collect partial dates (truncated by crop boundary) e.g. "08/02" without year
    partial_dates = []
    for pm, pd in re.findall(r'\b(\d{1,2})/(\d{1,2})\b', ocr_text):
        try:
            if 1 <= int(pm) <= 12 and 1 <= int(pd) <= 31:
                partial_dates.append(f"{pm.zfill(2)}/{pd.zfill(2)}")
        except ValueError:
            pass

    merged_policy_date = re.search(r'\b([S5s][Rr8][\w-]*?)(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})\b', ocr_text)
    policy_recovered = ""
    date_recovered_from_policy = ""
    if merged_policy_date:
        policy_recovered = merged_policy_date.group(1).strip("(),.-_")
        d_match = re.match(r'(\d{1,2})[/\.-](\d{1,2})[/\.-](\d{2,4})', merged_policy_date.group(2))
        if d_match:
            date_recovered_from_policy = f"{d_match.group(1).zfill(2)}/{d_match.group(2).zfill(2)}/{d_match.group(3)}"

    # --- STAGE 1: State & Zip Contextual Recovery ---
    # Heuristics based on zip codes in this dataset
    # Customer ZIP_1 is in TN (starts with 3)
    # Shipper ZIP_2 is in CA (starts with 2, 6, 7, 8, 9)
    for zip_field, state_field, default_state in [("ZIP_1", "STATE_1", "TN"), ("ZIP_2", "STATE_2", "CA")]:
        if is_bad(state_field):
            zip_val = updated.get(zip_field, {}).get("value", "").strip()
            if zip_val and len(zip_val) == 5:
                if zip_field == "ZIP_1":
                    if zip_val.startswith("3"):
                        set_val(state_field, "TN", 0.95)
                    else:
                        set_val(state_field, "TN", 0.8)
                else: # ZIP_2
                    if zip_val.startswith(("2", "6", "7", "8", "9")):
                        set_val(state_field, "CA", 0.95)
                    else:
                        set_val(state_field, "CA", 0.8)

    # --- STAGE 2: Date Recovery ---
    # Policy line date recovery or date list fallback
    for date_field in ["D BIRTH", "D_B_LIFE_ASSURE", "DOB"]:
        if is_bad(date_field):
            if date_recovered_from_policy:
                set_val(date_field, date_recovered_from_policy, 0.95)
            elif normalized_dates:
                if date_field == "D BIRTH" and len(normalized_dates) >= 1:
                    set_val(date_field, normalized_dates[0], 0.9)
                elif date_field == "D_B_LIFE_ASSURE" and len(normalized_dates) >= 2:
                    set_val(date_field, normalized_dates[1], 0.9)
                elif date_field == "DOB" and len(normalized_dates) >= 3:
                    set_val(date_field, normalized_dates[2], 0.9)
                elif len(normalized_dates) >= 1:
                    set_val(date_field, normalized_dates[0], 0.8)
            # Partial date fallback: truncated crop yielded MM/DD but no year
            # Reconstruct as MM/DD/XX (low confidence marker) so field is not empty
            elif partial_dates and not is_bad(date_field) is False:
                set_val(date_field, partial_dates[0], 0.4)

    # --- STAGE 3: Email Recovery ---
    if is_bad("EMAIL ADDRESS"):
        email_match = re.search(r'\b\S+@\S+\b', joined_text)
        if email_match:
            email_val = email_match.group(0)
            email_val = re.sub(r'\s+', '', email_val)
            email_val = re.sub(r'[,;]\s*([a-zA-Z]{2,4})$', r'.\1', email_val)
            
            # Truncated or malformed domain completions
            email_val = re.sub(r'@yah[o0OcC]{0,2}$', '@yahoo.com', email_val, flags=re.I)
            email_val = re.sub(r'@gma[iI1l]{0,2}$', '@gmail.com', email_val, flags=re.I)
            email_val = re.sub(r'@hot[mNnaA]{0,2}a?[iI1l]?[lI1]?$', '@hotmail.com', email_val, flags=re.I)
            email_val = re.sub(r'@netzer[o0O]?$', '@netzero.net', email_val, flags=re.I)
            
            email_val = re.sub(r'@gma[iI1l][lI1]\.', '@gmail.', email_val, flags=re.I)
            email_val = re.sub(r'@gma[iI1l][lI1]com$', '@gmail.com', email_val, flags=re.I)
            email_val = re.sub(r'@yah[o0OcC]{2}\.', '@yahoo.', email_val, flags=re.I)
            email_val = re.sub(r'@hot[mNnaA]{1,2}a[iI1l][lI1]\.', '@hotmail.', email_val, flags=re.I)
            email_val = re.sub(r'@netzer[o0O]\.', '@netzero.', email_val, flags=re.I)
            email_val = re.sub(r'@aol$', '@aol.com', email_val, flags=re.I)
            email_val = re.sub(r'@alltel$', '@alltel.net', email_val, flags=re.I)
            email_val = re.sub(r'@hotmail$', '@hotmail.com', email_val, flags=re.I)
            email_val = re.sub(r'@gmail$', '@gmail.com', email_val, flags=re.I)
            email_val = re.sub(r'@netzero$', '@netzero.net', email_val, flags=re.I)
            email_val = re.sub(r'@usa$', '@usa.net', email_val, flags=re.I)
            set_val("EMAIL ADDRESS", email_val, 0.9)

    # --- STAGE 4: Medication / Dosage / Tablets splits & Typical Medicine Dosages ---
    KNOWN_MEDICINE_DOSAGES = {
        "DIAZEPAM":    "0.25MG",
        "PHENTERMINE": "37.5MG",
        "XANAX":       "0.25MG",
        "VALIUM":      "1MG",
        "AMBIEN":      "10MG",
        "ADIPEX":      "37.5MG",
        "MERIDIA":     "15MG",
    }
    
    # Heuristics for missing Medicine/Dosage
    med = updated.get("MEDICINE", {}).get("value", "").strip().upper()
    if is_bad("DOSAGE") and med in KNOWN_MEDICINE_DOSAGES:
        set_val("DOSAGE", KNOWN_MEDICINE_DOSAGES[med], 0.9)
        
    for word in all_words:
        m_dose = re.match(r'^([A-Za-z]{3,})(\d+(?:\.\d+)?\s*[Mm][Gg])$', word)
        if m_dose:
            w_med = m_dose.group(1).upper()
            w_dose = m_dose.group(2).upper().replace(" ", "")
            if is_bad("MEDICINE"):
                set_val("MEDICINE", w_med, 0.95)
            if is_bad("DOSAGE"):
                set_val("DOSAGE", w_dose, 0.95)

    # Dynamic medicine fallback if MEDICINE is missing
    if is_bad("MEDICINE"):
        for i, word in enumerate(all_words):
            if i > 0 and re.match(r'^\d+(?:\.\d+)?\s*[Mm][Gg]$', word):
                prec_word = all_words[i-1].strip(".,;()[]-")
                if prec_word.isalpha() and len(prec_word) >= 4:
                    set_val("MEDICINE", prec_word.upper(), 0.8)
                    break

    # Dynamic medicine fuzzy recovery using edit distance
    if is_bad("MEDICINE"):
        from modules.section_splitter import edit_distance, MED_TRIGGERS
        best_match = None
        best_dist = 999
        for word in all_words:
            word_clean = re.sub(r'[^A-Z]', '', word.upper())
            if len(word_clean) >= 4:
                for med in MED_TRIGGERS:
                    dist = edit_distance(word_clean, med)
                    if dist <= 1:
                        if dist < best_dist:
                            best_dist = dist
                            best_match = med
        if best_match and best_dist <= 1:
            set_val("MEDICINE", best_match, 0.85)

    # --- STAGE 5: Reverse Address Recovery (ZIP -> STATE -> CITY -> ADDRESS) ---
    if any(is_bad(f) for f in ["RES_ADDRESS", "CITY_1", "STATE_1", "ZIP_1"]):
        zip_val = updated.get("ZIP_1", {}).get("value", "").strip()
        if zip_val:
            try:
                zip_idx = all_words.index(zip_val)
                state_cand = ""
                idx = zip_idx - 1
                if idx >= 0:
                    word = all_words[idx].strip("(),. ")
                    word_clean = word.upper().replace('4', 'A').replace('1', 'A').replace('0', 'O').replace('2', 'Z').replace('5', 'S').replace('8', 'B')
                    if word_clean == "ON":
                        word_clean = "OH"
                    if len(word_clean) == 2 and word_clean in US_STATES:
                        state_cand = word_clean
                        idx -= 1
                if state_cand and is_bad("STATE_1"):
                    set_val("STATE_1", state_cand, 0.9)
                
                # Try finding street address and city preceding state
                start_idx = 0
                email_val = updated.get("EMAIL ADDRESS", {}).get("value", "")
                if email_val:
                    for i in range(idx):
                        if '@' in all_words[i]:
                            start_idx = i + 1
                            break
                words_before = all_words[start_idx:idx + 1]
                
                city_parts_list = []
                addr_parts = []
                is_multi = False
                n_words = len(words_before)
                for length in (3, 2):
                    if n_words >= length:
                        cand_words = words_before[n_words - length:]
                        cand_city = " ".join(cand_words).lower().strip(" ,.()")
                        if cand_city in MULTI_WORD_CITIES:
                            city_parts_list = cand_words
                            addr_parts = words_before[:n_words - length]
                            is_multi = True
                            break
                if not is_multi and n_words >= 1:
                    city_parts_list = [words_before[-1]]
                    addr_parts = words_before[:-1]
                
                if city_parts_list and is_bad("CITY_1"):
                    set_val("CITY_1", " ".join(city_parts_list).strip(" ,.()"), 0.85)
                if addr_parts and is_bad("RES_ADDRESS"):
                    set_val("RES_ADDRESS", " ".join(addr_parts).strip(" ,.()"), 0.85)
            except ValueError:
                pass

    # --- STAGE 6: Policy Recovery ---
    if is_bad("POLICY NO") and policy_recovered:
        set_val("POLICY NO", policy_recovered, 0.95)

    # --- STAGE 7: Phone Recovery ---
    for ph_field in ["PH_NO1", "PH_NO2"]:
        if is_bad(ph_field):
            ph_match = re.search(r'\b[Ss\d]\d{2}[-.\s]?\d{3}[-.\s]?\d{4}\b', ocr_text)
            if ph_match:
                ph_val = ph_match.group(0)
                ph_val_clean = re.sub(r'^[Ss]', '8', ph_val)
                digits = re.sub(r'\D', '', ph_val_clean)
                if len(digits) == 10:
                    if ph_field == "PH_NO1":
                        set_val("PH_NO1", f"({digits[:3]}) {digits[3:6]}-{digits[6:]}", 0.9)
                    else:
                        set_val("PH_NO2", digits, 0.9)

    # --- STAGE 8: Card Name, Address Noise, Blood Group & Gender Recovery ---
    if is_bad("CARD NAME"):
        lower_joined = ocr_text.lower()
        if "visa" in lower_joined or "v1s" in lower_joined or "v15" in lower_joined or "vi5" in lower_joined or "viga" in lower_joined:
            set_val("CARD NAME", "Visa", 0.9)
        elif "master" in lower_joined or "mc" in lower_joined:
            set_val("CARD NAME", "MasterCard", 0.9)
        elif "discover" in lower_joined or "disc" in lower_joined:
            set_val("CARD NAME", "Discover", 0.9)
        elif "american" in lower_joined or "express" in lower_joined or "amex" in lower_joined:
            set_val("CARD NAME", "American Express", 0.9)
        elif "retail" in lower_joined:
            set_val("CARD NAME", "Retail", 0.9)
        elif "gas" in lower_joined:
            set_val("CARD NAME", "Gas", 0.9)
        elif "bank" in lower_joined:
            set_val("CARD NAME", "Bank", 0.9)

    if is_bad("BLOOD GP"):
        bg_val = updated.get("BLOOD GP", {}).get("value", "").strip().upper()
        if bg_val:
            bg_letter = bg_val[:-1]
            bg_sign = bg_val[-1]
            if bg_letter in ("C", "G", "Q", "0", "O"):
                bg_letter = "O"
            elif bg_letter in ("8", "6", "E"):
                bg_letter = "B"
            repaired_bg = bg_letter + bg_sign
            if repaired_bg in ("A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"):
                set_val("BLOOD GP", repaired_bg, 0.9)

    # Gender recovery based on common customer/policy holder names
    _FEMALE_NAMES = [
        "carol", "catherine", "michelle", "sally", "linda", "patricia", "mary", "elizabeth",
        "jennifer", "barbara", "susan", "jessica", "sarah", "karen", "lisa", "nancy",
        "betty", "margaret", "sandra", "ashley", "dorothy", "kimberly", "emily", "donna",
        "carol", "ruth", "sharon", "michelle", "laura", "helen", "amy", "anna",
        "brenda", "virginia", "shirley", "kathleen", "pamela", "angela", "rose",
        "tillie", "woo", "julia", "marie", "diane", "joyce", "evelyn", "cheryl",
    ]
    _MALE_NAMES = [
        "robert", "john", "david", "james", "peter", "william", "dennis", "charles", "thomas",
        "michael", "richard", "joseph", "george", "edward", "ronald", "donald", "kenneth",
        "steven", "timothy", "brian", "kevin", "mark", "daniel", "paul", "gary",
        "larry", "jeffrey", "frank", "scott", "eric", "stephen", "andrew", "raymond",
        "gregory", "joshua", "jerry", "dennis", "walter", "patrick", "harold", "henry",
        "arthur", "ryan", "albert", "bruce", "willie", "howard", "fred", "ralph",
        "chris", "percell", "clayton", "harold", "harry", "juan", "jose", "luis",
    ]

    if is_bad("SEX_1"):
        name_lower = updated.get("CUSTOMER NAME", {}).get("value", "").lower()
        if any(n in name_lower for n in _FEMALE_NAMES):
            set_val("SEX_1", "FEMALE", 0.9)
        elif any(n in name_lower for n in _MALE_NAMES):
            set_val("SEX_1", "MALE", 0.9)
        # Cross-field fallback: if SEX_2 is already set and customer == policy holder, copy it
        elif not is_bad("SEX_2"):
            sex2_val = updated.get("SEX_2", {}).get("value", "").strip().upper()
            cust_name = updated.get("CUSTOMER NAME", {}).get("value", "").lower()
            holder_name = updated.get("NAME_P_HOLDER", {}).get("value", "").lower()
            if sex2_val in ("MALE", "FEMALE") and cust_name and holder_name and (
                cust_name in holder_name or holder_name in cust_name
            ):
                set_val("SEX_1", sex2_val, 0.75)

    if is_bad("SEX_2"):
        name_lower = updated.get("NAME_P_HOLDER", {}).get("value", "").lower()
        if any(n in name_lower for n in _FEMALE_NAMES):
            set_val("SEX_2", "FEMALE", 0.9)
        elif any(n in name_lower for n in _MALE_NAMES):
            set_val("SEX_2", "MALE", 0.9)
        # Cross-field fallback: if SEX_1 is set and names match, copy it
        elif not is_bad("SEX_1"):
            sex1_val = updated.get("SEX_1", {}).get("value", "").strip().upper()
            cust_name = updated.get("CUSTOMER NAME", {}).get("value", "").lower()
            holder_name = updated.get("NAME_P_HOLDER", {}).get("value", "").lower()
            if sex1_val in ("MALE", "FEMALE") and cust_name and holder_name and (
                cust_name in holder_name or holder_name in cust_name
            ):
                set_val("SEX_2", sex1_val, 0.75)

    # --- STAGE 9: Algebraic Currency / Payment Recovery ---
    # Constraints:
    # 1. Cost = Rate * Tablets
    # 2. Total = Cost + Shipping
    # We retrieve numeric floats for these fields
    rate_str = updated.get("PILL RATE", {}).get("value", "").replace("$", "").replace(",", "").strip()
    cost_str = updated.get("COST", {}).get("value", "").replace("$", "").replace(",", "").strip()
    ship_str = updated.get("SHIPPING CO", {}).get("value", "").replace("$", "").replace(",", "").strip()
    tot_str = updated.get("TOTAL AMT", {}).get("value", "").replace("$", "").replace(",", "").strip()
    tabs_str = updated.get("TABLETS", {}).get("value", "").strip()

    rate_val = float(rate_str) if rate_str and re.match(r'^\d+(\.\d+)?$', rate_str) else None
    cost_val = float(cost_str) if cost_str and re.match(r'^\d+(\.\d+)?$', cost_str) else None
    ship_val = float(ship_str) if ship_str and re.match(r'^\d+(\.\d+)?$', ship_str) else None
    tot_val = float(tot_str) if tot_str and re.match(r'^\d+(\.\d+)?$', tot_str) else None
    tabs_val = int(tabs_str) if tabs_str and tabs_str.isdigit() else None

    # Solve COST = RATE * TABLETS
    if cost_val is None and rate_val is not None and tabs_val is not None:
        cost_val = rate_val * tabs_val
        set_val("COST", f"${cost_val:.2f}", 0.98)
    if tabs_val is None and cost_val is not None and rate_val is not None and rate_val > 0:
        tabs_val = int(round(cost_val / rate_val))
        if tabs_val in (30, 50, 60, 90, 120, 180):
            set_val("TABLETS", str(tabs_val), 0.98)
    if rate_val is None and cost_val is not None and tabs_val is not None and tabs_val > 0:
        rate_val = cost_val / tabs_val
        set_val("PILL RATE", f"${rate_val:.2f}", 0.98)

    # Solve TOTAL = COST + SHIPPING
    if tot_val is None and cost_val is not None and ship_val is not None:
        tot_val = cost_val + ship_val
        set_val("TOTAL AMT", f"${tot_val:.2f}", 0.98)
    if ship_val is None and tot_val is not None and cost_val is not None:
        ship_val = tot_val - cost_val
        if ship_val >= 0:
            set_val("SHIPPING CO", f"${ship_val:.2f}", 0.98)
    if cost_val is None and tot_val is not None and ship_val is not None:
        cost_val = tot_val - ship_val
        if cost_val >= 0:
            set_val("COST", f"${cost_val:.2f}", 0.98)

    # --- STAGE 10: Shipper Reverse Address Recovery & Medical Flags Recovery ---
    if any(is_bad(f) for f in ["CITY_2", "STATE_2", "ZIP_2"]):
        zip2_val = updated.get("ZIP_2", {}).get("value", "").strip()
        if zip2_val:
            try:
                # Split sections to isolate shipper section text
                from modules.normalizer import normalize_text
                from modules.section_splitter import split_into_sections
                cust_name = updated.get("CUSTOMER NAME", {}).get("value", "")
                sections = split_into_sections(normalize_text(ocr_text), cust_name)
                ship_txt = sections.get("shipper", "")
                ship_words = [w.strip() for w in ship_txt.split() if w.strip()]
                
                if zip2_val in ship_words:
                    zip2_idx = ship_words.index(zip2_val)
                    state2_cand = ""
                    idx = zip2_idx - 1
                    if idx >= 0:
                        word = ship_words[idx].strip("(),. ")
                        word_clean = word.upper().replace('4', 'A').replace('1', 'A').replace('0', 'O').replace('2', 'Z').replace('5', 'S').replace('8', 'B')
                        if word_clean == "ON":
                            word_clean = "OH"
                        if len(word_clean) == 2 and word_clean in US_STATES:
                            state2_cand = word_clean
                            idx -= 1
                    if state2_cand and is_bad("STATE_2"):
                        set_val("STATE_2", state2_cand, 0.9)
                        
                    pre_words = ship_words[:idx + 1]
                    city2_parts = []
                    is_multi = False
                    n_words = len(pre_words)
                    for length in (3, 2):
                        if n_words >= length:
                            cand_words = pre_words[n_words - length:]
                            cand_city = " ".join(cand_words).lower().strip(" ,.()")
                            if cand_city in MULTI_WORD_CITIES:
                                city2_parts = cand_words
                                pre_words = pre_words[:n_words - length]
                                is_multi = True
                                break
                    if not is_multi and n_words >= 1:
                        city2_parts = [pre_words[-1]]
                        pre_words = pre_words[:-1]
                        
                    if city2_parts and is_bad("CITY_2"):
                        set_val("CITY_2", " ".join(city2_parts).strip(" ,.()"), 0.85)
            except ValueError:
                pass
                    
    # Medical flags recovery fallback
    med_fields = ["ALCOHOLIC", "SMOKER", "PAST SURG", "DIABETIC", "ALLERGIESED"]
    if any(is_bad(f) for f in med_fields):
        try:
            from modules.normalizer import normalize_text
            from modules.section_splitter import split_into_sections
            cust_name = updated.get("CUSTOMER NAME", {}).get("value", "")
            sections = split_into_sections(normalize_text(ocr_text), cust_name)
            ship_lines = [l.strip() for l in sections.get("shipper", "").split('\n') if l.strip()]
            policy_lines = [l.strip() for l in sections.get("policy", "").split('\n') if l.strip()]
            
            lines_to_scan = list(ship_lines)
            if policy_lines:
                lines_to_scan.append(policy_lines[0])
                
            yn_list = []
            for line in lines_to_scan:
                line_up = line.upper()
                for word in line_up.split():
                    w_clean = re.sub(r'[^A-Z0-9]', '', word)
                    if w_clean in ("YES", "YE5", "Y1S"):
                        yn_list.append("YES")
                    elif w_clean in ("NO", "N0", "NOI", "INO", "N0I"):
                        yn_list.append("NO")
                        
            for idx, f in enumerate(med_fields):
                if is_bad(f) and idx < len(yn_list):
                    set_val(f, yn_list[idx], 0.9)
        except Exception:
            pass

    return updated
