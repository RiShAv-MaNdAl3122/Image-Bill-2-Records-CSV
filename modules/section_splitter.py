import re
from typing import List, Dict
from utils.regex_patterns import STAMP_PATTERN


# List of known card names for triggering transition to Medication section
CARD_TRIGGERS = [
    "VISA", "V1SA", "VI5A", "VIGA", "VIS4", "V1S4",
    "MASTER", "MASTERCARD", "MC ", " AMEX", "AMERICAN EXPRESS", "AMERICANEXPRESS",
    "DISCOVER", "DISC0VER", "DISCOVERR",
    "RETAIL", "RETALL", "RETAL", "GAS", "BANK", "OTHER", "OTHAR"
]

# List of known medicine names for triggering transition to Medication section
MED_TRIGGERS = [
    "PHENTERMINE", "DIAZEPAM", "XANAX", "VALIUM", "ADIPEX", "MERIDIA",
    "BONTRIL", "IONAMIN", "FASTIN", "PONDIMIN", "REDUX", "ZANTREX",
    "AMBIEN", "LUNESTA", "HALCION", "RESTORIL", "SONATA", "CELEBREX",
    "VIOXX", "MORPHINE", "OXYCONTIN", "PERCOCET", "VICODIN", "TYLENOL",
    "IBUPROFEN", "ASPIRIN", "NAPROXEN", "DIDREX", "ZOLOFT", "CLONAZEPAM", "CLONAZAPAM"
]

def edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

def is_name_repetition(line: str, customer_name: str) -> bool:
    if not customer_name or not line:
        return False
        
    # Clean both strings to lowercase alphanumeric
    c1 = re.sub(r'[^a-z0-9]', '', customer_name.lower())
    c2 = re.sub(r'[^a-z0-9]', '', line.lower())
    
    if not c1 or not c2:
        return False
        
    # Remove common title/suffix noise from start/end
    for noise in ("mr", "mrs", "ms", "dr", "iii", "ii"):
        if c1.startswith(noise) and len(c1) > len(noise) + 2:
            c1 = c1[len(noise):]
        if c2.startswith(noise) and len(c2) > len(noise) + 2:
            c2 = c2[len(noise):]
            
    # If customer name is short, require exact substring or close match on word-level
    if len(c1) <= 4:
        if c1 in c2:
            return True
        if len(c1) >= 3:
            for w2 in re.split(r'[\s\-_,.]+', line.lower()):
                w2_clean = re.sub(r'[^a-z0-9]', '', w2)
                if len(w2_clean) >= 3 and edit_distance(c1, w2_clean) <= 1:
                    return True
        return False
        
    # Check if a prefix of c2 of length len(c1) is close to c1
    for l in (len(c1) - 1, len(c1), len(c1) + 1):
        if l <= len(c2):
            sub = c2[:l]
            if edit_distance(c1, sub) <= 2:
                return True
                
    # Also check word-level edit distance for any significant word (>2 chars)
    w1_list = [w for w in re.split(r'[\s\-_,.]+', customer_name.lower()) if len(w) > 2 and w not in ("mrs", "mr", "ms", "dr", "iii", "ii")]
    w2_list = [w for w in re.split(r'[\s\-_,.]+', line.lower()) if len(w) > 2 and w not in ("mrs", "mr", "ms", "dr", "iii", "ii")]
    
    for w1 in w1_list:
        for w2 in w2_list:
            if edit_distance(w1, w2) <= 1:
                return True
                
    return False

def split_into_sections(ocr_text: str, customer_name: str = "") -> Dict[str, str]:
    """
    Splits the OCR text into 4 logical sections sequentially:
    1. customer: Customer demographics
    2. shipper: Shipper demographics and medical flags
    3. policy: Policy details and stamp
    4. medication: Medicine and payment details
    
    Returns a dict mapping section name to string.
    """
    lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
    
    sections = {
        "customer": [],
        "shipper": [],
        "policy": [],
        "medication": []
    }
    
    current_section = "customer"
    next_line_is_shipper = False
    
    # Pre-compile blood group pattern for robust layout division
    blood_gp_pat = re.compile(r'\b[ABO0ab]{1,2}\s*[+\-]\b', re.IGNORECASE)
    
    for idx, line in enumerate(lines):
        line_upper = line.upper()
        
        # --- Check Transitions ---
        if current_section == "customer":
            is_shipper = False
            if next_line_is_shipper:
                is_shipper = True
            elif customer_name and idx >= 2 and is_name_repetition(line, customer_name):
                is_shipper = True
            elif re.search(r'\bCA\b', line_upper):
                is_shipper = True
                
            # If current line contains a blood group, transition on the next line
            if blood_gp_pat.search(line):
                next_line_is_shipper = True
                
            if is_shipper:
                current_section = "shipper"
                
        elif current_section == "shipper":
            is_policy = False
            # Policy pattern: SR_ or SR10 or 5R or similar
            if re.search(r'\b(SR|5R|sR|Sr)[_0-9a-zA-Z]*\b', line) or "SR10" in line_upper or "SR_" in line_upper or "SR " in line_upper:
                is_policy = True
            # Stamp indicator
            elif "BAX_" in line_upper or "BAX " in line_upper or "BAX_05" in line_upper or STAMP_PATTERN.search(line):
                is_policy = True
            # Second email check
            elif "@" in line:
                is_policy = True
            # Premium amount check
            elif "$" in line and any(val in line for val in ["100.00", "150.00", "200.00", "250.00", "300.00"]):
                is_policy = True
                
            if is_policy:
                current_section = "policy"
                
        elif current_section == "policy":
            is_med = False
            # Card check
            if any(card in line_upper for card in CARD_TRIGGERS):
                is_med = True
            # Medicine check
            elif any(med in line_upper for med in MED_TRIGGERS):
                is_med = True
            # Dosage or currency density check
            elif re.search(r'\b\d+(?:\.\d+)?\s*MG\b', line_upper) or re.search(r'(?<![-/])\b(30|50|60|90|120|180)\b(?![-/])', line_upper):
                if len(re.findall(r'\$\d+', line)) >= 1 or "RATE" in line_upper or "COST" in line_upper or "TOTAL" in line_upper:
                    is_med = True
            
            if is_med:
                current_section = "medication"
                
        sections[current_section].append(line)
        
    # Reconstruct section strings
    return {k: "\n".join(v) for k, v in sections.items()}
