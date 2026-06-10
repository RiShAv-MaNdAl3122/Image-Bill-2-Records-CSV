import re
from typing import Tuple

# Pattern-based OCR normalizer. Keeps rules generic and lightweight.

_DIGIT_TO_LETTER = str.maketrans({
    '4': 'A', '0': 'O', '1': 'I', '5': 'S', '2': 'Z', '8': 'B', '6': 'G', '7': 'T'
})

US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN",
    "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV",
    "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
    "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
}


class NormalizedText(str):
    def __new__(cls, value, norm_factor=1.0):
        obj = super().__new__(cls, value)
        obj.norm_factor = norm_factor
        return obj

    def __iter__(self):
        yield str(self)
        yield self.norm_factor


def _fix_token_substitutions(token: str) -> Tuple[str, int]:
    """Fix common digit/letter confusions inside mixed tokens. Returns (new_token, corrections).
    Only applies when token contains both letters and digits and token length is small (<=12)."""
    corrections = 0
    # Avoid translating pure numeric tokens or common numeric patterns
    if re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', token):
        return token, 0
    if re.fullmatch(r'\d+(?:\.\d+)?[Mm][Gg]', token):
        return token, 0
    if re.fullmatch(r'\d{5}(?:-\d{4})?', re.sub(r'\D', '', token)):
        return token, 0
    if any(ch.isdigit() for ch in token) and any(ch.isalpha() for ch in token) and len(token) <= 12:
        new = token.translate(_DIGIT_TO_LETTER)
        if new != token:
            corrections = 1
            return new, corrections
    return token, 0


def _normalize_phone(token: str) -> Tuple[str, int]:
    digits = re.sub(r'\D', '', token)
    if len(digits) == 10:
        return f'({digits[0:3]}) {digits[3:6]}-{digits[6:10]}', 1
    return token, 0


def _normalize_zip(token: str) -> Tuple[str, int]:
    digits = re.sub(r'\D', '', token)
    if len(digits) == 9:
        return f'{digits[0:5]}-{digits[5:9]}', 1
    if len(digits) == 5:
        return digits, 0
    return token, 0


def _normalize_dates_spacing(line: str) -> Tuple[str, int]:
    """Fixes dates that are split by spaces, e.g. '0 7/11/49' -> '07/11/49'."""
    orig = line
    line = re.sub(r'\b(\d)\s+(\d/\d{1,2}/\d{2,4})\b', r'\1\2', line)
    line = re.sub(r'\b(\d{1,2}/\d)\s+(\d/\d{2,4})\b', r'\1\2', line)
    line = re.sub(r'(\d)\s*/\s*(\d)', r'\1/\2', line)
    return line, (1 if line != orig else 0)


def _split_city_state_zip(line: str) -> Tuple[str, int]:
    """Splits state/zip and city/state if they are merged, e.g. 'CA26728' or 'AnaheimcA 65847'."""
    orig = line
    # 1. Split state and zip when merged: e.g. "CA26728" -> "CA 26728"
    def state_zip_repl(m):
        state, zip_code = m.groups()
        if state.upper() in US_STATES:
            return f"{state} {zip_code}"
        return m.group(0)
    line = re.sub(r'\b([A-Za-z]{2})(\d{5})\b', state_zip_repl, line)

    # 2. Split city and state before a zip: e.g. "AnaheimcA 65847" -> "Anaheim CA 65847"
    def city_state_zip_repl(m):
        city_and_state, zip_code = m.groups()
        if len(city_and_state) >= 4:
            possible_state = city_and_state[-2:]
            if possible_state.upper() in US_STATES:
                city = city_and_state[:-2]
                return f"{city} {possible_state.upper()} {zip_code}"
        return m.group(0)
    line = re.sub(r'\b([A-Za-z]+)\s+(\d{5})\b', city_state_zip_repl, line)
    return line, (1 if line != orig else 0)


def normalize_text(text: str) -> Tuple[str, float]:
    """Apply pattern-based normalizations to OCR text.

    Returns (normalized_text, normalization_factor) where normalization_factor in [0.75, 1.0]
    reduces slightly when automatic corrections were applied to reflect lower confidence.
    """
    if not text:
        return NormalizedText(text, 1.0)

    corrections = 0
    lines = text.split('\n')
    out_lines = []

    date_pat = re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4})')
    # Match BaX/SaX/5aX followed by name part (no spaces allowed inside name part) and a 5-char code
    stamp_regex = re.compile(r'\b([BSbs5]aX)[\s_.\-]*([a-zA-Z0-9_\-.]+?)[\s_.\-]*([a-zA-Z0-9]{5})\b', re.IGNORECASE)

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # collapse whitespace
        line = re.sub(r'\s+', ' ', line)

        # Clean up division by zero or Excel formula errors: e.g. "#DIV/0$258.00" -> "$258.00"
        line = re.sub(r'#\s*DIV\s*/?\s*0\s*!?\s*', '', line, flags=re.IGNORECASE)

        # Split zip code and country/state when merged: e.g. "68489US" -> "68489 US"
        new_line = re.sub(r'\b(\d{5})([A-Za-z]{2,})\b', r'\1 \2', line)
        if new_line != line:
            corrections += 1
            line = new_line
            
        # Split zip code and merged phone number: e.g. "37874423351-1077" -> "37874 423351-1077"
        new_line = re.sub(r'\b(\d{5})(\d{3}\s*-?\s*\d{3}-?\d{4})\b', r'\1 \2', line)
        if new_line != line:
            corrections += 1
            line = new_line
            
        # Fix double or malformed slashes in dates: e.g. "04//07/1955" -> "04/07/1955"
        new_line = re.sub(r'/+', '/', line)
        new_line = re.sub(r'\\+', '/', new_line)
        if new_line != line:
            corrections += 1
            line = new_line


        # 1. Protect Stamp: Find and isolate stamp-like tokens
        stamp_placeholder = None
        m_stamp = stamp_regex.search(line)
        if m_stamp:
            prefix = m_stamp.group(1)
            name_part = m_stamp.group(2).replace(" ", "")
            code_part = m_stamp.group(3)
            clean_stamp = f"{prefix}_{name_part}_{code_part}"
            stamp_placeholder = clean_stamp
            line = line[:m_stamp.start()] + " __STAMP_PLACEHOLDER__ " + line[m_stamp.end():]
            line = re.sub(r'\s+', ' ', line).strip()

        # Split city, state, zip merges (done first before camel case)
        line, corr = _split_city_state_zip(line)
        corrections += corr

        # Camel Case Splitting: e.g. GeorgeUGriffin -> George U Griffin
        new_line = re.sub(r'([a-z])([A-Z])', r'\1 \2', line)
        if new_line != line:
            corrections += 1
            line = new_line

        # Fix space-split dates: e.g. 0 7/11/49 -> 07/11/49
        line, corr = _normalize_dates_spacing(line)
        corrections += corr

        # Fix merged dosage like DIDREX2.5MG -> DIDREX 2.5MG
        new_line = re.sub(r'([A-Za-z])(?=(\d+(?:\.\d+)?\s*MG\b))', r'\1 ', line, flags=re.IGNORECASE)
        if new_line != line:
            corrections += 1
            line = new_line

        # Ensure space before and after dates if merged with neighboring tokens
        new_line = re.sub(r'(?<!\s)(' + date_pat.pattern + r')', r' \1', line)
        new_line = re.sub(r'(' + date_pat.pattern + r')(?!\s)', r'\1 ', new_line)
        if new_line != line:
            corrections += 1
            line = new_line

        # Break runs of letters followed immediately by digits (e.g. POLICY12345)
        new_line = re.sub(r'([A-Za-z]{2,})(?=\d)', r'\1 ', line)
        if new_line != line:
            corrections += 1
            line = new_line

        # 2. Restore Stamp
        if stamp_placeholder:
            line = line.replace("__STAMP_PLACEHOLDER__", stamp_placeholder)
            line = re.sub(r'\s+', ' ', line).strip()

        tokens = line.split(' ')
        norm_tokens = []
        for tok in tokens:
            t = tok
            # Normalize phone-looking tokens
            if re.search(r'[\d\(][\d\)\-\.\s]{6,}\d', t):
                t2, corr = _normalize_phone(t)
                if corr:
                    corrections += corr
                    t = t2

            # Normalize zips
            if re.fullmatch(r'\d{5}(?:[-\s]?\d{4})?', re.sub(r'\D', '', t)):
                t2, corr = _normalize_zip(t)
                if corr:
                    corrections += corr
                    t = t2

            # Token-level digit-letter substitutions (conservative)
            t2, corr = _fix_token_substitutions(t)
            if corr:
                corrections += corr
                t = t2

            # Uppercase two-letter state-like tokens and correct obvious OCR digits
            if len(t) == 2 and re.search(r'[A-Za-z0-9]{2}', t):
                t_clean = t.upper().translate(_DIGIT_TO_LETTER)
                if t_clean != t:
                    corrections += 1
                t = t_clean

            norm_tokens.append(t)

        line = ' '.join(norm_tokens)

        # Tidy punctuation spaces
        line = re.sub(r'\s+([,.:;\-])', r'\1', line)
        line = re.sub(r'([\$])\s+', r'\1', line)

        out_lines.append(line)

    # Compute normalization factor conservatively
    norm_factor = max(0.75, 1.0 - min(5, corrections) * 0.05)
    return NormalizedText('\n'.join(out_lines), norm_factor)
