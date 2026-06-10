import re

# Centralized regex patterns for structured parsing

# Matches record number at the beginning (exactly 5 digits)
RECORD_NO_PATTERN = re.compile(r'^\s*(\d{5})\b')

# Matches standard emails
EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')

# Matches 5-digit zip codes
ZIP_PATTERN = re.compile(r'\b(\d{5})\b')

# Matches standard US phone numbers: (XXX) XXX-XXXX or XXX-XXX-XXXX or XXXXXXXXXX
PHONE_PATTERN = re.compile(r'\(?\d{3}\)?\s*-?\d{3}-?\d{4}')

# Matches dates in MM/DD/YY or MM/DD/YYYY format
DATE_PATTERN = re.compile(r'\b(\d{2}/\d{2}/\d{2,4})\b')

# Matches currency values (requires dollar sign, e.g. $150.00, $2.87, $20.00)
CURRENCY_PATTERN = re.compile(r'\$?\s*(\d+\.\d{2})')

# Matches stamp names like BaX_05aCp_14239 (case-insensitive, optional spaces/underscores)
STAMP_PATTERN = re.compile(r'\b([BSbs5]aX)[\s_.\-]*([a-zA-Z0-9_\-.]+?)[\s_.\-]*([a-zA-Z0-9]{5})\b', re.IGNORECASE)


