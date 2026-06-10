# ==============================================================================
# Pipeline Configuration Settings
# ==============================================================================

# Minimum confidence threshold (0.0 to 1.0) above which OCR words are trusted.
# Affects how initial raw extraction scores are calculated.
OCR_CONFIDENCE_THRESHOLD = 0.90

# Directory paths used by the pipeline
INPUT_FOLDER = "input"       # Directory containing raw scanned JPG/PNG records
OUTPUT_FOLDER = "output"     # Directory where final CSVs and NeedReview/ folder are exported
TEMP_FOLDER = "temp"         # Temporary folder cleared at startup; stores page sub-crops

# If True, retains intermediate crop pieces for audit purposes in the temp directory.
# Useful for visually validating divider segmentation coordinates.
CROP_AUDIT_MODE = True

# Debugging override: if True, skips all pages except the last one.
LAST_PAGE_ONLY = False
 
# Capping mechanism for testing. Set to an integer to limit pages processed
# (e.g. 5 or 50), or set to None for full commercial batch execution.
LIMIT_PAGES = None

# Optional page windowing parameters (1-based index, inclusive range).
# Set both to integers to process only a specific page span (e.g., START_PAGE=10, END_PAGE=20).
# Leave as None to process all pages within the limit.
START_PAGE = None
END_PAGE = None