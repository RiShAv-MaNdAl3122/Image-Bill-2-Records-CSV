# Image Bill 2 Records CSV

**Image Bill 2 Records CSV** is a local, high-performance, offline data extraction system designed to convert multi-column scanned health insurance/medical bills into structured, verified, pipe-delimited CSV files. 

Built with strict data privacy in mind, this system operates **100% locally** (zero cloud API dependencies, no external LLM processor connections).

---

## Core Features

- **Layout-Aware Column Partitioning**: Dynamically detects column coordinates and splits multi-column document sheets using horizontal Hough dividers and whitespace valley boundaries.
- **Whitespace Valley Snapping**: Automatically searches local pixel-row intensities to snap crop margins to whitespace zones, avoiding cut-off text.
- **Multi-Segment Cropping & Stitching**: Vertically merges records that cross page boundaries, creating a unified crop canvas for OCR.
- **High-Fidelity Offline OCR**: Employs a local PaddleOCR model to extract raw text with character-level bounding box coordinate matching.
- **Strict Sequential Section Splitting**: Enforces a strict state machine (`customer` $\rightarrow$ `shipper` $\rightarrow$ `policy` $\rightarrow$ `medication`) to partition sections safely and prevent text noise contamination.
- **10-Stage Contextual Recovery Engine**: 
  - Resolves missing states from ZIP codes.
  - Automatically completes truncated email domains (e.g. `@yah` $\rightarrow$ `@yahoo.com`).
  - Fuzzy-recovers misspelled medicine names (e.g. `ALIUM` $\rightarrow$ `VALIUM`) using a fast Levenshtein edit distance check.
  - Uses algebraic constraints to solve missing pricing ($Cost = Rate \times Tablets$; $Total = Cost + Shipping$).
- **Strict Data Validation & Partitioning**: Enforces strict database schema checks, atomic file writes, and outputs invalid records directly into a review CSV accompanied by visual sub-crops for human verification.
- **Statistics Dashboard**: Automatically exports an execution report (`stats.csv`) documenting success rates and field-level recovery statistics at the end of each batch run.

---

## Directory Structure

```text
ImageBillExtractor/
├── input/                  # Put source scanned JPG/PNG records here
├── output/                 # Destination for final results
│   ├── NeedReview/         # Crop images of records requiring manual review
│   ├── extracted_records.csv  # Pipe-delimited CSV of fully passing records
│   ├── failed_records.csv     # Pipe-delimited CSV of records requiring review
│   └── stats.csv           # Pipeline execution & field statistics dashboard
├── modules/                # Core processing subsystems
│   ├── cropper.py          # Bounding-box snapping and vertical stitching
│   ├── csv_schema.py       # Centralized database schema definition
│   ├── exporter.py         # Thread-safe atomic CSV exporters
│   ├── extractor.py        # Regex parsers and field matchers
│   ├── layout.py           # Hough divider and page column grid segmenters
│   ├── normalizer.py       # Character correction and token normalizers
│   ├── ocr.py              # PaddleOCR wrapper and coordinate caching
│   ├── recovery_engine.py  # 10-stage contextual and algebraic recovery solvers
│   ├── section_splitter.py # Sequential zone splitter and spelling distance checks
│   └── validator.py        # Data format and presence validator
├── utils/                  # Shared helper subsystems
│   ├── logger.py           # Formatted logger tracking execution states
│   └── regex_patterns.py   # Centralized regular expression compilation
├── config.py               # Global user configuration parameters
├── main.py                 # Core pipeline orchestration script
├── requirements.txt        # Package dependencies list
└── .gitignore              # Git version control ignore rules
```

---

## Quick Start

### 1. Prerequisites
- Python 3.10+ (64-bit recommended)
- Standard system dependencies for OpenCV and PaddleOCR (C++ build tools or appropriate runtime libraries depending on OS)

### 2. Installation
Clone the repository and set up a virtual environment:
```powershell
# Clone the repository
git clone <your-repository-url>
cd ImageBillExtractor

# Set up virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Pipeline
1. Place scanned document images inside the `input/` directory.
2. Configure parameters in `config.py` (e.g. set `LIMIT_PAGES = None` to run full production batches, or set to an integer to test on a small page range).
3. Execute the pipeline:
   ```powershell
   .venv\Scripts\python.exe main.py
   ```
4. Find the resulting CSV tables and manual review images inside the `output/` directory.

---

## Configuration Options (`config.py`)

- `OCR_CONFIDENCE_THRESHOLD`: Confidence score (0.0 to 1.0) above which raw OCR characters are trusted.
- `INPUT_FOLDER` / `OUTPUT_FOLDER` / `TEMP_FOLDER`: Directory path mappings.
- `LIMIT_PAGES`: Limit the number of pages processed (useful for quick incremental testing; set to `None` for full runs).
- `CROP_AUDIT_MODE`: Retains intermediate crops inside the temp directory for debugging divider boundaries.
- `START_PAGE` / `END_PAGE`: Process only a specific window range of pages (1-based index).
