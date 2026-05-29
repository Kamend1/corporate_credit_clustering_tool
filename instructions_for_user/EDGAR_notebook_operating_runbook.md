# EDGAR Notebook Operating Runbook

This runbook explains how to operate the EDGAR data acquisition notebook safely.

Notebook location:

```text
notebooks/01_obtain_model_training_data_EDGAR_cleaned.ipynb
```

The notebook assumes it is stored inside the `notebooks/` folder, so the project root is calculated as:

```python
PROJECT_ROOT = Path.cwd().parent
```

Generated data is written under:

```text
project-root/data/
```

---

## 1. Start Jupyter

From the repository root:

```bash
jupyter lab
```

Then open:

```text
notebooks/01_obtain_model_training_data_EDGAR_cleaned.ipynb
```

---

## 2. Install Required Dependencies

From your active Python environment:

```bash
pip install pandas numpy requests tqdm duckdb pyarrow fastparquet edgartools
```

If the project already has a `requirements.txt`, prefer:

```bash
pip install -r requirements.txt
```

---

## 3. Expected Repository Structure

The notebook creates missing data folders automatically, but the intended structure is:

```text
project-root/
├── notebooks/
│   └── 01_obtain_model_training_data_EDGAR_cleaned.ipynb
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── raw_financial_facts_parquet/
│   └── duckdb/
├── README.md
├── LICENSE
└── .gitignore
```

---

## 4. SEC User-Agent Requirement

Before running any SEC/EDGAR request, replace any placeholder user-agent value:

```python
"User-Agent": "your_email@domain.com"
```

Use a real contact identifier, for example:

```python
"User-Agent": "your.name@example.com"
```

Do not leave the placeholder in the notebook.

---

## 5. Main Control Flags

The notebook is controlled through execution flags near the top.

```python
RUN_EDGAR_SANITY_CHECK = False
RUN_TICKER_UNIVERSE_BUILD = False
RUN_SIC_TABLE_DOWNLOAD = False
RUN_SIC_ENRICHMENT = False
RUN_RAW_EDGAR_FACT_DOWNLOAD = False
RUN_CSV_TO_PARQUET = False
RUN_INCREMENTAL_REFRESH_AUDIT = False

MAX_TICKERS_FOR_DOWNLOAD = 10
CONFIRM_FULL_EDGAR_DOWNLOAD = False

TARGET_MIN_FISCAL_YEAR = 2020
TARGET_MAX_FISCAL_YEAR = 2025
```

The dangerous flag is:

```python
RUN_RAW_EDGAR_FACT_DOWNLOAD = True
```

A full download also requires:

```python
MAX_TICKERS_FOR_DOWNLOAD = None
CONFIRM_FULL_EDGAR_DOWNLOAD = True
```

This is intentional. It prevents accidental multi-gigabyte EDGAR downloads.

---

# Operating Modes

---

## Mode A — Safe Review Run

Use this mode when you only want to inspect the notebook without downloading large data.

### Flags

```python
RUN_EDGAR_SANITY_CHECK = False
RUN_TICKER_UNIVERSE_BUILD = False
RUN_SIC_TABLE_DOWNLOAD = False
RUN_SIC_ENRICHMENT = False
RUN_RAW_EDGAR_FACT_DOWNLOAD = False
RUN_CSV_TO_PARQUET = False
RUN_INCREMENTAL_REFRESH_AUDIT = False

MAX_TICKERS_FOR_DOWNLOAD = 10
CONFIRM_FULL_EDGAR_DOWNLOAD = False

TARGET_MIN_FISCAL_YEAR = 2020
TARGET_MAX_FISCAL_YEAR = 2025
```

### How to run

Run cells top to bottom.

### Expected result

No major EDGAR download should be triggered.

Some later cells may fail if local data files do not exist yet. That is expected in review mode.

---

## Mode B — Small Smoke Test

Use this mode before any full download.

It validates the pipeline on a small number of tickers.

### Flags

```python
RUN_EDGAR_SANITY_CHECK = True
RUN_TICKER_UNIVERSE_BUILD = True
RUN_SIC_TABLE_DOWNLOAD = True
RUN_SIC_ENRICHMENT = True
RUN_RAW_EDGAR_FACT_DOWNLOAD = True
RUN_CSV_TO_PARQUET = True
RUN_INCREMENTAL_REFRESH_AUDIT = False

MAX_TICKERS_FOR_DOWNLOAD = 10
CONFIRM_FULL_EDGAR_DOWNLOAD = False

TARGET_MIN_FISCAL_YEAR = 2020
TARGET_MAX_FISCAL_YEAR = 2025
```

### How to run

Run all cells top to bottom.

### Expected outputs

```text
data/raw/fundamental_universe.csv
data/interim/fundamental_universe_ticker_cik_sic.csv
data/processed/03_fundamental_universe_ticker_sic_industry.csv
data/raw/raw_financial_facts.csv
data/raw_financial_facts_parquet/part_00000.parquet
data/duckdb/financials.duckdb
data/processed/edgar_download_manifest.csv
```

### Purpose

This confirms that:

- SEC requests work.
- Ticker universe generation works.
- SIC enrichment works.
- Raw EDGAR facts extraction works.
- CSV-to-parquet conversion works.
- DuckDB can query the parquet dataset.

---

## Mode C — Full Initial EDGAR Download

Use this only when you intentionally want the large EDGAR pull.

### Flags

```python
RUN_EDGAR_SANITY_CHECK = True
RUN_TICKER_UNIVERSE_BUILD = True
RUN_SIC_TABLE_DOWNLOAD = True
RUN_SIC_ENRICHMENT = True
RUN_RAW_EDGAR_FACT_DOWNLOAD = True
RUN_CSV_TO_PARQUET = True
RUN_INCREMENTAL_REFRESH_AUDIT = False

MAX_TICKERS_FOR_DOWNLOAD = None
CONFIRM_FULL_EDGAR_DOWNLOAD = True

TARGET_MIN_FISCAL_YEAR = 2020
TARGET_MAX_FISCAL_YEAR = 2025
```

### How to run

Run all cells top to bottom.

### Important behavior

The raw facts download is intended to be resumable.

If the notebook is interrupted, rerun it later. The download logic checks completed tickers from:

```text
data/raw/raw_financial_facts.csv
```

and skips tickers already present.

### After download finishes

Make sure parquet conversion runs:

```python
RUN_CSV_TO_PARQUET = True
```

This creates or refreshes:

```text
data/raw_financial_facts_parquet/
```

DuckDB reads from the parquet folder.

---

## Mode D — Update Available Tickers Only

Use this when you want to check whether new EDGAR tickers appeared, without downloading company facts.

### Flags

```python
RUN_EDGAR_SANITY_CHECK = False
RUN_TICKER_UNIVERSE_BUILD = False
RUN_SIC_TABLE_DOWNLOAD = False
RUN_SIC_ENRICHMENT = False
RUN_RAW_EDGAR_FACT_DOWNLOAD = False
RUN_CSV_TO_PARQUET = False
RUN_INCREMENTAL_REFRESH_AUDIT = True

MAX_TICKERS_FOR_DOWNLOAD = 10
CONFIRM_FULL_EDGAR_DOWNLOAD = False
```

### How to run

Run these sections:

```text
1. Imports
2. Path configuration
3. Flag configuration
4. Existing universe load, if required
5. Incremental refresh audit
```

### Expected output

The notebook saves a dated ticker snapshot:

```text
data/raw/company_tickers_YYYY-MM-DD.csv
```

It also prints:

```text
New tickers: X
Removed / changed tickers: Y
```

### Important note

This mode does not automatically download facts for new tickers. It only identifies ticker-universe changes.

---

## Mode E — Download Only Newly Appeared Tickers

Use this after running the ticker update audit.

### Step 1 — Run the ticker audit

Use Mode D first.

This creates the `new_tickers` list.

### Step 2 — Change ticker selection in the raw download cell

Temporarily change:

```python
tickers = fundamental_df["ticker"].dropna().unique().tolist()
```

to:

```python
tickers = new_tickers
```

### Step 3 — Set download flags

```python
RUN_RAW_EDGAR_FACT_DOWNLOAD = True
RUN_CSV_TO_PARQUET = True

MAX_TICKERS_FOR_DOWNLOAD = None
CONFIRM_FULL_EDGAR_DOWNLOAD = True
```

Keep these off unless you are rebuilding the full universe:

```python
RUN_TICKER_UNIVERSE_BUILD = False
RUN_SIC_TABLE_DOWNLOAD = False
RUN_SIC_ENRICHMENT = False
```

### Step 4 — Run relevant sections

Run:

```text
1. Imports/configuration
2. Load final universe
3. Incremental audit
4. Raw EDGAR fact download
5. CSV to parquet
6. DuckDB view creation
7. Manifest generation
```

### Important note

The current parquet folder is treated as generated output. After adding new facts, rebuild parquet from the updated raw CSV.

---

## Mode F — Update New Fiscal Periods / New Annual Reports

Use this when companies have reported a new annual year.

Example: you have data through 2025 and now want 2026.

### Step 1 — Update target fiscal year

```python
TARGET_MIN_FISCAL_YEAR = 2020
TARGET_MAX_FISCAL_YEAR = 2026
```

### Step 2 — Enable incremental audit

```python
RUN_INCREMENTAL_REFRESH_AUDIT = True
```

### Step 3 — Run manifest and stale ticker detection

Run:

```text
1. Imports/configuration
2. DuckDB view creation
3. Manifest creation
4. Stale ticker detection
```

The stale ticker logic checks:

```python
manifest["max_fiscal_year"] < TARGET_MAX_FISCAL_YEAR
```

and produces:

```python
stale_tickers
```

### Step 4 — Change ticker selection in raw download cell

Temporarily change:

```python
tickers = fundamental_df["ticker"].dropna().unique().tolist()
```

to:

```python
tickers = stale_tickers
```

### Step 5 — Set download flags

```python
RUN_RAW_EDGAR_FACT_DOWNLOAD = True
RUN_CSV_TO_PARQUET = True

MAX_TICKERS_FOR_DOWNLOAD = None
CONFIRM_FULL_EDGAR_DOWNLOAD = True
```

### Step 6 — Fix resume logic for fiscal-period refresh

The existing full-download resume logic skips any ticker already present in the raw CSV:

```python
completed = set(existing["ticker"].unique())
```

For stale-period refresh, this is too aggressive.

Use this instead:

```python
completed = set(
    existing.loc[
        existing["fiscal_year"] >= TARGET_MAX_FISCAL_YEAR,
        "ticker"
    ].unique()
)
```

This skips only tickers that already have the target fiscal year locally.

### Step 7 — Run relevant sections

Run:

```text
1. Raw EDGAR fact download
2. CSV to parquet
3. DuckDB view creation
4. Credit model base creation
5. Concept coverage audit
6. Manifest update
```

---

# Generated Artifacts

The notebook may generate these files:

```text
data/raw/fundamental_universe.csv
data/raw/company_tickers_YYYY-MM-DD.csv
data/raw/raw_financial_facts.csv

data/interim/fundamental_universe_ticker_cik_sic.csv

data/processed/03_fundamental_universe_ticker_sic_industry.csv
data/processed/edgar_download_manifest.csv

data/raw_financial_facts_parquet/*.parquet

data/duckdb/financials.duckdb
```

---

# Git Policy

Commit notebook and metadata:

```text
notebooks/01_obtain_model_training_data_EDGAR_cleaned.ipynb
data/README.md
.gitignore
```

Do not commit generated data:

```text
data/raw/
data/interim/
data/processed/*.csv
data/raw_financial_facts_parquet/
data/duckdb/
*.duckdb
*.parquet
```

Recommended `.gitignore` entries:

```gitignore
data/raw/
data/interim/
data/processed/*.csv
data/raw_financial_facts_parquet/
data/duckdb/
*.duckdb
*.parquet

!data/README.md
```

---

# Recommended Next Notebook Improvement

The current notebook works, but new-ticker and stale-period modes still require manually changing the ticker selection line.

The clean next step is to add:

```python
DOWNLOAD_MODE = "full"  # "full", "new_tickers", "stale_periods", "manual"
MANUAL_TICKERS = []
```

Then replace manual ticker selection with:

```python
if DOWNLOAD_MODE == "full":
    tickers = fundamental_df["ticker"].dropna().unique().tolist()
elif DOWNLOAD_MODE == "new_tickers":
    tickers = new_tickers
elif DOWNLOAD_MODE == "stale_periods":
    tickers = stale_tickers
elif DOWNLOAD_MODE == "manual":
    tickers = MANUAL_TICKERS
else:
    raise ValueError(f"Unknown DOWNLOAD_MODE: {DOWNLOAD_MODE}")
```

This would make the notebook operator-grade and remove manual edits.

---

# Practical Default Workflow

Use this sequence in practice:

```text
1. Run Mode B — Small Smoke Test.
2. If successful, run Mode C — Full Initial EDGAR Download.
3. Later, run Mode D periodically to detect new tickers.
4. Use Mode E only if new tickers appear.
5. Use Mode F when a new fiscal year should be added.
```

Bottom line: never run full EDGAR download unless both controls are explicit:

```python
MAX_TICKERS_FOR_DOWNLOAD = None
CONFIRM_FULL_EDGAR_DOWNLOAD = True
```
