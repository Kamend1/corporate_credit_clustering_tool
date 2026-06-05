# EDGAR Notebook Operating Runbook

## Purpose of this document

This runbook explains how to operate Notebook 01 safely and reproducibly. It is designed for project maintainers, future users, and reviewers who need to understand the EDGAR data acquisition workflow without accidentally triggering a large data download.

Notebook 01 is the data acquisition layer of the Corporate Credit Clustering Tool. The final project notebooks should still contain their own explanatory markdown; this file is the operational reference.

---

## 1. Notebook location

Use the actual notebook filename in the repository. Recommended current naming convention:

```text
notebooks/01_obtain_model_training_data_EDGAR.ipynb
```

If the repository still uses:

```text
notebooks/01_obtain_model_training_data_EDGAR_cleaned.ipynb
```

rename references consistently across README, docs, and notebook markdown.

The notebook assumes it is run from inside the `notebooks/` folder, so project root is usually calculated as:

```python
PROJECT_ROOT = Path.cwd().parent
```

---

## 2. Legal and data-use note

The notebook uses public SEC EDGAR data. It must not attempt to bypass access controls, scrape private data, or overload public systems.

Operational requirements:

- use a valid SEC User-Agent;
- keep request behavior reasonable;
- do not run accidental full downloads;
- do not commit large generated data artifacts to GitHub;
- document data source and limitations.

This supports the final project requirement for legal and safe code execution.

---

## 3. Start Jupyter

From the repository root:

```bash
jupyter lab
```

or:

```bash
jupyter notebook
```

Then open Notebook 01.

---

## 4. Install dependencies

Preferred:

```bash
pip install -r requirements.txt
```

Notebook 01 specifically needs packages such as:

```bash
pip install pandas numpy requests tqdm duckdb pyarrow edgartools
```

If parquet support errors occur, confirm whether the project uses `pyarrow`, `fastparquet`, or both.

---

## 5. Intended local data structure

Generated data should live under `data/` and should generally not be committed.

```text
project-root/
├── notebooks/
│   └── 01_obtain_model_training_data_EDGAR.ipynb
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── raw_financial_facts_parquet/
│   └── duckdb/
├── docs/
├── src/
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## 6. SEC User-Agent requirement

Before running any SEC request, replace placeholder values such as:

```python
"User-Agent": "your_email@domain.com"
```

Use a real contact identifier.

Do not leave placeholders in submitted code.

---

## 7. Main control flags

The notebook should be controlled through explicit flags near the top.

Typical flags:

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

The dangerous operation is:

```python
RUN_RAW_EDGAR_FACT_DOWNLOAD = True
```

A full download should require both:

```python
MAX_TICKERS_FOR_DOWNLOAD = None
CONFIRM_FULL_EDGAR_DOWNLOAD = True
```

This double-control mechanism prevents accidental multi-hour or multi-gigabyte runs.

---

## 8. Recommended execution modes

### Mode A — Safe review mode

Purpose: inspect notebook logic without downloading large data.

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
```

Expected behavior: no major network data pull.

Some later cells may fail if local generated data does not exist. That is acceptable in review mode if clearly documented.

---

### Mode B — Small smoke test

Purpose: test the complete pipeline on a small ticker sample.

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
```

Expected outputs:

```text
data/raw/fundamental_universe.csv
data/interim/fundamental_universe_ticker_cik_sic.csv
data/processed/03_fundamental_universe_ticker_sic_industry.csv
data/raw/raw_financial_facts.csv
data/raw_financial_facts_parquet/*.parquet
data/duckdb/financials.duckdb
data/processed/edgar_download_manifest.csv
```

This is the safest mode for demonstrating functionality.

---

### Mode C — Full initial EDGAR download

Purpose: build the full raw dataset intentionally.

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
```

Use only when the full data build is intentional.

The download should be resumable by checking completed tickers in local raw data.

---

### Mode D — Incremental ticker audit

Purpose: check if new tickers appeared without downloading full company facts.

```python
RUN_EDGAR_SANITY_CHECK = False
RUN_TICKER_UNIVERSE_BUILD = False
RUN_SIC_TABLE_DOWNLOAD = False
RUN_SIC_ENRICHMENT = False
RUN_RAW_EDGAR_FACT_DOWNLOAD = False
RUN_CSV_TO_PARQUET = False
RUN_INCREMENTAL_REFRESH_AUDIT = True
```

Expected output:

```text
data/raw/company_tickers_YYYY-MM-DD.csv
```

This mode identifies new or changed tickers, but does not automatically download their facts.

---

### Mode E — Download only new tickers

Recommended future pattern:

```python
DOWNLOAD_MODE = "new_tickers"
```

rather than manually editing the ticker selection line.

If this control does not yet exist, the notebook may require temporarily setting:

```python
tickers = new_tickers
```

Manual edits should be replaced with a proper `DOWNLOAD_MODE` control in future development.

---

### Mode F — Refresh new fiscal periods

Purpose: add a new fiscal year, for example extending from 2025 to 2026.

```python
TARGET_MAX_FISCAL_YEAR = 2026
RUN_INCREMENTAL_REFRESH_AUDIT = True
```

The refresh logic should identify stale tickers whose max fiscal year is below the target year.

Important: the resume logic must not skip a ticker merely because it exists in the raw CSV. It should skip only if the target fiscal year already exists for that ticker.

Safer logic:

```python
completed = set(
    existing.loc[
        existing["fiscal_year"] >= TARGET_MAX_FISCAL_YEAR,
        "ticker"
    ].unique()
)
```

---

### Mode G — Examiner / reviewer mode

Purpose: allow Dancho or another reviewer to run the notebook safely without triggering a huge EDGAR pull.

Recommended settings:

```python
RUN_EDGAR_SANITY_CHECK = True
RUN_TICKER_UNIVERSE_BUILD = False
RUN_SIC_TABLE_DOWNLOAD = False
RUN_SIC_ENRICHMENT = False
RUN_RAW_EDGAR_FACT_DOWNLOAD = False
RUN_CSV_TO_PARQUET = False
RUN_INCREMENTAL_REFRESH_AUDIT = False

MAX_TICKERS_FOR_DOWNLOAD = 10
CONFIRM_FULL_EDGAR_DOWNLOAD = False
```

Expected behavior:

- no full download;
- no large generated files;
- notebook can explain the process;
- pre-generated sample outputs or documented output examples should be used where possible.

This mode is important for final-project grading because the examiner should not need to run a large data engineering job to understand the project.

---

## 9. Recommended future notebook control improvement

Add a single explicit mode variable:

```python
DOWNLOAD_MODE = "review"  # review, smoke_test, full, new_tickers, stale_periods, manual
MANUAL_TICKERS = []
```

Then select tickers with:

```python
if DOWNLOAD_MODE == "full":
    tickers = fundamental_df["ticker"].dropna().unique().tolist()
elif DOWNLOAD_MODE == "new_tickers":
    tickers = new_tickers
elif DOWNLOAD_MODE == "stale_periods":
    tickers = stale_tickers
elif DOWNLOAD_MODE == "manual":
    tickers = MANUAL_TICKERS
elif DOWNLOAD_MODE in {"review", "smoke_test"}:
    tickers = fundamental_df["ticker"].dropna().unique().tolist()[:MAX_TICKERS_FOR_DOWNLOAD]
else:
    raise ValueError(f"Unknown DOWNLOAD_MODE: {DOWNLOAD_MODE}")
```

This removes fragile manual edits and makes the notebook more operator-grade.

---

## 10. Generated artifacts

Notebook 01 may create:

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

These are generated artifacts, not source code.

---

## 11. Git policy

Commit:

```text
notebooks/01_obtain_model_training_data_EDGAR.ipynb
docs/EDGAR_notebook_operating_runbook.md
data/README.md
.gitignore
```

Do not commit large generated data:

```text
data/raw/
data/interim/
data/processed/*.csv
data/raw_financial_facts_parquet/
data/duckdb/
*.duckdb
*.parquet
```

Recommended `.gitignore`:

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

## 12. Final-project communication note

Notebook 01 should explain:

- what EDGAR/XBRL is;
- why public financial statements are used;
- why large downloads are guarded by flags;
- what data is stored locally;
- why generated data is excluded from Git;
- how the output supports Notebook 02.

This runbook supports operations. The notebook should still contain the direct story for the examiner.
