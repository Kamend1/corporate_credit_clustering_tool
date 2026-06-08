# EDGAR Notebook Operating Runbook

This runbook explains how to operate the EDGAR data acquisition notebook safely.

The EDGAR notebook is the data acquisition layer. It is separate from the feature engineering, clustering, scoring, and reporting layers.

---

## 1. Purpose

The EDGAR notebook supports:

- building the public-company universe;
- mapping tickers and CIKs;
- collecting SEC EDGAR/XBRL financial facts;
- converting raw outputs into local storage formats;
- preparing data for feature engineering and model training.

Configuration changes to thresholds, feature formulas, or scorecard weights do not require rerunning the EDGAR download. They require rerunning the feature engineering and model training workflow.

---

## 2. Safe operating principle

Do not trigger a full EDGAR download accidentally.

A full download should require explicit settings:

```python
MAX_TICKERS_FOR_DOWNLOAD = None
CONFIRM_FULL_EDGAR_DOWNLOAD = True
RUN_RAW_EDGAR_FACT_DOWNLOAD = True
```

For review or demonstration, use a smoke-test or no-download mode.

---

## 3. SEC user-agent requirement

SEC requests should use a valid user-agent/contact string. Do not leave placeholder values such as:

```python
"your_email@domain.com"
```

The project uses public SEC data and should not bypass access controls, scrape private data, or overload public systems.

---

## 4. Recommended modes

### Mode A — safe review mode

Purpose: inspect notebook structure without large downloads.

```python
RUN_RAW_EDGAR_FACT_DOWNLOAD = False
CONFIRM_FULL_EDGAR_DOWNLOAD = False
MAX_TICKERS_FOR_DOWNLOAD = 10
```

### Mode B — smoke test

Purpose: validate EDGAR access and processing on a small sample.

```python
RUN_RAW_EDGAR_FACT_DOWNLOAD = True
MAX_TICKERS_FOR_DOWNLOAD = 10
CONFIRM_FULL_EDGAR_DOWNLOAD = False
```

### Mode C — full initial download

Purpose: build the full dataset intentionally.

```python
RUN_RAW_EDGAR_FACT_DOWNLOAD = True
MAX_TICKERS_FOR_DOWNLOAD = None
CONFIRM_FULL_EDGAR_DOWNLOAD = True
```

### Mode D — incremental refresh audit

Purpose: check for new tickers or stale fiscal periods without rebuilding everything.

---

## 5. Generated artifacts

The notebook may create:

```text
data/raw/
data/interim/
data/processed/
data/raw_financial_facts_parquet/
data/duckdb/
```

Large generated files should generally not be committed to GitHub.

Recommended `.gitignore` entries:

```gitignore
data/raw/
data/interim/
data/processed/*.csv
data/raw_financial_facts_parquet/
data/duckdb/
*.duckdb
*.parquet
```

---

## 6. Reviewer-safe execution

For course review, the examiner should not be forced to run a full EDGAR download.

Recommended reviewer-safe approach:

- keep full download flags off;
- show sample outputs or precomputed artifacts where appropriate;
- document expected outputs;
- let modelling notebooks run from prepared data where possible.

---

## 7. Relationship to later notebooks

| Notebook | Role |
|---|---|
| Notebook 01 | EDGAR data acquisition |
| Notebook 02 | feature engineering, clustering, artifact saving |
| Notebook 03 | private-company scoring, scenarios, reports |
| Notebook 04 | alternative clustering comparison |

---

## 8. Summary

The EDGAR notebook should be operated conservatively. Full downloads should be intentional, while modelling changes should be handled downstream in the feature engineering and clustering notebooks.
