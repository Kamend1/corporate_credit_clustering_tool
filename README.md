# Corporate Credit Clustering Tool

> Transparent KMeans-based corporate credit risk scoring using SEC EDGAR financials, scorecard-style risk features, private-company scoring, scenario analysis, and professional Excel/PDF reporting.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Data: SEC EDGAR](https://img.shields.io/badge/Data-SEC%20EDGAR-lightblue)](https://www.sec.gov/edgar/)

---

## Executive Summary

This project builds a transparent, modular credit risk scoring workflow based on unsupervised machine learning. It uses financial statement data from SEC EDGAR filings to train a KMeans clustering model on public-company financial profiles, then uses the trained model as a benchmark for scoring companies by relative credit quality.

The project is structured as a broader credit analytics toolkit supporting:

- public-company financial data extraction and feature engineering;
- KMeans-based credit risk clustering;
- scorecard-style risk features;
- private-company/manual company scoring;
- cluster profiling and label mapping;
- adjacent bucket diagnostics and outlook flags;
- rule-based credit guardrails;
- Excel scenario analysis; and
- professional PDF credit reports.

The output is not a formal agency credit rating. It is an analytical credit-risk screening framework designed to make financial risk segmentation more transparent, explainable, and reproducible.

---

## What This Project Does

The project converts company financial statements into interpretable credit risk features and groups companies into financial risk clusters. These clusters are then mapped into ordered credit-risk labels.

The core idea is simple:

1. collect structured financial statement data;
2. engineer credit-relevant financial ratios and risk features;
3. train a KMeans clustering model on comparable public-company data;
4. profile and label the resulting clusters;
5. score new companies against the trained benchmark;
6. produce analyst-facing diagnostics and reports.

The model does not rely on external rating agency labels. Instead, it derives credit quality groupings from the financial characteristics of the company universe.

---

## Current Model Version

The current production-style version is the **v3 scorecard EBITDA model**.

Core design choices:

| Area | Current Design |
|---|---|
| Primary model | KMeans |
| Number of clusters | 5 |
| Main segment | Non-financial companies |
| Main feature layer | Scorecard-style domain risk features |
| KMeans initialization | 500 `n_init` |
| Primary output | Ordered credit-risk label, not raw cluster ID |
| Private-company scoring | Supported |
| Reporting | CSV, Excel, scenario analysis, guardrails, PDF report |

The model intentionally does **not** treat raw KMeans cluster IDs as meaningful credit labels. Cluster IDs such as `0`, `1`, `2`, `3`, or `4` are arbitrary identifiers assigned by the algorithm. The business interpretation comes from the mapped risk rank and credit-risk label.

---

## Credit Risk Label Scale

The model reports a stable 1-5 risk label scale.

| Risk Rank | Label | Interpretation |
|---:|---|---|
| 1 | Strong relative credit profile | Low-risk profile relative to the model universe |
| 2 | Good credit profile | Sound credit profile with moderate weaknesses |
| 3 | Leveraged / elevated risk profile | Material leverage or operating risk, but not clearly distressed |
| 4 | Weak credit profile | Weak financial profile requiring close monitoring |
| 5 | Distressed / near-default proxy | Severe financial weakness, distress-like characteristics, or near-default proxy |

The scale is benchmark-relative. It should be interpreted as a model-derived credit risk signal, not as an agency rating equivalent.

---

## Methodology

### 1. Data Extraction - SEC EDGAR

Financial statement data is collected from SEC EDGAR filings for a broad universe of US public companies. The extraction process focuses on structured accounting concepts that can be mapped into credit-relevant financial metrics.

Typical raw inputs include:

- revenue;
- total assets;
- current assets;
- cash and cash equivalents;
- inventory;
- receivables;
- total liabilities;
- current liabilities;
- total debt;
- equity;
- net income;
- operating cash flow;
- interest expense;
- depreciation and amortization; and
- EBITDA or EBITDA proxy fields where available.

### 2. Feature Engineering

Raw accounting data is converted into financial ratios, warning flags, and scorecard-style risk features. The model is designed to handle incomplete financial inputs pragmatically, especially for private-company scoring, while still flagging lower feature coverage when data quality is weak.

Lower-level ratios include liquidity, leverage, profitability, operating cash-flow, and debt-serviceability measures. These are then transformed into higher-level risk dimensions used for clustering and scoring.

### 3. Scorecard Feature Design

The current model is not trained directly on dozens of raw accounting ratios. Instead, it uses interpretable domain-level risk features.

| Feature | Interpretation |
|---|---|
| `structural_distress_risk` | Gradient balance-sheet vulnerability based on equity buffer and liabilities/assets; hard distress flags remain separate guardrails |
| `earnings_risk` | Profitability weakness and negative earnings pressure |
| `operating_cashflow_risk` | Weak operating cash generation relative to both assets and debt |
| `liquidity_risk` | Short-term liquidity plus internal debt-repayment capacity from FCF/debt |
| `leverage_risk` | Balance-sheet leverage, equity buffer, and net debt/EBITDA pressure |
| `debt_service_risk` | Interest coverage, FCF/debt, EBITDA coverage, and debt/EBITDA |

This approach keeps the model explainable. Instead of clustering companies on a black box of raw ratios, the model clusters them on credit concepts that are easier to interpret and report.

### 4. Preprocessing

The preprocessing layer is designed to make financial ratios usable for distance-based clustering.

Typical steps include:

- denominator safety checks;
- missing value handling;
- weight-renormalized domain features, so a missing component (e.g. an EBITDA-based input for a negative-EBITDA issuer, or a debt-relative input for a debt-free issuer) reweights the remaining components instead of nulling the entire risk dimension;
- bounded risk-score transformation;
- optional outlier control for diagnostic ratios;
- segment filtering; and
- feature coverage diagnostics.

Because KMeans is distance-based, the model uses bounded and directionally consistent risk features so that distances between companies remain interpretable.

### 5. Clustering

KMeans is the primary clustering model. It is used because it is simple, explainable, stable enough for a structured scorecard workflow, and easy to connect to distance-based diagnostics.

Alternative clustering methods are also explored for comparison:

- Agglomerative clustering;
- DBSCAN; and
- dimensionality-reduction visual diagnostics such as PCA.

These alternatives are useful for validation and sensitivity analysis, but KMeans remains the main model for the current scorecard workflow.

### 6. Cluster Profiling and Label Mapping

After clustering, each cluster is profiled using median financial ratios, risk scores, and diagnostic statistics. The clusters are then ordered from strongest to weakest credit profile and mapped into the 1-5 risk label scale.

This is a critical design point: the raw KMeans cluster number is not the credit rating. The mapped risk rank and descriptive label are the analyst-facing outputs.

---

## Private Company Scoring

The trained model can be used to score a private or manually entered company by supplying core financial statement inputs.

Typical manual inputs include:

- company name;
- fiscal year;
- reporting currency;
- revenue;
- total assets;
- current assets;
- cash;
- receivables;
- inventory;
- equity;
- total liabilities;
- current liabilities;
- long-term debt;
- short-term debt;
- net income;
- operating cash flow;
- interest expense;
- depreciation/amortization; and
- EBITDA where available.

The scoring engine applies the same feature construction logic used in the model training pipeline. It then assigns the company to the closest trained cluster, calculates soft cluster affinities, measures distance to adjacent risk buckets, compares the company to cluster benchmarks, and produces reporting diagnostics.

---

## Guardrails and Analyst Diagnostics

The model output is supplemented with rule-based credit guardrails. These are not a replacement for clustering and are not used as automatic rating overrides. They are analyst-facing diagnostics designed to highlight specific credit red flags.

Examples include:

- high debt/assets;
- liabilities/assets above critical thresholds;
- weak interest coverage;
- weak EBITDA interest coverage;
- weak FCF/debt;
- current ratio below 1.0;
- quick ratio below 0.5;
- negative equity;
- liabilities exceeding assets;
- negative EBITDA; and
- low feature coverage.

Guardrails are important because a company can receive a reasonable cluster assignment while still having one or two severe financial weaknesses that require attention.

---

## Scenario Analysis

The private-company scoring workflow supports scenario testing. A base company profile can be stressed or improved by modifying key inputs such as revenue, EBITDA, debt, cash, operating cash flow, and liquidity.

Scenario analysis helps answer questions such as:

- How sensitive is the risk label to weaker profitability?
- How much debt reduction is needed to improve the credit profile?
- Does liquidity or leverage drive the score more strongly?
- Is the company close to an adjacent weaker or stronger bucket?

Scenario outputs can be exported to Excel and included in reporting packs.

---

## Outputs

The project can generate the following outputs:

- company-level scoring table;
- assigned risk label and risk rank;
- raw KMeans cluster ID for traceability;
- cluster affinity scores;
- near-default affinity;
- distance to assigned cluster;
- distance to adjacent stronger and weaker buckets;
- outlook flag and outlook reason;
- feature coverage percentage;
- warning flags;
- guardrail diagnostics;
- cluster comparison table;
- scenario analysis workbook;
- Excel report tabs; and
- professional PDF credit report.

The reporting layer is designed for review by analysts, finance professionals, credit users, and project evaluators. It is not only a model development output.

---

## Repository Structure

```text
corporate_credit_clustering_tool/
│
├── notebooks/
│   ├── 01_credit_clustering.ipynb
│   ├── 02_agglomerative_dbscan_credit_clustering_comparison.ipynb
│   ├── 03_private_company_credit_scoring_tool_feature_patch.ipynb
│   └── 04_obtain_model_training_data_EDGAR_communicated.ipynb
│
├── src/
│   ├── credit_clustering/
│   │   ├── config.py                  # feature lists, weights, thresholds, guardrail rules
│   │   ├── features.py                # single source of truth for feature engineering
│   │   ├── clustering.py              # segment clustering, k-evaluation, KMeans pipeline
│   │   ├── alternative_clustering.py  # Agglomerative / DBSCAN / PCA comparison
│   │   ├── scoring.py                 # private-company serving / scoring
│   │   ├── profiling.py               # cluster profiling and rating-style label mapping
│   │   ├── diagnostics.py             # adjacent-bucket outlook diagnostics
│   │   ├── guardrails.py              # rule-based credit guardrails
│   │   ├── artifacts.py               # build / validate / save / load model artifacts
│   │   └── edgar_concepts.py          # EDGAR / XBRL concept mapping
│   │
│   └── utils/
│       ├── credit_report_util.py      # Excel reporting
│       └── credit_pdf_report_util.py  # PDF credit report
│
├── docs/
├── inputs/
├── requirements.txt
├── LICENSE
└── README.md
```

The notebooks are intended to show the full workflow, while the `src/` package contains reusable project logic.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Kamend1/corporate_credit_clustering_tool.git
cd corporate_credit_clustering_tool
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Notebooks

Start Jupyter:

```bash
jupyter notebook notebooks/
```

The notebook *file numbers* reflect development history, not run order. The recommended execution order is:

1. `04_obtain_model_training_data_EDGAR_communicated.ipynb`  
   Extract and prepare the public-company EDGAR training panel.

2. `01_credit_clustering.ipynb`  
   Engineer features, train the KMeans model, profile clusters, and save the model artifact.

3. `03_private_company_credit_scoring_tool_feature_patch.ipynb`  
   Score a manually entered / private company, run scenarios, and produce report outputs.

4. `02_agglomerative_dbscan_credit_clustering_comparison.ipynb`  
   Compare KMeans with Agglomerative and DBSCAN for methodology validation.

---

## Example Private Company Scoring Workflow

A simplified scoring workflow looks like this:

```python
from pathlib import Path
import pandas as pd

from src.credit_clustering.scoring import (
    score_companies,
    infer_near_default_cluster_from_artifact,
)
from src.credit_clustering.artifacts import load_artifact

MODEL_PATH = Path("outputs/saved_models/nonfinancial_credit_scorecard_kmeans_k5_v3_clean.joblib")

artifact = load_artifact(MODEL_PATH)

manual_company = pd.DataFrame([
    {
        "company_name": "Manual 2025 Company",
        "fiscal_year": 2025,
        "revenue": 48_585_294,
        "assets": 29_721_275,
        "current_assets": 10_037_746,
        "cash": 34_805,
        "receivables": 2_208_235,
        "inventory": 7_794_706,
        "equity": 9_082_353,
        "current_liabilities": 12_722_353,
        "liabilities": 20_638_824,
        "long_term_debt": 7_910_588,
        "short_term_debt": 1_478_235,
        "net_income": 949_412,
    }
])

segment = "Non-financial"
near_default_cluster = infer_near_default_cluster_from_artifact(artifact, segment)

scored = score_companies(
    manual_company,
    artifact=artifact,
    segment=segment,
    temperature=1.0,
    fx_to_model_currency=1.0,
    min_denominator=None,
    near_default_cluster=near_default_cluster,
)

scored[[
    "company_name",
    "fiscal_year",
    "risk_rank",
    "cluster_label",
    "cluster_affinity",
    "near_default_affinity",
    "warning_flags",
]]
```

Adjacent-bucket outlook fields are added later by the diagnostics layer used in Notebook 03.

Exact imports may vary as the package is refined, but the intended workflow is stable: load artifact, prepare company inputs, score company, review diagnostics, export outputs.

---

## Use Cases

Potential use cases include:

- credit screening of company watchlists;
- private-company credit diagnostics;
- internal finance and CFO analysis;
- peer benchmarking;
- credit surveillance;
- early warning signal design;
- portfolio monitoring;
- academic or professional demonstration of unsupervised learning in credit risk;
- model explainability exercises; and
- professional reporting automation.

---

## Why KMeans?

KMeans is not the most sophisticated clustering algorithm, but it is a strong fit for this project because the objective is not only prediction. The objective is a transparent and explainable credit segmentation workflow.

KMeans provides:

- clear cluster centroids;
- intuitive distance-to-cluster diagnostics;
- stable reporting logic;
- simple comparison to adjacent risk buckets;
- compatibility with analyst-facing explanations; and
- easier communication to non-technical finance users.

The model is deliberately designed for interpretability and professional usability, not only for algorithmic complexity.

---

## Important Methodological Notes

### Cluster IDs Are Arbitrary

KMeans cluster numbers are algorithmic identifiers. They are not credit ratings. The same model structure can assign different numeric cluster IDs across runs. This is why the project maps clusters to ordered business labels.

### Private Companies Are Scored by Benchmarking

Private-company scoring works by comparing the manually entered company to the trained public-company financial-risk space. This is useful, but it also means that interpretation should consider differences between public-company and private-company financial structures.

---

## Limitations

This tool is a credit-risk screening and analytical support model. It is not a rating agency model and does not issue formal credit ratings.

Main limitations:

- labels are model-derived and benchmark-relative;
- public-company SEC data may not fully represent SME/private-company risk;
- sector effects are only partially captured;
- qualitative credit factors are not included;
- covenant packages, collateral, ownership support, refinancing access, and management quality are outside the model;
- missing financial inputs can reduce scoring reliability;
- EDGAR/XBRL data quality depends on issuer tagging consistency;
- KMeans cluster IDs are arbitrary and must not be interpreted directly;
- guardrails are diagnostics, not automatic rating overrides; and
- the model should support, not replace, professional credit judgment.

---

## Roadmap

Planned and potential future improvements:

- improve sector-specific calibration;
- add richer PDF report sections;
- add sensitivity analysis appendices;
- improve feature coverage diagnostics;
- add benchmark validation against external credit indicators;
- add optional Altman-style comparison layer;
- extend scenario templates;
- package the scoring workflow into a cleaner command-line or app interface;
- improve unit tests and reproducibility checks; and
- add more detailed documentation for configuration settings.

---

## License

Distributed under the Apache License 2.0. See [`LICENSE`](LICENSE) for details.

© 2026 Kamen Dimitrov, CFA

---

## Author

**Kamen Dimitrov, CFA**  
[KSB Analytica](https://ksbanalytica.com) | [GitHub](https://github.com/Kamend1)

---

## Disclaimer

This project is for educational, analytical, and research purposes. It does not provide investment advice, credit ratings, lending recommendations, or regulated financial services. Any model output should be reviewed by a qualified professional before being used in a real credit, investment, audit, or financing decision.
