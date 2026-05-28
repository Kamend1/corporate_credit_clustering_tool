# Corporate Credit Clustering Tool

> **Unsupervised credit quality labeling for 7,000+ public companies using SEC EDGAR data**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Data: SEC EDGAR](https://img.shields.io/badge/Data-SEC%20EDGAR-lightblue)](https://www.sec.gov/edgar/)

---

## Overview

This project applies unsupervised machine learning — primarily clustering — to assign **credit quality labels** to corporate entities, without relying on pre-existing rating agency scores. Financial data for more than **7,000 US public companies** is scraped directly from the SEC's [EDGAR](https://www.sec.gov/edgar/) database, processed into credit-relevant features, and passed through a clustering pipeline that groups companies by financial risk profile.

The resulting cluster labels serve as proxy credit quality buckets (analogous to investment-grade vs. speculative-grade designations), enabling portfolio screening, peer benchmarking, and exploratory credit analysis at scale — entirely data-driven and model-derived.

---

## Motivation

Traditional credit ratings are:
- **Sparse** — only a fraction of public companies carry agency ratings
- **Lagging** — often updated infrequently relative to financial disclosures
- **Opaque** — methodology not fully transparent

By leveraging structured financial filings from EDGAR and unsupervised ML, this tool produces **timely, replicable, and transparent** credit quality signals across a much broader universe of companies.

---

## Project Structure

```
corporate_credit_clustering_tool/
│
├── notebooks/                         # End-to-end Jupyter notebooks
│   ├── 01_data_extraction.ipynb       # EDGAR scraping & raw data assembly
│   ├── 02_feature_engineering.ipynb   # Credit ratio calculation & preprocessing
│   ├── 03_clustering.ipynb            # Model training, tuning & evaluation
│   └── 04_analysis_outputs.ipynb      # Cluster profiling & visualisations
│
├── src/                               # Modular Python source code
│   ├── edgar_scraper.py               # EDGAR data extraction utilities
│   ├── feature_builder.py             # Financial ratio computation
│   ├── preprocessor.py                # Scaling, imputation, PCA
│   └── clustering_engine.py           # Clustering models & evaluation metrics
│
├── credit_clustering_outputs_v3/      # Persisted outputs (latest run)
│   ├── clustered_companies.csv        # Company-level cluster assignments
│   ├── cluster_profiles.csv           # Aggregate statistics per cluster
│   └── figures/                       # Charts and visualisations
│
├── .gitignore
├── LICENSE
└── README.md
```

> **Note:** The folder names above reflect the actual repository structure. Specific notebook and script filenames within `notebooks/` and `src/` may differ slightly from those shown.

---

## Methodology

### 1. Data Extraction — SEC EDGAR
Raw financial statement data is pulled from SEC EDGAR filings (10-K, 10-Q) using the EDGAR full-text search and XBRL APIs. Coverage spans **7,000+ US-listed public companies** across all major sectors.

Key data items extracted include:
- Total revenue and EBITDA
- Total assets, total debt, and equity
- Interest expense and cash from operations
- Current assets and current liabilities

### 2. Feature Engineering
Raw financials are transformed into **credit-relevant ratios** that proxy leverage, coverage, liquidity, and profitability:

| Feature | Description |
|---|---|
| Net Debt / EBITDA | Core leverage metric |
| EBIT / Interest Expense | Interest coverage ratio |
| FFO / Total Debt | Cash flow-based debt serviceability |
| Current Ratio | Short-term liquidity |
| Debt / Total Assets | Balance sheet leverage |
| EBITDA Margin | Operational efficiency |
| Revenue Growth (YoY) | Business trajectory |

### 3. Preprocessing
- **Outlier winsorisation** at the 1st/99th percentile
- **StandardScaler** normalisation across all features
- **PCA** for dimensionality reduction and noise suppression prior to clustering

### 4. Clustering
Multiple algorithms are benchmarked to identify the most stable and interpretable groupings:

- **K-Means** (primary) — optimised via elbow method and silhouette score
- **Agglomerative Hierarchical Clustering** — dendrogram analysis for cluster count selection
- **DBSCAN** — used to identify outlier/distressed entities

Cluster count is tuned to reflect recognisable credit quality tiers (e.g., 5–7 clusters mapping broadly from high-grade to distressed).

### 5. Cluster Profiling & Labelling
Each cluster is characterised by median financial ratios and assigned a qualitative credit quality label:

| Cluster | Label | Profile |
|---|---|---|
| 1 | **Investment Grade — High Quality** | Low leverage, strong coverage, stable cash flows |
| 2 | **Investment Grade — Core** | Moderate leverage, adequate coverage |
| 3 | **Crossover / BBB-** | Elevated leverage, volatile margins |
| 4 | **Speculative Grade** | High leverage, weak coverage, limited liquidity |
| 5 | **Highly Leveraged / Distressed** | Very high debt load, negative FCF signals |

*Labels are indicative and derived from cluster financial characteristics, not agency ratings.*

---

## Getting Started

### Prerequisites

```bash
Python 3.9+
pip install -r requirements.txt
```

Key dependencies:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
requests
beautifulsoup4
sec-edgar-downloader
jupyter
```

### Installation

```bash
git clone https://github.com/Kamend1/corporate_credit_clustering_tool.git
cd corporate_credit_clustering_tool
pip install -r requirements.txt
```

### Running the Pipeline

Execute the notebooks in order:

```bash
jupyter notebook notebooks/
```

Or run each stage independently via the `src/` modules:

```python
from src.edgar_scraper import fetch_filings
from src.feature_builder import build_credit_features
from src.clustering_engine import run_clustering

df_raw = fetch_filings(ticker_list)
df_features = build_credit_features(df_raw)
df_clustered = run_clustering(df_features, n_clusters=6)
```

Pre-computed outputs from the latest run are available in `credit_clustering_outputs_v3/clustered_companies.csv`.

---

## Outputs

The `credit_clustering_outputs_v3/` folder contains the most recent run results:

- **`clustered_companies.csv`** — full company universe with cluster ID, credit label, and key ratios
- **`cluster_profiles.csv`** — median and interquartile statistics per cluster
- **Visualisations** — PCA scatter plots (colour-coded by cluster), radar charts per cluster profile, silhouette plots

---

## Use Cases

- **Portfolio screening** — filter a watchlist by derived credit quality tier
- **Peer benchmarking** — identify a company's closest financial comparables
- **Credit surveillance** — detect cluster migration over time as a deterioration signal
- **Research** — academic or practitioner research into unsupervised credit modelling

---

## Limitations

- Labels are **model-derived**, not agency-validated; they should be used as a starting point for analysis, not as a substitute for formal credit assessment
- EDGAR data quality depends on consistent XBRL tagging across filers; some companies may have incomplete feature vectors and are excluded or imputed
- The model does not incorporate **qualitative factors** (management quality, industry dynamics, covenant structures)

---

## License

Distributed under the GNU General Public License v3.0. See [`LICENSE`](LICENSE) for details.

© 2026 Kamen Dimitrov, CFA

---

## Author

**Kamen Dimitrov, CFA**  
[KSB Analytica](https://ksbanalytica.com) | [GitHub](https://github.com/Kamend1)

---

*Contributions, issues, and feature requests are welcome.*
