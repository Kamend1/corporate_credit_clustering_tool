# Methodology

This document describes the end-to-end pipeline of the Corporate Credit Clustering Tool — from raw SEC EDGAR data to a scored, labelled credit quality tier for any non-financial company.

---

## 1. Problem statement

Formal credit ratings cover only a small fraction of public companies. Agency ratings are updated infrequently and the underlying methodology is not publicly reproducible. This project derives **data-driven, unsupervised credit quality labels** for 7 000+ US-listed non-financial companies using structured financial data extracted directly from SEC EDGAR filings (10-K, 10-Q).

The output is not a formal credit rating. It is a transparent, replicable proxy that maps companies into five ordered risk buckets analogous to investment-grade through distressed designations.

---

## 2. Data acquisition (Notebook 01)

Raw financial data is pulled from SEC EDGAR using the `edgartools` library and the EDGAR XBRL API.

**Scope**

| Dimension | Value |
|---|---|
| Source | SEC EDGAR (10-K annual filings) |
| Fiscal years | 2020 – 2025 |
| Fiscal period filter | Full-year (`FY`) only |
| Universe | All US-listed companies with a valid CIK and ticker |
| Minimum total assets | USD 1 000 000 (training filter) |
| Excluded segments | Financial / Insurance / Real Estate (SIC-based) |

**Concept extraction**

Each filing is parsed for a fixed set of XBRL concepts defined in `edgar_concepts.py`. Where a company reports under multiple XBRL tags for the same economic item (e.g. `Revenues` vs `RevenueFromContractWithCustomerExcludingAssessedTax`), the concept map selects a deterministic preferred tag and falls back to alternatives in order of priority.

Raw facts are stored as Parquet chunks on Cloudflare R2 and queried locally via DuckDB, avoiding repeated large downloads.

**Incremental refresh**

Notebook 01 is designed for incremental operation. Execution flags (`RUN_*`) guard every EDGAR-calling cell, and a maximum-ticker cap (`MAX_TICKERS_FOR_DOWNLOAD`) enables small test runs without full dataset rebuilds.

---

## 3. Feature engineering (`src/credit_clustering/features.py`)

Raw accounting items are transformed into credit-relevant features through a deterministic, multi-step pipeline. The same function — `engineer_private_company_features` — is used for both model training (Notebook 02) and private-company scoring (Notebook 03), ensuring feature consistency.

### 3.1 Derived accounting values

Before any ratios are computed, several composite items are constructed:

```
total_debt         = long_term_debt + short_term_debt
net_debt           = total_debt − cash
free_cash_flow     = CFO − |capex|
ebitda             = direct EBITDA (if available)
                   OR operating_income + depreciation_amortization
```

If both `long_term_debt` and `short_term_debt` are missing, `total_debt` is set to NaN rather than zero.

### 3.2 Base financial ratios

| Ratio | Numerator | Denominator |
|---|---|---|
| `liabilities_to_assets` | liabilities | assets |
| `equity_to_assets` | equity | assets |
| `debt_to_assets` | total_debt | assets |
| `debt_to_equity` | total_debt | equity |
| `cash_to_assets` | cash | assets |
| `net_income_to_assets` | net_income | assets |
| `cfo_to_assets` | CFO | assets |
| `revenue_to_assets` | revenue | assets |
| `current_ratio` | assets_current | liabilities_current |
| `quick_ratio` | cash + receivables | liabilities_current |
| `log_assets` | log1p(assets) | — |

Denominators are protected by `safe_divide()`, which returns NaN whenever the denominator is zero, negative (for cases where sign matters), or below the configurable `SME_MIN_DENOMINATOR` threshold (default: 1 000). This threshold is intentionally small so that private-company scoring is not blocked on valid but modest interest-expense figures.

### 3.3 EBITDA diagnostics

| Metric | Formula |
|---|---|
| `ebitda_margin` | EBITDA / revenue |
| `debt_to_ebitda` | total_debt / EBITDA |
| `net_debt_to_ebitda` | net_debt / EBITDA |
| `ebitda_interest_coverage` | EBITDA / \|interest_expense\| |

Where EBITDA ≤ 0, the leverage ratios `debt_to_ebitda` and `net_debt_to_ebitda` are set to NaN rather than producing a misleading negative value. All EBITDA columns are nulled for financial companies (SIC-flagged), where EBITDA is not a meaningful concept.

### 3.4 Bounded risk factors (component level)

Each base ratio is transformed into a directional risk score on [0, 1] using a piecewise linear mapping:

**`linear_risk_bad_high(x, low, high)`** — used where higher values indicate more risk (leverage, debt load):

$$r = \text{clip}\!\left(\frac{x - \text{low}}{\text{high} - \text{low}},\; 0,\; 1\right)$$

**`linear_risk_bad_low(x, good, bad)`** — used where lower values indicate more risk (coverage, liquidity):

$$r = \text{clip}\!\left(\frac{\text{good} - x}{\text{good} - \text{bad}},\; 0,\; 1\right)$$

A score of **0 means no risk detected** for that dimension; **1 means maximum risk** within the calibrated range. Values outside the calibrated range are clipped to [0, 1]; they do not extrapolate. Thresholds are defined in `config.py` and documented in the [User Guide](user_guide_config.md).

### 3.5 Domain risk features (model inputs)

Fourteen component risks are aggregated into six domain-level features. These are the **direct inputs to the KMeans clustering model**.

| Domain feature | Sub-components and weights |
|---|---|
| `leverage_risk` | 0.40 × liabilities_risk + 0.35 × debt_load_risk + 0.25 × equity_buffer_risk |
| `liquidity_risk` | 0.45 × current_liquidity_risk + 0.40 × quick_liquidity_risk + 0.15 × cash_buffer_risk |
| `earnings_risk` | profitability_risk (1:1) |
| `operating_cashflow_risk` | cashflow_risk (1:1) |
| `debt_service_risk` | 0.35 × coverage_risk + 0.25 × fcf_risk + 0.25 × debt_to_ebitda_risk + 0.15 × ebitda_coverage_risk (falls back to 0.60 × coverage_risk + 0.40 × fcf_risk when EBITDA is unavailable) |
| `structural_distress_risk` | max(negative_equity_flag, liabilities_exceed_assets_flag) |

Because all domain features are already bounded [0, 1] and directionally comparable, **no StandardScaler is applied before clustering**. Scaling would compress the variation within this intentionally bounded space and obscure the credit-relevant differences the features encode.

### 3.6 Scorecard credit score

A weighted sum of the six domain features produces a composite `scorecard_credit_score` on [0, 100], where 0 is lowest risk and 100 is highest. Weights are applied proportionally to available features so that a company with one or more missing domains is not penalised beyond the data gap itself.

Domain weights: leverage 25%, liquidity 20%, operating CF 20%, earnings 15%, debt service 15%, structural distress 5%.

The scorecard score is used exclusively for **post-hoc cluster ranking and interpretation**; it is not a KMeans input.

---

## 4. Preprocessing

Before clustering:

1. **Winsorisation** — optional; controlled by `winsor_caps` parameter in `engineer_private_company_features`. In the v3 training run, this is `None` (no winsorisation). Winsorisation caps are supported for private-company scoring where user-supplied inputs may lie far outside the training distribution.

2. **Missing-value imputation** — median imputation via scikit-learn `SimpleImputer` is applied inside the KMeans pipeline. Imputation uses the median of each feature across the training segment, not a global median.

3. **Row-level coverage filter** — rows with fewer than `max(DEFAULT_MIN_FEATURES, ceil(n_features × DEFAULT_ROW_FEATURE_COVERAGE))` non-null model features are excluded from clustering. With six features and the default 60% coverage threshold, a row needs at least four non-null features to be clustered.

4. **Feature availability filter** — a feature is dropped if it is entirely unavailable across the segment (column is all-NaN). By default (`DEFAULT_MIN_FEATURE_COVERAGE = 0.0`) partial availability is accepted, and the imputer handles the gaps.

---

## 5. Clustering model (`src/credit_clustering/clustering.py`)

### 5.1 Algorithm

The primary model is **KMeans** with k-means++ initialisation. The objective function minimises within-cluster inertia:

$$\underset{C_1,\ldots,C_k}{\arg\min} \sum_{j=1}^{k} \sum_{\mathbf{x} \in C_j} \left\|\mathbf{x} - \boldsymbol{\mu}_j\right\|^2$$

where $\boldsymbol{\mu}_j$ is the centroid of cluster $C_j$ and $\|\cdot\|$ is the Euclidean norm in the six-dimensional risk-feature space.

**Configuration**

| Parameter | Value | Rationale |
|---|---|---|
| `k` | 5 | Five tiers broadly mirroring IG-high / IG-core / crossover / speculative / distressed |
| `init` | k-means++ | Reduces sensitivity to random centroid initialisation |
| `n_init` | 500 | Runs 500 independent initialisations; the one with lowest inertia is kept |
| `random_state` | 42 | Reproducibility |

### 5.2 Segment isolation

The model is trained exclusively on non-financial companies (segment = "Non-financial", controlled by `financial_flag` column derived from SIC codes). Financial companies are excluded because leverage and EBITDA metrics are structurally different for banks and insurers, and the risk thresholds calibrated for industrial companies are not appropriate for them.

### 5.3 Alternative algorithms (Notebook 04)

Agglomerative hierarchical clustering and DBSCAN are benchmarked against the KMeans result on a reproducible sample. These comparisons test whether the five risk groups behave like a genuine hierarchical ladder and whether density-based methods identify the same distressed outlier region. Both methods are computationally heavier than KMeans and are run on a downsampled frame to keep the notebook practical.

### 5.4 k selection

The selected k = 5 is evaluated against k ∈ {2, …, 8} using three internal metrics:

| Metric | Preferred direction | What it measures |
|---|---|---|
| Silhouette coefficient | Higher is better | Separation relative to cohesion |
| Calinski-Harabász index | Higher is better | Ratio of between-cluster to within-cluster dispersion |
| Davies-Bouldin index | Lower is better | Average ratio of within-cluster scatter to between-cluster distance |

The k-range sweep is in Notebook 02 (Section 6) and saved to `cluster_k_tests_v3_by_financial_flag.csv`.

---

## 6. Cluster profiling and labelling (`src/credit_clustering/profiling.py`)

After clustering, each cluster is characterised by:

- Median values of all INTERPRET_FEATURES (ratios, risk scores, size flags)
- Count of issuer-year observations and unique issuers
- Industry / sector composition
- Representative tickers (companies closest to the centroid in feature space, selected by Euclidean distance to the cluster median)

Clusters are ranked within each segment from lowest to highest `median_scorecard_credit_score`. The five resulting ranks receive the following default labels:

| Rank | Label |
|---|---|
| 1 | 1 — Low risk / investment-grade-like |
| 2 | 2 — Moderate risk / lower-investment-grade-like |
| 3 | 3 — Elevated risk / leveraged |
| 4 | 4 — High risk / speculative |
| 5 | 5 — Distressed / near-default proxy |

These labels are interpretive aids. They are not formal credit ratings and have not been calibrated against agency rating transitions.

---

## 7. Artifact persistence (`src/credit_clustering/artifacts.py`)

The fitted model and its associated metadata are serialised to a single `.joblib` file:

```
saved_models/nonfinancial_credit_scorecard_kmeans_k5_v3.joblib
```

The artifact dictionary contains:

- `pipeline` — fitted sklearn Pipeline (imputer + KMeans)
- `feature_cols` — ordered list of model input features
- `cluster_labels` — `{cluster_id: label}` mapping
- `risk_rank` — `{cluster_id: rank}` mapping
- `cluster_profile_ranked` — full cluster summary table
- `scorecard_domain_weights` — weights used for the composite score
- `winsor_caps` — training-time caps (None for v3)
- `artifact_version`, `notes`, training metadata

The artifact is the only file Notebook 03 and the scoring utility require at inference time.

---

## 8. Scoring private companies (Notebook 03, `src/credit_clustering/scoring.py`)

Any company — public, private, simulated — can be scored against the trained model by supplying raw financial inputs. The `score_companies()` function:

1. Engineers features using the same `engineer_private_company_features` function used in training.
2. Predicts the assigned KMeans cluster via `pipeline.predict()`.
3. Computes Euclidean distances to all cluster centroids via `pipeline.transform()`.
4. Converts distances to **soft cluster affinities** using an exponential kernel: $a_j = \exp(-d_j / T) / \sum_i \exp(-d_i / T)$, where $T$ is the temperature parameter (default 1.0).
5. Computes `near_default_affinity` — the soft affinity for the most distressed cluster.
6. Adds `warning_flags` for structural anomalies (negative equity, assets below model threshold, coverage below 1×, etc.).

Notebook 03 also runs **adjacent-bucket diagnostics** via `diagnostics.py`, which computes distances to the next-higher and next-lower risk buckets and assigns an outlook flag (Positive / Neutral / Negative) based on the relative distance ratios.

---

*See also: [Model Interpretation](model_interpretation.md) | [Limitations](limitations.md) | [Sensitivity Analysis](sensitivity_analysis.md)*
