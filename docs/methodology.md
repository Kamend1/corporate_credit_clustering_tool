# Methodology

## Purpose of this document

This document describes the technical and mathematical methodology of the Corporate Credit Clustering Tool. It is project documentation, not a replacement for the explanatory markdown that should be added directly inside Notebooks 01–04.

The notebook write-up should tell the story step by step for the SoftUni final project. This document gives the durable project-level reference.

---

## 1. Project framing

The project studies whether unsupervised machine learning can produce an interpretable corporate credit-risk segmentation from public financial statement data.

### Real-world problem

Many companies do not have external agency ratings. Even when ratings exist, they may be costly, lagged, sparse, or unsuitable for quick exploratory screening. A finance professional may still need a structured way to compare companies by leverage, liquidity, profitability, cash-flow strength, and debt-service ability.

### Mathematical problem

Each company-year observation is transformed into a six-dimensional bounded risk vector:

```text
x_i = [structural_distress_risk,
       earnings_risk,
       operating_cashflow_risk,
       liquidity_risk,
       leverage_risk,
       debt_service_risk]
```

The task is to partition these vectors into five groups using KMeans clustering, then rank the clusters from strongest to weakest using financial profiles.

### Machine learning task

This is an unsupervised learning problem. There are no ground-truth labels such as official ratings or observed defaults used for training. The model learns structure from the engineered financial-risk feature space.

The output is a model-relative credit-risk label, not a formal credit rating.

---

## 2. Data acquisition: Notebook 04

Raw financial data is obtained from SEC EDGAR filings through structured XBRL facts.

| Dimension | Project choice |
|---|---|
| Source | SEC EDGAR |
| Filing type | Annual financial statement facts, primarily 10-K / FY facts |
| Fiscal years | 2020–2025 in the current project version |
| Universe | US-listed companies with valid CIK/ticker data |
| Training focus | Non-financial companies |
| Minimum training asset filter | USD 1,000,000 |

Financial companies are excluded because their balance sheets are fundamentally different. Banks, insurers, and many real estate entities cannot be evaluated with the same leverage, EBITDA, liquidity, and working-capital ratios used for industrial companies.

Notebook 01 should be explained in the final project as a data engineering notebook: it builds the raw data foundation, applies safe execution flags, stores large generated outputs outside Git, and prepares a queryable DuckDB/parquet dataset.

---

## 3. Concept mapping and accounting normalization

SEC filers may use different XBRL tags for economically similar items. The project therefore relies on deterministic concept mapping: preferred concepts are selected first, with fallback concepts used when the preferred tag is missing.

Examples of normalized items:

| Economic item | Possible source concepts |
|---|---|
| Revenue | `Revenues`, `SalesRevenueNet`, `RevenueFromContractWithCustomerExcludingAssessedTax` |
| Assets | `Assets` |
| Liabilities | `Liabilities` |
| Equity | `StockholdersEquity`, equivalent equity concepts |
| Operating cash flow | `NetCashProvidedByUsedInOperatingActivities` |
| Capex | capital expenditure / payments to acquire property and equipment concepts |

The objective is not perfect accounting reconstruction for every filer. The objective is a robust, repeatable feature panel suitable for a broad clustering model.

---

## 4. Feature engineering

The same feature engineering logic is used for both model training and private-company scoring. This is critical: a company scored in Notebook 03 must be represented in the same feature space as the companies used to train the KMeans artifact in Notebook 02.

### 4.1 Derived accounting values

Before ratios are calculated, the model derives key accounting values:

```text
total_debt      = long_term_debt + short_term_debt
net_debt        = total_debt - cash
free_cash_flow  = operating_cash_flow - abs(capex)
ebitda          = direct EBITDA, if available,
                  otherwise operating_income + depreciation_amortization
```

If both long-term and short-term debt are missing, `total_debt` is treated as missing rather than zero. This avoids creating false low-leverage observations.

### 4.2 Base ratios

| Ratio | Formula | Credit interpretation |
|---|---|---|
| `liabilities_to_assets` | liabilities / assets | Balance-sheet leverage |
| `debt_to_assets` | total debt / assets | Interest-bearing debt load |
| `equity_to_assets` | equity / assets | Equity cushion |
| `cash_to_assets` | cash / assets | Immediate liquidity buffer |
| `net_income_to_assets` | net income / assets | Profitability relative to asset base |
| `cfo_to_assets` | CFO / assets | Operating cash generation |
| `current_ratio` | current assets / current liabilities | Short-term liquidity |
| `quick_ratio` | cash + receivables / current liabilities | Liquid-asset coverage |
| `interest_coverage` | operating income / interest expense | Ability to cover interest from EBIT |
| `fcf_to_debt` | free cash flow / total debt | Cash repayment capacity |
| `cfo_to_debt` | operating cash flow / total debt | Operating cash generation relative to debt burden |
| `debt_repayment_capacity` | free cash flow / total debt, debt-service calibrated | Debt repayment capacity used in liquidity/debt-service risk |
| `debt_to_ebitda` | total debt / EBITDA | Leverage relative to operating earnings |
| `net_debt_to_ebitda` | net debt / EBITDA | Net leverage |
| `ebitda_interest_coverage` | EBITDA / interest expense | EBITDA-based interest cushion |

All ratio divisions are protected by denominator rules. Very small denominators are converted to missing values rather than producing meaningless extreme ratios.

---

## 5. Bounded credit-risk transformation

Raw ratios have different units and scales. KMeans is distance-based, so directly clustering raw ratios would be unstable and hard to interpret. The project therefore maps raw ratios into bounded component risk scores on `[0, 1]`.

For ratios where higher values are worse:

```text
risk = clip((x - low) / (high - low), 0, 1)
```

For ratios where lower values are worse:

```text
risk = clip((good - x) / (good - bad), 0, 1)
```

Interpretation:

```text
0 = low risk for that component
1 = high risk for that component
```

This makes the feature space mathematically suitable for Euclidean distance and financially interpretable.

### Current threshold calibration philosophy

The current v3 scorecard EBITDA calibration uses analyst-defined thresholds that are broad enough to avoid excessive clipping, but still anchored in financially interpretable corporate-credit ranges.

| Area | Calibration logic |
|---|---|
| Leverage | Liability, debt, and equity-buffer thresholds distinguish normal leverage from balance-sheet pressure. |
| Liquidity | Current, quick, and cash-buffer thresholds are combined with debt-repayment capacity rather than treating liquidity as purely working-capital based. |
| Profitability and cash generation | Net-income/assets and CFO/assets thresholds use realistic corporate operating ranges rather than extreme profitability assumptions. |
| Debt service | Interest coverage and EBITDA-interest coverage use conservative “strong coverage” thresholds, so merely positive coverage may still carry moderate risk. |
| EBITDA leverage | Debt/EBITDA and net debt/EBITDA thresholds distinguish low leverage from elevated debt burden. |

---

## 6. Domain-level model features

Component risks are aggregated into six domain features.

| Domain feature | Construction logic |
|---|---|
| `leverage_risk` | liabilities risk, debt load risk, equity-buffer risk, and net debt/EBITDA pressure |
| `liquidity_risk` | current liquidity, quick liquidity, cash buffer, and FCF/debt repayment capacity |
| `earnings_risk` | profitability risk based on net income relative to assets |
| `operating_cashflow_risk` | operating cash-flow generation relative to assets and debt burden |
| `debt_service_risk` | EBIT interest coverage, FCF/debt, debt/EBITDA, and EBITDA interest coverage |
| `structural_distress_risk` | gradient balance-sheet vulnerability based on equity buffer and liabilities/assets |

Because these features are already bounded and directionally aligned, the v3 scorecard model does not apply StandardScaler before KMeans. Scaling would weaken the intended financial meaning of the bounded risk space.

---

## 7. Why Altman Z-score is not a primary model feature

Altman Z-score is a useful distress diagnostic, but it is not used as a primary KMeans feature in this project.

Reason: Altman Z-score is itself a composite distress model. Including it directly would partially pre-label the unsupervised feature space and duplicate several underlying dimensions already captured by the scorecard features.

Better use:

```text
Altman-style distress score = external benchmark / appendix diagnostic / sensitivity check
```

not:

```text
Altman-style distress score = core KMeans feature
```

This preserves the integrity of the unsupervised design.

---

## 8. KMeans clustering model

KMeans partitions observations into `k` clusters by minimizing within-cluster squared Euclidean distance:

```text
minimize Σ_j Σ_{x_i in C_j} ||x_i - μ_j||²
```

where:

| Symbol | Meaning |
|---|---|
| `x_i` | six-dimensional risk vector for company-year observation `i` |
| `C_j` | cluster `j` |
| `μ_j` | centroid of cluster `j` |
| `k` | number of clusters, currently 5 |

### Current configuration

| Parameter | Value | Rationale |
|---|---:|---|
| `k` | 5 | Practical five-tier risk scale |
| initialization | k-means++ | Improves initial centroid placement |
| `n_init` | 500 | Improves stability against local optima |
| `random_state` | 42 | Reproducibility |
| segment | Non-financial only | Keeps accounting interpretation consistent |

### Why KMeans is appropriate

KMeans is appropriate because the engineered feature space is:

- numeric;
- low-dimensional;
- bounded;
- directionally consistent;
- explainable;
- suitable for centroid-based interpretation.

The goal is not to predict default. The goal is to group companies with similar financial-risk profiles.

### KMeans limitation

KMeans assumes that Euclidean distance and centroid proximity are meaningful. It also tends to favor compact, roughly spherical clusters. Credit risk may not naturally form perfect spherical groups, so the project compares KMeans with hierarchical clustering and DBSCAN in Notebook 04.

---

## 9. k selection and internal metrics

The model evaluates different values of `k`, usually from 2 to 8, using internal clustering metrics.

| Metric | Preferred direction | Interpretation |
|---|---|---|
| Inertia | Lower | Within-cluster squared distance; always decreases as k increases. |
| Silhouette coefficient | Higher | How well observations fit their own cluster vs neighboring clusters. |
| Calinski-Harabasz index | Higher | Ratio of between-cluster to within-cluster dispersion. |
| Davies-Bouldin index | Lower | Average similarity between each cluster and its nearest alternative. |

The selected `k = 5` is not chosen mechanically from one metric. It reflects a balance between mathematical separation and interpretable credit-risk tiers.

---

## 10. Cluster ranking and labels

Raw KMeans cluster IDs are arbitrary. Cluster `0` is not necessarily low risk, and cluster `4` is not necessarily high risk.

The project ranks clusters post-hoc by their median `scorecard_credit_score`. The cluster with the lowest median score becomes risk rank 1; the highest becomes risk rank 5.

Recommended labels:

| Risk rank | Label |
|---:|---|
| 1 | Strong relative credit profile |
| 2 | Good credit profile |
| 3 | Leveraged / elevated risk profile |
| 4 | Weak credit profile |
| 5 | Distressed / near-default proxy |

These labels are not formal ratings. They are model-relative interpretations.

---

## 11. Scorecard credit score

The scorecard credit score is a weighted index of the six domain risk features:

```text
score = 100 × Σ(w_d × r_d) / Σ(w_d for available domains)
```

where:

| Symbol | Meaning |
|---|---|
| `r_d` | domain risk feature |
| `w_d` | domain weight |

The score is not the KMeans input. It is used to:

1. rank clusters;
2. explain a company’s continuous risk position;
3. support report interpretation.

---

## 12. Soft cluster affinity

After scoring a company, distances to all centroids are converted to soft affinities:

```text
a_j = exp(-d_j / T) / Σ_i exp(-d_i / T)
```

where:

| Symbol | Meaning |
|---|---|
| `d_j` | distance to cluster centroid `j` |
| `T` | temperature parameter |
| `a_j` | soft affinity to cluster `j` |

Affinity is a similarity measure. It is not probability of default and not statistical confidence in a rating.

---

## 13. Alternative methods: Notebook 02

Notebook 02 compares KMeans with other unsupervised approaches.

| Method | Why tested | Why not primary |
|---|---|---|
| Agglomerative clustering | Tests whether risk groups form a hierarchy | Harder to operationalize for new scoring; heavier computationally |
| DBSCAN | Detects dense regions and outliers | Sensitive to `eps`; may produce noise labels instead of business tiers |
| PCA visualization | Helps visualize six-dimensional structure in 2D/3D | Visualization only, not the model itself |
| KMeans | Interpretable centroids and easy scoring of new companies | Assumes Euclidean centroid structure |

PCA or Plotly visualizations should be described as communication tools, not as the core credit model.

---

## 14. Validation strategy for an unsupervised credit model

Because there are no labelled target ratings or default outcomes, validation is not based on accuracy.

The project uses or should use the following validation logic:

1. Internal clustering metrics: inertia, silhouette, Calinski-Harabasz, Davies-Bouldin.
2. Financial monotonicity: weaker clusters should show weaker leverage, liquidity, profitability, cash flow, and coverage.
3. Representative-company review: companies close to centroids should make business sense.
4. Alternative algorithm comparison: KMeans structure should be broadly compatible with hierarchical and density-based checks.
5. Sensitivity analysis: labels should remain reasonably stable under controlled perturbations.
6. Guardrail diagnostics: raw financial red flags should qualify model interpretation.

This is the correct validation posture for a domain-guided unsupervised financial model.

---

## 15. Private-company scoring

Notebook 03 applies the trained public-company benchmark model to manually entered or private-company financials.

The scoring sequence is:

```text
raw financial inputs
→ FX normalization and validation
→ derived accounting values
→ ratios and component risks
→ six domain-level features
→ KMeans cluster assignment
→ distance and affinity diagnostics
→ adjacent-bucket outlook
→ guardrails
→ Excel and PDF reporting
```

Private-company scoring is useful, but it is also partly out-of-distribution. The model is trained on SEC public-company data, so the report should disclose data-source limitations and avoid rating-equivalent claims.

---

## 16. Limitations summary

Key limitations:

- not a formal credit rating;
- no external default/rating validation in the current version;
- SEC public-company universe may not represent SMEs or non-US companies;
- survivorship bias under-represents companies that disappeared before the data pull;
- median imputation may centralize low-data observations;
- financial companies are excluded;
- qualitative factors, covenants, collateral, ownership support, and sector position are outside the model.

See `limitations.md` for the full discussion.

---

## 17. Final project communication note

In the SoftUni notebooks, the methodology should be explained directly in markdown cells, not only linked through this document.

Notebook communication should explicitly show:

- problem formulation;
- mathematical translation;
- data pipeline;
- feature engineering logic;
- KMeans objective;
- model evaluation;
- limitations;
- conclusion.

The `.md` files support the project, but the notebooks remain the primary final-exam communication layer.
