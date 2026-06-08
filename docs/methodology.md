# Methodology

This document explains the current methodology of the Corporate Credit Clustering Tool.

The project is a domain-guided unsupervised machine learning workflow. It converts financial statements into bounded credit-risk features, applies KMeans clustering, profiles the resulting clusters, and maps them into a 1–5 credit-risk label scale.

---

## 1. Problem formulation

### Real-world problem

Many companies do not have public agency credit ratings. Even when ratings exist, they are not fully reproducible from public financial data. A transparent screening model can support credit analysis, education, benchmarking, and private-company diagnostics.

### Mathematical problem

Each company-year is represented as a six-dimensional vector:

```text
x_i = [
    structural_distress_risk,
    earnings_risk,
    operating_cashflow_risk,
    liquidity_risk,
    leverage_risk,
    debt_service_risk
]
```

Each feature is directionally aligned:

```text
0 = stronger / lower-risk
1 = weaker / higher-risk
```

The clustering task is to partition company-year observations into groups with similar financial-risk profiles.

---

## 2. Data source

The training universe is based on SEC EDGAR public-company financial statement data. The main model segment is non-financial companies.

Financial companies are excluded because banks, insurers, REITs, and other financial entities require different leverage, liquidity, and capital adequacy logic.

Typical raw inputs include revenue, assets, liabilities, equity, cash, receivables, inventory, current assets, current liabilities, long-term debt, short-term debt, net income, CFO, operating income, interest expense, capex, depreciation/amortization, and EBITDA where available.

---

## 3. Derived accounting values

The feature pipeline derives:

```text
total_debt = long_term_debt + short_term_debt
net_debt = total_debt - cash
fcf = CFO - |capex|
EBITDA = direct EBITDA or operating_income + depreciation_amortization
```

If both debt components are missing, total debt is treated as missing rather than zero.

---

## 4. Base financial ratios

| Ratio | Formula | Purpose |
|---|---|---|
| `liabilities_to_assets` | liabilities / assets | total obligation intensity |
| `debt_to_assets` | total debt / assets | interest-bearing debt intensity |
| `equity_to_assets` | equity / assets | equity cushion |
| `cash_to_assets` | cash / assets | cash buffer |
| `net_income_to_assets` | net income / assets | profitability |
| `cfo_to_assets` | CFO / assets | operating cash generation relative to asset base |
| `cfo_to_debt` | CFO / total debt | operating cash-flow debt capacity |
| `current_ratio` | current assets / current liabilities | current liquidity |
| `quick_ratio` | (cash + receivables) / current liabilities | liquid short-term coverage |
| `interest_coverage` | operating income / interest expense | EBIT-based coverage |
| `fcf_to_debt` | FCF / total debt | free-cash-flow debt repayment capacity |
| `ebitda_margin` | EBITDA / revenue | operating profitability before D&A |
| `debt_to_ebitda` | total debt / EBITDA | debt burden relative to EBITDA |
| `net_debt_to_ebitda` | net debt / EBITDA | net leverage |
| `ebitda_interest_coverage` | EBITDA / interest expense | EBITDA-based coverage |

Ratios are calculated with denominator protection. Invalid, missing, zero, or immaterial denominators return missing values rather than unstable ratios.

---

## 5. Bounded risk transformation

For bad-high metrics:

```text
risk = clip((x - low) / (high - low), 0, 1)
```

For bad-low metrics:

```text
risk = clip((good - x) / (good - bad), 0, 1)
```

This transforms heterogeneous accounting ratios into comparable risk components. The current thresholds use broad ranges to preserve gradient and reduce over-clipping.

---

## 6. Domain-level model features

### `leverage_risk`

```text
leverage_risk =
    0.30 × liabilities_risk
  + 0.25 × debt_load_risk
  + 0.20 × equity_buffer_risk
  + 0.25 × net_debt_to_ebitda_risk
```

### `liquidity_risk`

```text
liquidity_risk =
    0.35 × current_liquidity_risk
  + 0.30 × quick_liquidity_risk
  + 0.20 × debt_repayment_risk
  + 0.15 × cash_buffer_risk
```

### `earnings_risk`

```text
earnings_risk = profitability_risk
```

### `operating_cashflow_risk`

```text
operating_cashflow_risk =
    0.50 × cashflow_risk
  + 0.50 × cfo_to_debt_risk
```

### `debt_service_risk`

```text
debt_service_risk =
    0.35 × coverage_risk
  + 0.25 × fcf_risk
  + 0.25 × debt_to_ebitda_risk
  + 0.15 × ebitda_coverage_risk
```

Fallback when EBITDA diagnostics are unavailable:

```text
debt_service_risk_legacy =
    0.60 × coverage_risk
  + 0.40 × fcf_risk
```

### `structural_distress_risk`

```text
structural_distress_risk =
    0.60 × equity_buffer_risk
  + 0.40 × liabilities_risk
```

This is a gradient balance-sheet vulnerability measure. Hard flags such as negative equity and liabilities exceeding assets remain separate guardrail indicators.

---

## 7. Scorecard credit score

The scorecard score is a weighted average of the six domain risks:

| Domain | Weight |
|---|---:|
| leverage risk | 25% |
| debt-service risk | 25% |
| operating cash-flow risk | 20% |
| earnings risk | 15% |
| liquidity risk | 10% |
| structural vulnerability | 5% |

The scorecard score supports cluster ranking, interpretation, and reporting. It is not a formal rating or probability of default.

---

## 8. KMeans clustering

KMeans partitions observations into five clusters by minimizing within-cluster squared Euclidean distance:

```text
minimize Σ_j Σ_{x in C_j} ||x - μ_j||²
```

where `μ_j` is the centroid of cluster `j`.

Current configuration:

| Parameter | Value |
|---|---:|
| clusters | 5 |
| initialization | k-means++ |
| n_init | 500 |
| random_state | 42 |

KMeans is used because the feature space is compact, numeric, bounded, and interpretable.

---

## 9. Cluster profiling and labels

Raw KMeans cluster IDs are arbitrary. After fitting, clusters are profiled using medians and diagnostic metrics, then ranked from strongest to weakest.

| Rank | Label |
|---:|---|
| 1 | Strong relative credit profile |
| 2 | Good credit profile |
| 3 | Leveraged / elevated risk profile |
| 4 | Weak credit profile |
| 5 | Distressed / near-default proxy |

The label scale is model-relative and not an agency rating equivalent.

---

## 10. Private-company scoring

A private or manually entered company can be scored against the trained public-company benchmark by applying the same feature engineering logic, predicting the nearest cluster, calculating distances and affinities, and applying diagnostics and guardrails.

Private-company scoring is a benchmark comparison, not a substitute for full credit due diligence.

---

## 11. Validation strategy

Because this is unsupervised learning, validation relies on:

1. internal clustering metrics;
2. cluster size balance;
3. financial monotonicity across risk ranks;
4. representative-company review;
5. alternative clustering comparison;
6. component saturation checks;
7. missingness and feature coverage checks;
8. sensitivity analysis around thresholds and formula weights.

Financial interpretability matters more than a single mathematical metric.

---

## 12. Altman-style scores

Altman-style distress scores are not primary clustering features because they are already composite distress indicators. They are better used as external benchmarks, appendices, or sensitivity checks.

---

## 13. Summary

The methodology is a debt-capacity-aware, domain-guided KMeans framework. It combines financial statement feature engineering, bounded credit-risk transformations, KMeans clustering, post-hoc label mapping, private-company scoring, guardrails, scenario analysis, and professional reporting.
