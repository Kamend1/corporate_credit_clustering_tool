# User Guide: Configuration Reference

This document is the authoritative reference for every configurable parameter in `src/credit_clustering/config.py`. It explains what each constant controls, why its default was chosen, and when you might want to change it.

---

## 1. Model feature lists

### `SCORECARD_CLUSTER_FEATURES`
The six domain-level features passed to the KMeans model as input. **Do not change this list without retraining the model.** The saved artifact's pipeline and centroids are tied to exactly this feature set.

```python
SCORECARD_CLUSTER_FEATURES = [
    "structural_distress_risk",
    "earnings_risk",
    "operating_cashflow_risk",
    "liquidity_risk",
    "leverage_risk",
    "debt_service_risk",
]
```

### `SCORECARD_COMPONENT_FEATURES`
The fourteen intermediate risk scores that feed into the six domain features. These are diagnostic outputs included in the full scored dataframe but are **not** direct KMeans inputs.

### `INTERPRET_FEATURES`
The list of columns included in cluster profile tables and median comparisons. Covers both raw ratios and risk scores. Safe to extend for reporting purposes.

---

## 2. Denominator and size thresholds

### `SME_MIN_DENOMINATOR` (default: `1_000`)
The minimum absolute denominator value accepted by `safe_divide()`. Denominators smaller than this are treated as zero and return NaN.

This value is intentionally small to support private-company and SME scoring, where interest expense of e.g. $5 000 is economically valid even though it is far below public-company materiality. Raising this threshold will suppress more ratios for smaller companies; lowering it further increases the risk of meaningless divisions.

### `PUBLIC_COMPANY_MIN_ASSETS` (default: `1_000_000`)
The training-time minimum total assets filter applied in Notebook 01. Companies below this threshold were excluded from the training universe. This constant is also used by `make_warning_flags()` to raise `assets_below_model_threshold` on scored companies.

**Do not lower this for inference** unless you intentionally want to score micro-companies against a model not calibrated for them.

### `SMALL_COMPANY_ASSET_LIMIT` / `MID_COMPANY_ASSET_LIMIT` / `LARGE_COMPANY_ASSET_LIMIT`
Asset size band boundaries used exclusively for diagnostics and reporting labels (the `asset_size_band` column). They do not affect clustering.

| Band | Range |
|---|---|
| Small | assets < $5M |
| Mid | $5M ≤ assets < $50M |
| Large | $50M ≤ assets < $100B |
| Very large | assets ≥ $100B |

---

## 3. Clustering parameters

### `DEFAULT_N_CLUSTERS` (default: `5`)
Number of KMeans clusters. Changing this requires full retraining.

### `DEFAULT_N_INIT` (default: `500`)
Number of independent KMeans initialisations. The run with lowest inertia is kept. 500 is high by library defaults (scikit-learn default is 10) and ensures stable centroid placement across runs. Reduce to 50–100 for exploratory work; keep at 500 or higher for artifact production.

### `DEFAULT_RANDOM_STATE` (default: `42`)
Seeds both KMeans initialisation and any other random operations. Change only if you want to verify that results are not seed-dependent.

### `DEFAULT_MIN_ROWS_PER_SEGMENT` (default: `500`)
Minimum number of rows required for a segment to be clustered. Segments below this threshold are skipped. Prevents clustering on thin data.

### `DEFAULT_MIN_FEATURES` (default: `4`)
Minimum number of non-null model features required per row to include that row in clustering. Works together with `DEFAULT_ROW_FEATURE_COVERAGE`.

### `DEFAULT_ROW_FEATURE_COVERAGE` (default: `0.60`)
Minimum share of model features that must be non-null per row. With six features, the effective minimum is `max(4, ceil(6 × 0.60)) = 4`. Rows with fewer than 4 available features are excluded before clustering.

### `DEFAULT_MIN_FEATURE_COVERAGE` (default: `0.0`)
Minimum non-null coverage required **per feature column** across the segment. At 0.0, a feature is retained even if it is almost entirely missing (imputation handles the gaps). Raise to e.g. 0.30 to drop features that are rarely available, which may improve imputation quality at the cost of information loss.

---

## 4. Risk thresholds

`RISK_THRESHOLDS` defines the calibration points for each piecewise linear component risk mapping. All monetary thresholds are in the model's training currency (USD, as reported in EDGAR).

### Leverage thresholds

| Metric | Parameter | Value | Meaning |
|---|---|---|---|
| `liabilities_to_assets` | `low` | 0.45 | At or below 45% liabilities/assets: zero liabilities_risk |
| | `high` | 1.00 | At or above 100%: maximum liabilities_risk (liabilities ≥ assets) |
| `debt_to_assets` | `low` | 0.25 | At or below 25% debt/assets: zero debt_load_risk |
| | `high` | 0.85 | At or above 85%: maximum debt_load_risk |
| `equity_to_assets` | `good` | 0.40 | ≥40% equity/assets: zero equity_buffer_risk |
| | `bad` | 0.00 | ≤0% equity/assets: maximum risk (negative equity) |

### Liquidity thresholds

| Metric | Parameter | Value | Meaning |
|---|---|---|---|
| `current_ratio` | `good` | 2.00 | ≥2×: zero current_liquidity_risk |
| | `bad` | 0.75 | ≤0.75×: maximum risk |
| `quick_ratio` | `good` | 1.00 | ≥1×: zero quick_liquidity_risk |
| | `bad` | 0.25 | ≤0.25×: maximum risk |
| `cash_to_assets` | `good` | 0.10 | ≥10% cash/assets: zero cash_buffer_risk |
| | `bad` | 0.01 | ≤1%: maximum risk |

### Earnings and cash flow thresholds

| Metric | Parameter | Value | Meaning |
|---|---|---|---|
| `net_income_to_assets` | `good` | 0.05 | ≥5% ROA: zero profitability_risk |
| | `bad` | −0.05 | ≤−5% ROA: maximum risk |
| `cfo_to_assets` | `good` | 0.08 | ≥8% CFO/assets: zero cashflow_risk |
| | `bad` | −0.03 | ≤−3%: maximum risk |

### Debt service thresholds

| Metric | Parameter | Value | Meaning |
|---|---|---|---|
| `interest_coverage` | `good` | 3.00 | ≥3×: zero coverage_risk |
| | `bad` | 1.00 | ≤1×: maximum risk |
| `fcf_to_debt` | `good` | 0.15 | ≥15% FCF/debt: zero fcf_risk |
| | `bad` | −0.10 | ≤−10%: maximum risk |
| `ebitda_margin` | `good` | 0.20 | ≥20% margin: zero ebitda_margin_risk |
| | `bad` | 0.00 | ≤0%: maximum risk |
| `debt_to_ebitda` | `low` | 2.0 | ≤2×: zero debt_to_ebitda_risk |
| | `high` | 6.0 | ≥6×: maximum risk |
| `net_debt_to_ebitda` | `low` | 1.5 | ≤1.5×: zero net_debt_to_ebitda_risk |
| | `high` | 5.0 | ≥5×: maximum risk |
| `ebitda_interest_coverage` | `good` | 4.0 | ≥4×: zero ebitda_coverage_risk |
| | `bad` | 1.5 | ≤1.5×: maximum risk |

**To recalibrate:** change the values in `RISK_THRESHOLDS`, re-run `engineer_private_company_features()` on the training panel, and retrain the KMeans model. Changing thresholds without retraining will produce features that are inconsistent with the saved centroids.

---

## 5. Domain weights

### `SCORECARD_DOMAIN_WEIGHTS`

```python
SCORECARD_DOMAIN_WEIGHTS = {
    "leverage_risk":           0.25,
    "liquidity_risk":          0.20,
    "earnings_risk":           0.15,
    "operating_cashflow_risk": 0.20,
    "debt_service_risk":       0.15,
    "structural_distress_risk":0.05,
}
```

These weights affect only the `scorecard_credit_score` composite index, not the KMeans feature space. Changing them changes the numeric score and the post-hoc cluster ranking (because clusters are ranked by median scorecard score) but does **not** change which cluster each company is assigned to.

To change these weights for production scoring, update the constant and re-save the artifact with the new `scorecard_domain_weights` key.

---

## 6. Profiling defaults

### `DEFAULT_RATING_STYLE_LABELS`

```python
DEFAULT_RATING_STYLE_LABELS = {
    1: "1 - Low risk / investment-grade-like",
    2: "2 - Moderate risk / lower-investment-grade-like",
    3: "3 - Elevated risk / leveraged",
    4: "4 - High risk / speculative",
    5: "5 - Distressed / near-default proxy",
}
```

Can be overridden in `add_rating_style_labels()` via the `label_by_rank` argument. Useful for domain-specific terminology without changing any code.

### `DEFAULT_EXTREME_QUANTILES`
Quantile levels used in `build_feature_extremes()` for distribution diagnostics: `(0.001, 0.01, 0.05, 0.50, 0.95, 0.99, 0.999)`. Extend as needed for tighter tail inspection.

---

## 7. Scoring defaults

### `DEFAULT_SCORING_TEMPERATURE` (default: `1.0`)
Controls the sharpness of the soft affinity distribution. Lower values concentrate affinity on the nearest cluster; higher values spread affinity more evenly. Practical range: 0.3 (very sharp) to 3.0 (very flat). Do not change without documenting the impact on `near_default_affinity` comparability.

### `DEFAULT_FX_TO_MODEL_CURRENCY` (default: `1.0`)
Multiplier applied to all monetary columns before ratio computation. Set to the relevant USD exchange rate when scoring companies that report in a non-USD currency. Ratios are dimensionless, so only the FX consistency across numerator and denominator matters — but `log_assets` and asset size bands are in model currency and will be misleading if the wrong multiplier is used.

### `DEFAULT_SCORING_MIN_DENOMINATOR` (default: `SME_MIN_DENOMINATOR = 1_000`)
Passed to `engineer_private_company_features()` for private-company scoring. Lower than the public-company equivalent to support SME-scale interest expenses.

---

*See also: [Methodology](methodology.md) | [Sensitivity Analysis](sensitivity_analysis.md)*
