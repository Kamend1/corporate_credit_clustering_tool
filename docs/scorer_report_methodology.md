# Scorer Report Methodology

This document explains the mechanics behind the private-company scoring workflow in Notebook 03, including how raw financial inputs are converted into a scored report, what each section of that report means, and how the scenario analysis and outlook diagnostics are constructed.

---

## 1. Overview

Notebook 03 (`03_private_company_credit_scoring_tool_feature_patch.ipynb`) applies the trained KMeans model — serialised as a `.joblib` artifact — to score any company whose financials can be expressed in the input schema. Typical use cases are:

- Scoring a private company against the public-company benchmark model.
- Re-scoring a previously rated public company on updated financials.
- Running stress scenarios on a base set of financials to observe cluster migration.
- Generating a structured credit quality report for audit or advisory purposes.

The entire scoring chain is implemented in `src/credit_clustering/scoring.py` and `src/credit_clustering/diagnostics.py`.

---

## 2. Input schema

The minimum required input is a dictionary (or DataFrame row) containing the company's raw financial statement items. All monetary values must be in the same currency and consistent unit (e.g. all in thousands of USD). Use `DEFAULT_FX_TO_MODEL_CURRENCY` to convert non-USD inputs.

| Field | Required | Notes |
|---|---|---|
| `assets` | Yes | Total assets |
| `liabilities` | Yes | Total liabilities |
| `equity` | Yes | Total equity (can be negative) |
| `revenue` | Yes | Total revenue / net sales |
| `net_income` | Yes | Net income / profit after tax |
| `cfo` | Yes | Cash from operations (operating cash flow) |
| `cash` | Recommended | Cash and cash equivalents |
| `long_term_debt` | Recommended | Long-term debt and finance leases |
| `short_term_debt` | Recommended | Short-term borrowings and current portion of LT debt |
| `assets_current` | Recommended | Current assets |
| `liabilities_current` | Recommended | Current liabilities |
| `receivables` | Optional | Trade receivables (for quick ratio) |
| `operating_income` | Optional | EBIT; used for EBITDA calculation when direct EBITDA is absent |
| `depreciation_amortization` | Optional | D&A; used for EBITDA calculation |
| `ebitda` | Optional | Direct EBITDA; overrides the calculated value if provided |
| `interest_expense` | Optional | Interest paid; required for coverage ratios |
| `gross_profit` | Optional | Gross profit; used for gross margin |
| `capex` | Optional | Capital expenditure; used for free cash flow |
| `inventory` | Optional | Inventory; used in future asset-quality extensions |
| `fiscal_year` | Optional | Label for reporting |
| `company_name` | Optional | Label for reporting |

Fields that are absent are treated as NaN and handled by the imputation layer. The `feature_coverage_pct` output reflects how many of the six model features could be computed from the supplied data.

---

## 3. Feature engineering (scoring path)

The function `engineer_private_company_features()` in `features.py` is called identically for training data (Notebook 02) and private-company inputs (Notebook 03). This ensures that a company scored today is evaluated in the same feature space as the companies the model was trained on.

The scoring path applies the following additional options not present in the training path:

**`winsor_caps`**: if provided from the artifact, ratio values are clipped to the training-time winsorisation bounds before the risk transformation. This prevents extreme private-company inputs from extrapolating into undefined regions of the feature space. In the v3 artifact, `winsor_caps = None` (no winsorisation was applied during training), so this step is a pass-through.

**`fx_to_model_currency`**: multiplies all monetary columns by the supplied FX rate before computing ratios. Default is 1.0 (inputs already in USD).

**`min_denominator`**: set to `SME_MIN_DENOMINATOR = 1 000` for private-company scoring, lower than the public-company training default, to avoid suppressing valid but small interest-expense values.

---

## 4. Cluster assignment

Once features are engineered, the artifact's sklearn Pipeline is invoked in two steps:

```
1. pipe.predict(X_new)        → assigned_cluster (integer, 0–4)
2. pipe.transform(X_new)      → distance matrix [n_companies × 5]
```

The Pipeline contains a `SimpleImputer` (median, fit on the training segment) followed by the fitted KMeans model. Both steps are applied in sequence automatically by the Pipeline.

`predict()` returns the cluster with the smallest Euclidean distance to each company's feature vector. `transform()` returns the distances to all five centroids, which are then used to compute soft affinities and outlook diagnostics.

---

## 5. Soft affinity computation

The distance matrix is converted to a probability-like affinity vector using an exponential kernel:

$$a_j = \frac{e^{-d_j / T}}{\sum_{i=1}^{5} e^{-d_i / T}}$$

where $d_j$ is the distance to centroid $j$ and $T$ is the temperature parameter (default 1.0).

**Effect of temperature on affinity distribution:**

| T | Interpretation |
|---|---|
| 0.3 | Very sharp: nearly all affinity concentrated on assigned cluster |
| 1.0 | Moderate: adjacent clusters receive meaningful affinity if close |
| 3.0 | Flat: affinities are approximately equal across all clusters |

The default T = 1.0 is appropriate for most use cases. Lower T is useful when you want a binary-style assignment signal; higher T is useful for visualising how borderline a company's placement is.

---

## 6. Adjacent-bucket outlook diagnostics

The `add_adjacent_bucket_distances_and_outlook()` function in `diagnostics.py` extends the scored output with a directional outlook flag by comparing the company's distances to adjacent cluster centroids.

**Step-by-step:**

1. Look up the assigned cluster's `risk_rank` (1–5).
2. Identify the cluster with `risk_rank − 1` (upper/better bucket) and `risk_rank + 1` (lower/worse bucket) from the artifact's `risk_rank` mapping.
3. Retrieve the distances to those clusters from the full distance matrix.
4. Compare those distances to the assigned-cluster distance using two multipliers:
   - `neutral_band = 0.15`: a 15% buffer within which differences are ignored.
   - `upgrade_boundary_multiplier = 1.35`: an adjacent cluster must be reachable within 1.35× the assigned-cluster distance to trigger a directional signal.
5. Apply the flag logic:

```
diff = lower_distance − upper_distance
threshold = assigned_distance × 0.15

Positive:  diff > threshold AND upper_distance ≤ assigned_distance × 1.35
Negative:  diff < −threshold AND lower_distance ≤ assigned_distance × 1.35
Neutral:   otherwise
```

**What the outlook flag means:**
The flag is a **static cluster-position signal** — it describes where the company sits relative to its neighbours in the current-year feature space. A Negative outlook does not predict that the company will migrate to a worse cluster next year. It means that, given the current financials, the company's risk profile is more similar to the next-worse cluster than to the next-better cluster.

---

## 7. Scenario analysis

`make_scenarios()` in `scoring.py` constructs four pre-defined stress cases from a base set of raw financials:

| Scenario | Changes applied to base inputs |
|---|---|
| `base` | No change — the company as reported |
| `revenue_down_15pct` | Revenue −15%; net income −30%; CFO −25%; operating income −30% |
| `debt_up_25pct` | Long-term debt +25%; liabilities increase accordingly; cash slightly reduced |
| `cash_burn_case` | Cash halved; CFO set to negative; net income set to negative |
| `near_default_stress` | Liabilities = 110% of assets (insolvent balance sheet); LT debt = 75% of assets; ST debt = 15% of assets; operating income = 40% of interest expense |

Each scenario row is scored identically to the base case. The scenario output table shows the cluster label, risk rank, affinity, near-default affinity, outlook flag, and scorecard score for all five scenarios side by side.

**Reading scenario results:**
- If the base case is in tier 3 and the `debt_up_25pct` scenario shifts to tier 4, leverage is the binding constraint for this company's credit quality.
- If the `near_default_stress` scenario remains in tier 4 (not tier 5), it means the company's feature profile in the stress case is still closer to the speculative centroid than to the distressed centroid — a structural resilience signal.
- The `near_default_affinity` is the most sensitive indicator of proximity to the distressed cluster across scenarios.

---

## 8. Company vs cluster median comparison

`compare_to_cluster_profile()` in `diagnostics.py` produces a row-level comparison of a single scored company against the median profile of its assigned cluster:

| Column | Description |
|---|---|
| `metric` | Financial ratio or risk feature name |
| `company_value` | The scored company's value for this metric |
| `assigned_cluster_median` | Median value for this metric across all companies in the assigned cluster |
| `difference` | `company_value − assigned_cluster_median` |
| `relative_position` | `above_cluster_median`, `below_cluster_median`, or `equal_to_cluster_median` |

This table is useful for identifying which specific dimensions drive the company's position within its cluster, and whether the company is a well-centred representative or an outlier at the boundary.

---

## 9. Warning flags in the context of a report

Warning flags generated by `make_warning_flags()` are diagnostic signals, not disqualifiers. In a formal credit report they serve three functions:

1. **Data quality check**: `invalid_assets` or `assets_below_model_threshold` flag potential input errors or companies outside the model's training scope.
2. **Structural stress indicators**: `liabilities_exceed_assets`, `negative_equity`, `interest_coverage_below_1` flag conditions that are independently meaningful regardless of cluster assignment.
3. **Consistency check**: if a company is assigned to tier 1 (Low risk) but carries `interest_coverage_below_1` and `debt_to_ebitda_above_6`, this is a signal that the cluster assignment may reflect an unusually well-performing peer group rather than genuinely low-risk financials — warranting closer inspection of the `cluster_affinity` and `feature_coverage_pct`.

---

*See also: [Model Interpretation](model_interpretation.md) | [User Guide: Configuration](user_guide_config.md) | [Methodology](methodology.md)*
