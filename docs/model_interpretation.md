# Model Interpretation

This document explains how to read and interpret every output column produced by the scoring pipeline. It covers cluster labels, distance metrics, soft affinities, outlook flags, warning flags, and the scorecard credit score.

---

## 1. The five risk tiers

After clustering, each company-year observation is assigned to one of five ordered buckets. The mapping from KMeans cluster ID to label is established by ranking clusters within the Non-financial segment from lowest to highest `median_scorecard_credit_score`:

| Risk rank | Default label | Analogy |
|---|---|---|
| 1 | 1 — Low risk / investment-grade-like | BBB+ and above |
| 2 | 2 — Moderate risk / lower-investment-grade-like | BBB− to BB+ |
| 3 | 3 — Elevated risk / leveraged | BB to B+ |
| 4 | 4 — High risk / speculative | B to CCC |
| 5 | 5 — Distressed / near-default proxy | CCC and below |

> **Important:** The analogies are illustrative. The model has not been calibrated against agency rating transitions. See [Limitations](limitations.md).

The cluster IDs (integers 0–4) are assigned by KMeans and are **not** inherently ordered. The risk rank is the post-hoc ordering derived from cluster profiles.

---

## 2. Output columns reference

### 2.1 Core assignment

| Column | Type | Description |
|---|---|---|
| `assigned_cluster` | int | Raw KMeans cluster ID (0-indexed, unordered) |
| `cluster_label` | str | Human-readable risk tier label |
| `risk_rank` | int | Ordered risk rank 1 (lowest) → 5 (highest) |

### 2.2 Distance and affinity

| Column | Type | Description |
|---|---|---|
| `distance_to_assigned_cluster` | float | Euclidean distance from the company's feature vector to its assigned cluster centroid. Lower = more representative of its tier. |
| `cluster_affinity` | float | Soft affinity for the assigned cluster (exponential kernel, temperature T=1.0). Range [0, 1]; higher = more confident assignment. |
| `near_default_affinity` | float | Soft affinity specifically for the most distressed cluster (risk rank 5). A value approaching 1.0 signals near-default proximity even if the company is assigned a higher tier. |
| `cluster_0_affinity … cluster_4_affinity` | float | Full affinity vector across all five clusters. |
| `cluster_0_distance … cluster_4_distance` | float | Full distance vector across all five clusters. |

**How soft affinity is computed:**

$$a_j = \frac{\exp(-d_j / T)}{\sum_{i=1}^{5} \exp(-d_i / T)}$$

where $d_j$ is the Euclidean distance to centroid $j$ and $T$ is the temperature parameter (default 1.0). A lower temperature sharpens the affinity distribution; a higher temperature spreads it.

**Reading affinity scores:**

- `cluster_affinity > 0.70`: strong, well-centred assignment; the company is a clear representative of its tier.
- `cluster_affinity 0.40–0.70`: moderate confidence; the company sits between tiers.
- `cluster_affinity < 0.40`: low confidence; the company is approximately equidistant from multiple centroids and its label should be treated with caution.

### 2.3 Outlook flags (adjacent-bucket diagnostics)

The outlook analysis in `diagnostics.py` computes the company's distance to the immediately stronger (lower risk rank) and immediately weaker (higher risk rank) adjacent clusters and issues a directional flag.

| Column | Description |
|---|---|
| `upper_bucket_cluster` | Cluster ID of the next-lower-risk adjacent tier |
| `upper_bucket_label` | Label of that tier |
| `distance_to_upper_bucket` | Distance to the upper (better) tier centroid |
| `lower_bucket_cluster` | Cluster ID of the next-higher-risk adjacent tier |
| `lower_bucket_label` | Label of that tier |
| `distance_to_lower_bucket` | Distance to the lower (worse) tier centroid |
| `upper_distance_ratio_to_assigned` | `distance_to_upper_bucket / distance_to_assigned_cluster` |
| `lower_distance_ratio_to_assigned` | `distance_to_lower_bucket / distance_to_assigned_cluster` |
| `outlook_flag` | **Positive**, **Neutral**, or **Negative** |
| `outlook_reason` | Plain-language explanation of the flag |

**Outlook flag logic (simplified):**

```
threshold_band = assigned_distance × 0.15        # 15% neutral band
upgrade_boundary = assigned_distance × 1.35
downgrade_boundary = assigned_distance × 1.35

if (lower_distance − upper_distance > threshold_band) AND (upper_distance ≤ upgrade_boundary):
    → Positive   # Closer to a better tier and meaningfully so

elif (upper_distance − lower_distance > threshold_band) AND (lower_distance ≤ downgrade_boundary):
    → Negative   # Closer to a worse tier and meaningfully so

else:
    → Neutral    # Firmly in the assigned tier
```

> **Important:** The outlook flag is a **static, cross-sectional cluster-position signal**, not a forward-looking forecast. It reflects how the company's current financial profile relates to adjacent cluster centroids. It is not a prediction that the company will migrate to a different tier.

### 2.4 Feature quality

| Column | Description |
|---|---|
| `feature_coverage_pct` | Share of the six model features that were non-null before imputation. Range [0, 1]. Values below 0.67 indicate imputation is substituting for more than one feature. |

### 2.5 Warning flags

The `warning_flags` column contains a comma-separated list of structural anomalies detected in the raw or derived financials. A value of `"none"` means no flags were raised.

| Flag | Condition |
|---|---|
| `invalid_assets` | `assets ≤ 0` |
| `assets_below_model_threshold` | `assets < $1 000 000` (below training minimum) |
| `liabilities_exceed_assets` | `liabilities_to_assets > 1` |
| `negative_equity` | `equity_to_assets < 0` |
| `high_debt_to_assets` | `debt_to_assets > 0.75` |
| `current_ratio_below_1` | `current_ratio < 1` |
| `quick_ratio_below_0_5` | `quick_ratio < 0.5` |
| `interest_coverage_below_1` | `interest_coverage < 1` |
| `ebitda_interest_coverage_below_1_5` | `ebitda_interest_coverage < 1.5` |
| `debt_to_ebitda_above_6` | `debt_to_ebitda > 6` |
| `net_debt_to_ebitda_above_5` | `net_debt_to_ebitda > 5` |
| `negative_or_zero_ebitda` | `ebitda ≤ 0` |
| `negative_cfo_to_assets` | `cfo_to_assets < 0` |

Multiple flags can appear together. They are diagnostic tools, not disqualifiers. A company can legitimately carry several flags and still be a well-understood credit.

### 2.6 Scorecard credit score

`scorecard_credit_score` is a continuous index on [0, 100], calculated as a weighted sum of the six domain risk features with domain weights renormalised to available features:

$$\text{score} = 100 \times \frac{\sum_{d} w_d \cdot r_d}{\sum_{d \in \text{available}} w_d}$$

A score of **0** means all available domain features indicate zero risk. A score of **100** means maximum risk across all available dimensions.

This score is **not** a KMeans model input. It is computed after clustering and used to:
1. Rank the five clusters post-hoc (clusters are sorted by their segment median scorecard score to assign risk ranks 1–5);
2. Provide a continuous complement to the discrete cluster label for comparison and reporting.

---

## 3. Reading a scored company: worked example

```
company_name              : ExampleCorp Inc.
assigned_cluster          : 2
cluster_label             : 3 — Elevated risk / leveraged
risk_rank                 : 3
cluster_affinity          : 0.58
near_default_affinity     : 0.07
distance_to_assigned_cluster : 0.31
outlook_flag              : Negative
outlook_reason            : Company is closer to the weaker adjacent bucket...
scorecard_credit_score    : 54.2
feature_coverage_pct      : 1.00
warning_flags             : debt_to_ebitda_above_6, interest_coverage_below_1
```

**Reading:**
- Assigned to tier 3 (Elevated risk) with moderate confidence (affinity 0.58). Not a clean centre-of-cluster assignment.
- Near-default affinity of 0.07 is low — the company is not imminently proximate to the distressed cluster.
- **Negative outlook** signals the company's feature profile is meaningfully closer to the tier 4 centroid than to tier 2. Under the current financials, the balance of risk is tilted toward a downgrade-equivalent position.
- Two structural warning flags: interest cover is below 1× and debt/EBITDA exceeds 6×, both of which drove the debt_service_risk domain to a high value.
- Full feature coverage — all six domain features were computed without imputation.

---

## 4. Cluster profile interpretation

The cluster profile table (output of `build_cluster_profile()`) reports **median** values of all diagnostic ratios for each segment-cluster combination. When interpreting:

- The five clusters should show **monotone progressions** for leverage, coverage, and profitability metrics. If cluster 1 (lowest risk) does not show materially lower `debt_to_ebitda` than cluster 5, this is a sign that k selection or threshold calibration may need adjustment.
- Industry mix (`build_industry_cluster_mix()`) helps confirm that the distressed cluster is not dominated by a single sector, which would indicate a sector-specific feature mis-calibration rather than a cross-sectional risk signal.
- Representative tickers (`representatives()`) are selected by minimum Euclidean distance to the cluster centroid in feature space — they are the companies whose risk profiles are most typical of the cluster, not necessarily the most prominent names.

---

*See also: [Methodology](methodology.md) | [Scorer Report Methodology](scorer_report_methodology.md)*
