# Model Interpretation Guide

## Purpose of this document

This document explains how to read the outputs produced by the Corporate Credit Clustering Tool. It covers cluster labels, risk ranks, distance metrics, soft affinities, outlook flags, scorecard scores, warning flags, guardrails, and common interpretation mistakes.

The model output should be read as a **relative financial-risk diagnostic**, not as a formal credit rating.

---

## 1. The five risk tiers

The model assigns each scored company-year to a KMeans cluster. Raw cluster IDs are arbitrary, so the project ranks clusters after training by median `scorecard_credit_score`.

| Risk rank | Recommended label | Interpretation |
|---:|---|---|
| 1 | Strong relative credit profile | Stronger financial-risk profile than most companies in the benchmark universe. |
| 2 | Good credit profile | Generally sound profile, but not the strongest risk bucket. |
| 3 | Loss-making / cash-flow weak profile | Material earnings, operating cash-flow, or debt-service weakness, often with relatively low leverage or some liquidity support. |
| 4 | Leveraged / weak operating credit profile | Leveraged profile with weaker operating performance, liquidity pressure, and/or elevated debt-service risk. |
| 5 | Distressed / near-default proxy | Severe financial weakness or distress-like balance-sheet / debt-service profile. |

These are model-relative labels. They are not external credit ratings and are not calibrated to agency notches.

---

## 2. Raw cluster ID vs risk rank

| Field | Meaning | Business interpretation |
|---|---|---|
| `assigned_cluster` | Raw KMeans cluster ID, usually 0–4 | Technical ID only; unordered. |
| `risk_rank` | Ordered rank from 1 to 5 | Main ordered risk scale. |
| `cluster_label` | Human-readable label tied to risk rank | Main report-facing label. |

Critical rule:

```text
Never interpret cluster 0, 1, 2, 3, or 4 directly.
Always interpret risk_rank and cluster_label.
```

KMeans can assign different numeric cluster IDs after retraining even if the economic structure is similar.

---

## 3. Core output columns

| Column | Meaning |
|---|---|
| `company_name` | Name used for report display. |
| `fiscal_year` | Fiscal year or scenario year. |
| `assigned_cluster` | Raw KMeans technical cluster ID. |
| `cluster_label` | Business label mapped from risk rank. |
| `risk_rank` | Ordered risk tier, 1 = strongest, 5 = weakest. |
| `scorecard_credit_score` | Continuous risk index from 0 to 100. Higher means weaker. |
| `feature_coverage_pct` | Share of model features available before imputation. |
| `warning_flags` | Mechanical data or financial red flags. |
| `guardrail_level` | Highest post-model analyst caution level. |
| `guardrail_summary` | Short professional explanation of guardrail issues. |

---

## 4. Scorecard credit score

The scorecard credit score is a weighted average of six domain-level risks, scaled to `[0, 100]`:

```text
score = 100 × Σ(w_d × r_d) / Σ(w_d for available domains)
```

where:

| Symbol | Meaning |
|---|---|
| `r_d` | domain risk value |
| `w_d` | domain weight |

Interpretation:

| Score range | Broad interpretation |
|---:|---|
| 0–20 | Strong model-relative profile |
| 20–40 | Generally sound profile |
| 40–60 | Elevated / mixed risk profile |
| 60–80 | Weak profile |
| 80–100 | Distress-like profile |

These ranges are practical reading aids, not hard rating thresholds.

The score is not itself the KMeans model. It is used to rank clusters and support interpretation.

The current threshold calibration is conservative in debt-service interpretation. Strong interest coverage and strong EBITDA-interest coverage are treated as low-risk signals; merely positive coverage can still produce moderate risk if it falls materially below the high-quality coverage thresholds.

---

## 5. Distance metrics

KMeans assigns a company to the centroid with the smallest Euclidean distance in the six-dimensional risk-feature space.

| Column | Meaning |
|---|---|
| `distance_to_assigned_cluster` | Distance from company vector to assigned cluster centroid. |
| `cluster_0_distance` ... `cluster_4_distance` | Distance to each centroid. |

Lower distance means the company is more representative of that cluster.

Higher distance means the company is less typical and may sit near a boundary or be an outlier.

Because model features are bounded and directionally aligned, distances are more interpretable than they would be on raw financial ratios. Still, distance is relative to the trained model, not an absolute credit-risk measure.

---

## 6. Soft cluster affinity

The model converts distances into soft affinities using an exponential distance kernel:

```text
a_j = exp(-d_j / T) / Σ_i exp(-d_i / T)
```

where:

| Symbol | Meaning |
|---|---|
| `d_j` | distance to centroid `j` |
| `T` | temperature parameter |
| `a_j` | affinity to cluster `j` |

| Column | Meaning |
|---|---|
| `cluster_affinity` | Affinity to the assigned cluster. |
| `near_default_affinity` | Affinity to the weakest/distressed proxy cluster. |
| `cluster_0_affinity` ... `cluster_4_affinity` | Full affinity vector. |

### Reading affinity

| Affinity | Interpretation |
|---:|---|
| Above 0.70 | Clear, well-centered assignment. |
| 0.40–0.70 | Moderate assignment; company may be between tiers. |
| Below 0.40 | Borderline or weak assignment; interpret carefully. |

Important:

```text
Affinity is not probability of default.
Near-default affinity is not probability of default.
```

It is a normalized distance-based similarity measure.

---

## 7. Near-default affinity

`near_default_affinity` measures how similar the company is to the weakest model cluster. It is useful because a company can be assigned to a non-distressed cluster while still showing some proximity to the distressed centroid.

Interpretation:

| Near-default affinity | Reading |
|---:|---|
| Low | Company is far from the distressed proxy centroid. |
| Moderate | Some distress-like features are present. |
| High | Company resembles the weakest cluster and requires careful review. |

This is not default probability. It is a proximity signal.

---

## 8. Adjacent-bucket outlook

The outlook diagnostic compares the company’s distance to the immediately stronger and weaker adjacent buckets.

| Column | Meaning |
|---|---|
| `upper_bucket_cluster` | Raw cluster ID of the next stronger bucket. |
| `upper_bucket_label` | Label of the next stronger bucket. |
| `distance_to_upper_bucket` | Distance to stronger adjacent centroid. |
| `lower_bucket_cluster` | Raw cluster ID of the next weaker bucket. |
| `lower_bucket_label` | Label of the next weaker bucket. |
| `distance_to_lower_bucket` | Distance to weaker adjacent centroid. |
| `outlook_flag` | Positive, Neutral, or Negative. |
| `outlook_reason` | Plain-language explanation. |

Interpretation:

| Outlook | Meaning |
|---|---|
| Positive | Current profile is closer to the stronger adjacent bucket than to the weaker one. |
| Neutral | Current profile is centered enough in the assigned bucket. |
| Negative | Current profile is closer to the weaker adjacent bucket than to the stronger one. |

Important:

```text
The outlook flag is not a forecast.
```

It does not predict future migration. It only describes the current position relative to adjacent centroids.

---

## 9. Feature coverage

`feature_coverage_pct` reports how many of the six model features were available before imputation.

| Coverage | Interpretation |
|---:|---|
| 1.00 | Full model-feature availability. |
| 0.80–0.99 | Strong coverage. |
| 0.67–0.79 | Acceptable but should be noted. |
| Below 0.67 | Weak basis for scoring; manual review required. |

Low coverage means the model may rely heavily on imputed median values. This can pull the company toward average-risk clusters.

---

## 10. Warning flags

Warning flags are mechanical signals generated from raw and derived financials.

Examples:

| Flag | Meaning |
|---|---|
| `invalid_assets` | Assets are zero, negative, or unusable. |
| `assets_below_model_threshold` | Company is below the public-company training size threshold. |
| `liabilities_exceed_assets` | Balance-sheet liabilities exceed assets. |
| `negative_equity` | Equity is negative. |
| `high_debt_to_assets` | Debt/assets is high. |
| `current_ratio_below_1` | Current liabilities exceed current assets. |
| `quick_ratio_below_0_5` | Liquid assets are low relative to current liabilities. |
| `interest_coverage_below_1` | EBIT does not cover interest expense. |
| `negative_or_zero_ebitda` | EBITDA is zero or negative. |
| `negative_cfo_to_assets` | Operating cash flow is negative relative to assets. |

Warning flags are not automatic disqualifiers. They are prompts for analyst review.

---

## 11. Guardrails

Guardrails are a professional interpretation layer added after model scoring. They help prevent over-optimistic conclusions when raw financial red flags exist.

| Guardrail level | Meaning |
|---|---|
| Clear | No material contradiction detected. |
| Monitor | Minor weakness; explain but do not overreact. |
| Caution | Meaningful caveat; qualify the conclusion. |
| High caution | Material weakness; avoid clean low-risk framing. |
| Override required | Severe red flag; manual analyst review required. |

A company can have a strong model-relative label and still trigger caution. In that case, the report should explain the tension instead of hiding it.

---

## 12. Cluster profile interpretation

Cluster profile tables summarize median values by cluster. A credible credit-risk clustering should show reasonable progression across risk ranks.

Expected patterns:

| Metric family | Expected movement from rank 1 to rank 5 |
|---|---|
| Leverage | Worsens |
| Liquidity | Worsens |
| Coverage | Worsens |
| Profitability | Worsens |
| Operating cash flow | Worsens |
| Structural distress | Increases |

The progression does not need to be perfect for every ratio, because real companies are mixed. But the overall profile should make financial sense.

---

## 13. Worked example

Example output:

```text
company_name: ExampleCorp
risk_rank: 3
cluster_label: 4 - Leveraged / weak operating credit profile
cluster_affinity: 0.42
near_default_affinity: 0.20
scorecard_credit_score: 59.8
outlook_flag: Neutral
feature_coverage_pct: 1.00
warning_flags: interest_coverage_below_1
guardrail_level: High caution
```

Interpretation:

The company is assigned to the leveraged / weak operating bucket. The near-default affinity is a proximity signal, not a default probability. The neutral outlook means the company is not clearly pulled toward either adjacent bucket, but the guardrail flags show that weak interest coverage and leverage-related diagnostics should qualify the analyst narrative.

---

## 14. Common incorrect interpretations

Do not say:

```text
The model rated the company BB.
```

Say:

```text
The company is assigned to a model-relative risk bucket.
```

Do not say:

```text
Near-default affinity of 20% means 20% probability of default.
```

Say:

```text
Near-default affinity indicates relative proximity to the weakest cluster centroid.
```

Do not say:

```text
Negative outlook means the company will deteriorate.
```

Say:

```text
Negative outlook means the current financial profile is closer to the weaker adjacent bucket.
```

Do not say:

```text
Cluster 4 is always the worst cluster.
```

Say:

```text
Raw cluster IDs are arbitrary; the ordered risk rank is the meaningful scale.
```

---

## 15. Bottom line

The model output is best read as:

```text
structured, model-relative financial-risk benchmarking supported by diagnostics and guardrails
```

not as:

```text
external credit rating or default prediction
```
