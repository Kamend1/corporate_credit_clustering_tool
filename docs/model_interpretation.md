# Model Interpretation

This document explains how to read the outputs of the Corporate Credit Clustering Tool.

The model output is a model-relative credit-risk signal. It is not an agency rating, default probability, lending approval, investment recommendation, or regulated credit opinion.

---

## 1. Main interpretation rule

Use the mapped outputs:

- `risk_rank`;
- `cluster_label`;
- `scorecard_credit_score`;
- distance and affinity diagnostics;
- warning flags;
- guardrails.

Do not interpret raw KMeans cluster IDs directly.

---

## 2. Risk label scale

| Risk rank | Label | Interpretation |
|---:|---|---|
| 1 | Strong relative credit profile | stronger profile relative to the model universe |
| 2 | Good credit profile | sound profile with manageable weaknesses |
| 3 | Leveraged / elevated risk profile | meaningful leverage, liquidity, earnings, or cash-flow risk |
| 4 | Weak credit profile | weak profile requiring close review |
| 5 | Distressed / near-default proxy | severe weakness or distress-like model profile |

---

## 3. Core output columns

| Column | Meaning |
|---|---|
| `assigned_cluster` | raw KMeans cluster ID |
| `cluster_label` | human-readable risk label |
| `risk_rank` | ordered rank from 1 strongest to 5 weakest |
| `scorecard_credit_score` | model-relative score from 0 to 100 |
| `feature_coverage_pct` | share of model features available before imputation |
| `warning_flags` | mechanical warning flags |
| `guardrail_level` | analyst caution level from guardrails |

---

## 4. Scorecard credit score

The scorecard score is a weighted average of six domain risks.

| Domain | Weight |
|---|---:|
| leverage risk | 25% |
| debt-service risk | 25% |
| operating cash-flow risk | 20% |
| earnings risk | 15% |
| liquidity risk | 10% |
| structural balance-sheet vulnerability | 5% |

A higher score indicates weaker model-relative credit profile. It is not probability of default.

---

## 5. Interpreting domain risk features

### `leverage_risk`

High values may reflect high liabilities/assets, high debt/assets, weak equity buffer, or high net debt/EBITDA.

### `liquidity_risk`

High values may reflect weak current ratio, weak quick ratio, weak cash buffer, or weak FCF/debt repayment capacity.

Liquidity risk is not only working-capital liquidity. It also captures internal debt repayment capacity.

### `earnings_risk`

High values reflect weak or negative net income relative to assets.

### `operating_cashflow_risk`

High values may reflect weak CFO/assets, weak CFO/debt, or both.

### `debt_service_risk`

High values may reflect weak interest coverage, weak FCF/debt, high debt/EBITDA, or weak EBITDA interest coverage.

### `structural_distress_risk`

High values reflect structural balance-sheet vulnerability. This does not automatically mean legal insolvency.

Hard structural flags remain separate:

- `negative_equity_flag`;
- `liabilities_exceed_assets_flag`.

---

## 6. Distance and affinity

`distance_to_assigned_cluster` measures distance from the company to its assigned centroid.

`cluster_affinity` converts distances into a soft similarity score. Higher affinity means the company is more clearly located near the assigned centroid.

`near_default_affinity` is similarity to the weakest cluster. It is a proximity signal, not probability of default.

---

## 7. Outlook flag

The outlook flag compares distance to adjacent stronger and weaker buckets.

| Flag | Meaning |
|---|---|
| Positive | closer to stronger adjacent bucket |
| Neutral | reasonably centered in assigned bucket |
| Negative | closer to weaker adjacent bucket |

The outlook flag is a current-position diagnostic, not a forecast.

---

## 8. Warning flags and guardrails

Warning flags identify mechanical issues. Guardrails translate selected issues into analyst caution levels.

| Level | Interpretation |
|---|---|
| Clear | no material issue |
| Monitor | mild issue |
| Caution | meaningful issue |
| High caution | serious weakness |
| Override required | manual analyst review required |

A strong cluster label with severe guardrails should be interpreted cautiously.

---

## 9. Incorrect interpretations to avoid

| Incorrect | Correct |
|---|---|
| Cluster ID is a rating | Raw cluster ID is arbitrary |
| Affinity is default probability | Affinity is distance-based similarity |
| Near-default affinity is PD | It is proximity to weakest cluster |
| Outlook is a forecast | Outlook is current-position geometry |
| Structural distress score means insolvency | It is gradient balance-sheet vulnerability |
| Guardrail is automatic default | It is an analyst review signal |

---

## 10. Summary

Interpret the result by combining label, score, feature coverage, domain drivers, distances, affinities, warning flags, and guardrails. The model is a structured credit diagnostic, not a replacement for professional judgment.
