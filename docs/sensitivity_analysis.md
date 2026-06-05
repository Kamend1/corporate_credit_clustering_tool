# Sensitivity Analysis

This document defines the sensitivity tests that should be run to establish confidence in the model's outputs and to quantify how much the cluster labels depend on specific design choices. These tests are not implemented in the current notebooks but are framed here as a reproducible research agenda.

---

## 1. Why sensitivity analysis matters for this model

The Corporate Credit Clustering Tool contains several analyst-chosen parameters that are not derived from data:

1. **Risk thresholds** — the `good / bad` and `low / high` boundaries in `RISK_THRESHOLDS` that govern the [0, 1] component risk mappings.
2. **Sub-component weights** — the fixed weights within each domain feature (e.g. 0.40 / 0.35 / 0.25 inside `leverage_risk`).
3. **Domain weights** — the top-level weights in `SCORECARD_DOMAIN_WEIGHTS`.
4. **k (number of clusters)** — currently fixed at 5.
5. **Imputation strategy** — median imputation inside the KMeans pipeline.

Each of these choices affects the cluster centroids and therefore the label assigned to every company. The core scientific question is: **are the five risk tiers robust to reasonable variation in these choices, or do they collapse, merge, or reorder under perturbation?**

---

## 2. Threshold sensitivity

### 2.1 Test design

For each risk threshold pair (e.g. `interest_coverage: good=3.0, bad=1.0`), perturb both boundaries by ±10% and ±20% of their current values and re-run the full pipeline. Record:

- The Adjusted Rand Index (ARI) between the baseline cluster assignment and the perturbed assignment for each company-year row.
- Whether the risk rank ordering of clusters (by median scorecard score) is preserved.

```python
from sklearn.metrics import adjusted_rand_score

# Example: perturb interest_coverage thresholds
perturbations = [-0.20, -0.10, +0.10, +0.20]

for delta in perturbations:
    modified_thresholds = RISK_THRESHOLDS.copy()
    modified_thresholds["interest_coverage"]["good"] *= (1 + delta)
    modified_thresholds["interest_coverage"]["bad"] *= (1 + delta)
    # Re-engineer features with modified thresholds, re-cluster, compute ARI
```

### 2.2 Interpretation

- ARI > 0.85 across all perturbations: labels are robust to threshold variation in that metric.
- ARI < 0.70: the cluster assignment is sensitive to that threshold; either the threshold is near a decision boundary for many companies or the clustering relies disproportionately on that component.

The thresholds most likely to show high sensitivity are `interest_coverage` and `debt_to_ebitda` because they have the largest sub-component weights within `debt_service_risk`.

---

## 3. Domain weight sensitivity

### 3.1 Uniform weight baseline

Replace `SCORECARD_DOMAIN_WEIGHTS` with equal weights (1/6 each) and re-run clustering. Compare:

- Cluster composition (which companies move between tiers).
- ARI against the baseline assignment.
- Whether the cluster profile medians remain monotonically ordered.

### 3.2 Leave-one-domain-out

Drop one domain feature entirely from the clustering (not just zero-weight it — remove it from `SCORECARD_CLUSTER_FEATURES`) and re-cluster with k=5. Record ARI and whether the cluster ordering is preserved. This tests which domain is most responsible for cluster separation.

### 3.3 Grid search over domain weights

Sample 1 000 random weight vectors from a Dirichlet distribution over the six domains. For each:
1. Compute the scorecard score with those weights (does not require re-clustering — the cluster model is fixed).
2. Check whether the cluster rank ordering (by median scorecard under new weights) is preserved.

A stable model should maintain the same rank ordering for >90% of reasonable weight vectors.

---

## 4. k sensitivity

The current k=5 was selected by inspecting silhouette, Calinski-Harabász, and Davies-Bouldin indices across k ∈ {2, …, 8}. The following additional tests strengthen this choice:

### 4.1 ARI stability across k

For k ∈ {4, 5, 6}:
- Train three separate models.
- For k=4 vs k=5: check whether the k=4 clusters are approximately unions of adjacent k=5 clusters (hierarchical consistency). If cluster 3 and cluster 4 in k=5 merge cleanly into one cluster in k=4, the five-tier structure is well-founded.

### 4.2 Cluster stability under bootstrap

For k=5, train on 80% bootstrap samples of the Non-financial panel (50 iterations). For each bootstrap model:
- Score the held-out 20% using both the bootstrap model and the full-data model.
- Compute ARI between the two assignments.

Average ARI > 0.80 across bootstrap iterations indicates that the cluster centroids are stable and not driven by a small subset of companies.

---

## 5. Imputation sensitivity

The KMeans pipeline uses median imputation. Two alternative strategies to benchmark:

| Strategy | Implementation |
|---|---|
| Complete-case analysis | Drop any row with a missing model feature; cluster only fully-observed rows |
| Zero imputation | Replace missing values with 0.5 (the midpoint of the [0, 1] risk space) |

For each strategy, compute ARI against the baseline (median imputation) on the rows that appear in all three versions. If ARI > 0.85 in both cases, imputation strategy is not a material driver of the labels.

---

## 6. Summary table: tests and targets

| Test | Metric | Pass threshold |
|---|---|---|
| Threshold ±10% perturbation (any single threshold) | ARI vs baseline | > 0.85 |
| Threshold ±20% perturbation (any single threshold) | ARI vs baseline | > 0.70 |
| Equal domain weights | ARI vs baseline | > 0.75 |
| Leave-one-domain-out (any domain) | ARI vs baseline | > 0.65 |
| Random weight vectors (Dirichlet) | % preserving rank order | > 90% |
| k=4 vs k=5 cluster mergeability | Qualitative review | Adjacent clusters merge cleanly |
| Bootstrap label stability (k=5) | Mean ARI on held-out | > 0.80 |
| Complete-case imputation | ARI vs baseline | > 0.85 |
| Zero imputation | ARI vs baseline | > 0.80 |

---

## 7. Reporting sensitivity results

Sensitivity results should be reported as a table alongside the cluster profile in any formal presentation of the model. A minimal report contains:

- The baseline ARI (always 1.0 — reference point).
- The minimum ARI observed across all single-threshold perturbation tests.
- The ARI for equal domain weights.
- The bootstrap mean ARI.

If any result falls below the pass threshold, document which design choice is responsible and either justify the choice with additional evidence or adjust the calibration.

---

*See also: [Methodology](methodology.md) | [Limitations](limitations.md)*
