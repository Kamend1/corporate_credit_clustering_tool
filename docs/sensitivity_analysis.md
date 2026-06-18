# Sensitivity Analysis and Robustness Plan

## Purpose of this document

This document defines the sensitivity and robustness checks for the Corporate Credit Clustering Tool. It is both a future-development roadmap and a final-project methodological reference.

The current project is an unsupervised model, so validation cannot rely on ordinary classification accuracy. Robustness must be assessed through cluster stability, financial interpretability, and sensitivity to design choices.

---

## 1. Why sensitivity analysis matters

The model contains analyst-defined choices:

1. risk transformation thresholds;
2. component risk formulas;
3. domain weights;
4. number of clusters `k`;
5. imputation strategy;
6. feature coverage rules;
7. guardrail thresholds;
8. scoring temperature for affinity.

These choices are defensible, but they are still assumptions. A strong academic and professional project should show how much the results depend on them.

Core question:

```text
Do the five risk tiers remain broadly stable under reasonable changes in assumptions?
```

---

## 2. Baseline model to compare against

Baseline assumptions:

| Parameter | Baseline |
|---|---|
| Segment | Non-financial companies |
| Features | Six scorecard domain risk features |
| Number of clusters | 5 |
| Algorithm | KMeans |
| Initialization | k-means++ |
| `n_init` | 500 |
| Imputation | Median imputation |
| Scorecard weights | Leverage 25%, liquidity 10%, earnings 15%, operating cash flow 20%, debt service 25%, structural distress 5% |

All sensitivity tests should compare results back to this baseline.

---

## 3. Key robustness metrics

### Adjusted Rand Index

Adjusted Rand Index, or ARI, compares two cluster assignment vectors while ignoring arbitrary cluster labels.

Interpretation:

| ARI | Reading |
|---:|---|
| 1.00 | Identical clustering |
| 0.85+ | Very stable |
| 0.70–0.85 | Moderately stable |
| 0.50–0.70 | Sensitive; investigate |
| below 0.50 | Major instability |

ARI is useful because KMeans cluster IDs can change between runs.

### Rank preservation

After perturbation, clusters should still rank in a financially sensible order by median scorecard score.

### Financial monotonicity

Higher-risk clusters should generally show weaker:

- leverage;
- liquidity;
- profitability;
- cash flow;
- interest coverage;
- structural distress indicators.

### Representative-company plausibility

Representative companies near each centroid should make intuitive business sense for the assigned risk tier.

---

## 4. k sensitivity

### Test design

Train KMeans for:

```text
k = 2, 3, 4, 5, 6, 7, 8
```

For each value, record:

- inertia;
- silhouette score;
- Calinski-Harabasz index;
- Davies-Bouldin index;
- cluster sizes;
- cluster profile medians.

### Interpretation

`k = 5` is acceptable when it balances:

- reasonable internal metrics;
- interpretable tier structure;
- usable cluster sizes;
- monotone financial profiles;
- business communication value.

The point is not to blindly maximize one metric. In credit-risk segmentation, interpretability matters.

---

## 5. Random seed and initialization stability

### Test design

Run the baseline KMeans model across multiple random states:

```python
random_states = [1, 7, 21, 42, 100]
```

For each run:

1. fit KMeans with the same `k` and feature set;
2. compute ARI versus the baseline assignment;
3. compare cluster profiles;
4. check whether risk-rank ordering is preserved.

### Expected standard

A stable model should show high ARI and similar cluster financial profiles across random states, especially with `n_init = 500`.

---

## 6. Bootstrap stability

### Test design

For the non-financial training panel:

1. draw bootstrap samples containing 80% of observations;
2. train KMeans with `k = 5`;
3. score the held-out or full common sample;
4. compare assignments to the baseline model using ARI;
5. repeat 30–50 times.

### Target

| Metric | Target |
|---|---:|
| Mean ARI | above 0.80 |
| Minimum ARI | preferably above 0.65 |
| Rank preservation | most bootstrap runs |

Bootstrap instability would indicate that clusters are driven by a narrow subset of companies.

---

## 7. Threshold sensitivity

### Test design

Perturb selected `RISK_THRESHOLDS` by ±10% and ±20%.

High-priority baseline thresholds:

| Threshold | Current baseline |
|---|---:|
| `liabilities_to_assets` | low 0.30 / high 0.95 |
| `debt_to_assets` | low 0.10 / high 0.75 |
| `equity_to_assets` | good 0.65 / bad 0.00 |
| `current_ratio` | good 2.50 / bad 0.50 |
| `quick_ratio` | good 1.50 / bad 0.25 |
| `net_income_to_assets` | good 0.10 / bad -0.05 |
| `cfo_to_assets` | good 0.20 / bad -0.05 |
| `cfo_to_debt` | good 0.60 / bad 0.02 |
| `interest_coverage` | good 12.0 / bad 0.80 |
| `fcf_to_debt` | good 0.40 / bad -0.10 |
| `debt_repayment_capacity` | good 0.40 / bad -0.08 |
| `debt_to_ebitda` | low 1.0 / high 6.0 |
| `net_debt_to_ebitda` | low 1.0 / high 5.0 |
| `ebitda_interest_coverage` | good 20.0 / bad 1.00 |

For each perturbation:

1. re-engineer features;
2. rerun KMeans;
3. rank clusters;
4. calculate ARI versus baseline;
5. inspect financial monotonicity.

### Suggested pass levels

| Perturbation | Desired ARI |
|---|---:|
| ±10% | above 0.85 |
| ±20% | above 0.70 |

If a single threshold causes major instability, the model is overly dependent on that assumption.

Coverage thresholds should be interpreted as strong-credit thresholds, not minimum survival thresholds. For example, positive interest coverage may still receive non-zero risk because the baseline distinguishes excellent debt-service capacity from merely adequate debt-service capacity.

---

## 8. Domain-weight sensitivity

Domain weights affect the `scorecard_credit_score` and post-hoc ranking. Depending on implementation, they may also affect engineered feature construction if domain aggregation is changed.

### Test A: Equal weights

Replace the default domain weights with equal weights:

```text
1/6 for each domain
```

Then check:

- whether cluster ranking changes;
- whether labels remain sensible;
- whether representative companies still look plausible.

### Test B: Conservative lender weights

Increase leverage and debt-service emphasis.

### Test C: SME liquidity/cash-flow weights

Increase liquidity and operating cash-flow emphasis.

### Test D: Random Dirichlet weights

Sample random weight vectors and test how often cluster ranking is preserved.

Suggested target:

```text
risk-rank ordering preserved in more than 90% of reasonable weight vectors
```

---

## 9. Leave-one-domain-out sensitivity

### Test design

Remove one of the six domain features and retrain KMeans:

- without leverage risk;
- without liquidity risk;
- without earnings risk;
- without operating cash-flow risk;
- without debt-service risk;
- without structural distress risk.

For each model:

- calculate ARI versus baseline;
- inspect cluster profiles;
- identify which removed domain causes the most disruption.

### Interpretation

If removing one domain collapses the cluster structure, that domain is the dominant separator. This is not automatically bad, but it should be disclosed.

---

## 10. Imputation sensitivity

Baseline uses median imputation inside the scikit-learn pipeline.

Alternative tests:

| Strategy | Description |
|---|---|
| Complete-case only | Drop rows with any missing model feature. |
| Midpoint imputation | Impute missing bounded features to 0.5. |
| Conservative imputation | Impute missing risk features to worse-than-median values. |

Compare assignments on the common sample using ARI.

Purpose:

```text
Check whether the model is learning credit-risk structure or missing-data structure.
```

---

## 11. Feature coverage sensitivity

Test stricter and looser row inclusion rules.

| Setting | Meaning |
|---|---|
| Minimum 3 of 6 features | More inclusive, more imputation. |
| Minimum 4 of 6 features | Current balanced default. |
| Minimum 5 of 6 features | Cleaner but smaller sample. |
| Minimum 6 of 6 features | Complete model-feature coverage only. |

For each setting:

- observe sample size;
- run KMeans;
- compare ARI;
- inspect whether clusters become cleaner or less representative.

---

## 12. Alternative algorithm comparison

Notebook 02 compares KMeans to:

| Method | Robustness question |
|---|---|
| Agglomerative clustering | Do the risk tiers have hierarchical structure? |
| DBSCAN | Are there natural dense groups or clear outlier/distress regions? |
| PCA visualization | Does the six-dimensional structure show visible separation in 2D/3D? |

Alternative methods do not need to replace KMeans. Their role is to test whether KMeans is producing plausible structure rather than arbitrary segmentation. Notebook 04 is a separate EDGAR data-acquisition appendix, not the alternative-clustering comparison notebook.

---

## 13. Guardrail sensitivity

Guardrails are post-model diagnostics, but they affect interpretation.

Tests:

- tighten leverage guardrails;
- loosen leverage guardrails;
- tighten liquidity guardrails;
- increase interest coverage threshold;
- test how many companies move from Clear/Monitor to Caution/High caution.

Report:

```text
percentage of scored companies affected by each guardrail family
```

This helps distinguish model classification from analyst caution.

---

## 14. Scoring temperature sensitivity

Temperature affects affinities, not cluster assignment.

Test:

```text
T = 0.3, 0.5, 1.0, 2.0, 3.0
```

Record:

- assigned-cluster affinity;
- near-default affinity;
- whether interpretation becomes too sharp or too flat.

Recommended default remains `T = 1.0` unless the report clearly explains a different value.

---

## 15. Minimum recommended tests for the SoftUni final version

For the final course project, the minimum academically useful robustness package is:

| Test | Status target |
|---|---|
| k sweep from 2 to 8 | Implemented in Notebook 02 |
| cluster profile monotonicity | Implemented in Notebook 02 |
| alternative clustering comparison | Implemented in Notebook 02 |
| one random-state stability test | Recommended |
| one imputation or threshold sensitivity test | Recommended |
| scenario sensitivity in Notebook 03 | Implemented / demonstrated |

If time is limited, prioritize one small sensitivity table that Dancho can see in the notebook.

---

## 16. Reporting template for sensitivity results

| Test | Metric | Result | Interpretation |
|---|---|---:|---|
| Baseline k=5 | Reference | 1.00 | Baseline model |
| Random state variation | ARI vs baseline | TBD | Stability of initialization |
| Equal domain weights | Rank preservation | TBD | Weight robustness |
| Complete-case imputation | ARI vs baseline | TBD | Missing-data robustness |
| k=4 vs k=5 | Qualitative | TBD | Whether adjacent tiers merge sensibly |
| DBSCAN comparison | Noise share / NMI | TBD | Outlier structure |

Do not invent results. If a test has not been run, mark it as planned or future work.

---

## 17. Bottom line

A good unsupervised ML project is not validated by accuracy alone. For this project, credibility comes from:

```text
stable clusters + interpretable financial profiles + sensible sensitivity behavior + transparent limitations
```

The sensitivity framework should gradually move from roadmap to executed evidence as the project matures.
