# Sensitivity Analysis

This document defines robustness checks for the Corporate Credit Clustering Tool.

The model is unsupervised and domain-guided. Therefore, validation must test both mathematical cluster behavior and financial interpretation.

---

## 1. Why sensitivity analysis matters

The model contains expert-defined assumptions:

- risk thresholds;
- component weights inside domain features;
- scorecard domain weights;
- imputation strategy;
- number of clusters;
- feature coverage thresholds;
- guardrail thresholds.

Sensitivity analysis checks whether the model remains useful under reasonable changes to those assumptions.

---

## 2. Component saturation analysis

Because risk scores are bounded, inspect how often component scores equal 0 or 1.

```python
saturation = pd.DataFrame({
    "zero_share": (df[risk_cols] == 0).mean(),
    "one_share": (df[risk_cols] == 1).mean(),
    "missing_share": df[risk_cols].isna().mean(),
}).sort_values("one_share", ascending=False)
```

| Result | Meaning |
|---|---|
| high `zero_share` | threshold may be too lenient or metric has limited variation |
| high `one_share` | threshold may be too strict or data is extreme |
| high `missing_share` | feature availability issue |
| balanced distribution | useful gradient for KMeans |

A model dominated by exact 0s and 1s can become too binary for distance-based clustering.

---

## 3. Threshold sensitivity

For each threshold pair, perturb the calibration and rerun the pipeline.

Suggested perturbations:

```text
-20%, -10%, +10%, +20%
```

Record:

- Adjusted Rand Index versus baseline;
- cluster size changes;
- risk-rank ordering preservation;
- financial monotonicity;
- companies moving by more than one risk rank.

Important thresholds to test:

- liabilities/assets;
- debt/assets;
- equity/assets;
- CFO/assets;
- CFO/debt;
- FCF/debt;
- interest coverage;
- debt/EBITDA;
- net debt/EBITDA;
- EBITDA interest coverage.

---

## 4. Domain formula sensitivity

### Leverage risk

Baseline:

```text
0.30 liabilities_risk
0.25 debt_load_risk
0.20 equity_buffer_risk
0.25 net_debt_to_ebitda_risk
```

Test equal weights, balance-sheet-heavy weights, and EBITDA-leverage-heavy weights.

### Liquidity risk

Baseline:

```text
0.35 current_liquidity_risk
0.30 quick_liquidity_risk
0.20 debt_repayment_risk
0.15 cash_buffer_risk
```

Test pure working-capital liquidity, higher FCF/debt emphasis, and equal component weights.

### Operating cash-flow risk

Baseline:

```text
0.50 cashflow_risk
0.50 cfo_to_debt_risk
```

Test 70/30, 30/70, pure CFO/assets, and pure CFO/debt variants.

### Structural distress risk

Baseline:

```text
0.60 equity_buffer_risk
0.40 liabilities_risk
```

Test 50/50, equity-heavy, and liabilities-heavy variants.

---

## 5. Scorecard domain weight sensitivity

Current scorecard weights:

```python
{
    "leverage_risk": 0.25,
    "liquidity_risk": 0.10,
    "earnings_risk": 0.15,
    "operating_cashflow_risk": 0.20,
    "debt_service_risk": 0.25,
    "structural_distress_risk": 0.05,
}
```

Test alternatives:

- equal-weight scorecard;
- creditor-focused scorecard;
- cash-flow-focused scorecard;
- liquidity-focused scorecard.

Metrics:

- cluster rank preservation;
- median scorecard score by cluster;
- change in representative companies;
- change in report conclusions.

Scorecard weights affect ranking and interpretation. They do not directly change KMeans assignments unless explicitly used inside the clustering feature space.

---

## 6. k sensitivity

Test:

```text
k = 2, 3, 4, 5, 6, 7, 8
```

Metrics:

- inertia;
- silhouette coefficient;
- Calinski-Harabasz index;
- Davies-Bouldin index;
- cluster size balance;
- financial monotonicity;
- interpretability.

Financial monotonicity matters more than a single metric.

---

## 7. Bootstrap stability

Recommended approach:

1. sample 80% of observations;
2. fit KMeans with k = 5;
3. compare assignments with the baseline model using ARI;
4. repeat 30–50 times.

Useful outputs:

- mean ARI;
- minimum ARI;
- rank preservation;
- centroid stability.

---

## 8. Imputation sensitivity

Compare:

| Strategy | Description |
|---|---|
| median imputation | baseline |
| complete-case only | drop rows with missing model features |
| midpoint imputation | fill missing bounded risks with 0.5 |
| missingness indicators | add flags for missing components |

Inspect cluster assignments, feature coverage, and financial profile quality.

---

## 9. Missingness sensitivity for debt-capacity fields

Debt-capacity fields are high impact:

| Input | Affected features |
|---|---|
| total debt | debt/assets, CFO/debt, FCF/debt, debt/EBITDA |
| CFO | CFO/assets, CFO/debt, FCF/debt |
| capex | FCF/debt |
| EBITDA | EBITDA margin, debt/EBITDA, net debt/EBITDA, EBITDA coverage |
| interest expense | EBIT coverage, EBITDA coverage |

Low feature coverage should reduce confidence in the output.

---

## 10. Reporting sensitivity results

A compact sensitivity table should include:

| Test | Metric | Result | Interpretation |
|---|---|---|---|
| threshold perturbation | minimum ARI | value | stable/sensitive |
| k sweep | selected k | value | rationale |
| scorecard weights | rank preservation | value | stable/sensitive |
| imputation test | ARI | value | stable/sensitive |
| saturation test | max zero/one share | value | acceptable/problematic |
| bootstrap | mean ARI | value | stable/sensitive |

---

## 11. Summary

A good unsupervised credit model should remain financially interpretable under reasonable changes to assumptions. If clusters collapse, reorder unpredictably, or lose financial monotonicity under small changes, the configuration should be reviewed.
