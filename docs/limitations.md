# Limitations

This document explains the main limitations of the Corporate Credit Clustering Tool.

The model is useful for education, analysis, and screening. It is not a formal credit rating model.

---

## 1. Not a formal credit rating

The model does not produce agency ratings, probability of default, expected loss, lending decisions, investment recommendations, or regulated credit opinions.

The output should support professional judgment, not replace it.

---

## 2. Domain-guided unsupervised learning

The model is unsupervised because it does not train on external rating or default labels.

However, the feature space is shaped by credit-domain assumptions:

- selected ratios;
- risk thresholds;
- component formulas;
- scorecard weights;
- guardrail rules.

This improves interpretability but means the model reflects expert design choices.

---

## 3. Threshold assumptions

The model converts raw ratios into bounded risk scores using predefined thresholds.

Different thresholds can change component scores, domain scores, scorecard values, cluster rankings, and potentially cluster assignments after retraining.

Mitigation:

- inspect score saturation;
- run threshold sensitivity tests;
- review cluster profiles;
- document calibration assumptions.

---

## 4. Formula-weight assumptions

The six domain features are weighted composites.

Examples:

- leverage includes net debt/EBITDA;
- liquidity includes FCF/debt repayment capacity;
- operating cash flow includes CFO/assets and CFO/debt;
- structural distress is a gradient balance-sheet vulnerability score.

These formulas are transparent but judgment-based.

---

## 5. Public-company benchmark bias

The training data comes from SEC EDGAR public-company filings.

Private companies, SMEs, non-US companies, and IFRS-reporting companies may differ materially from the public-company benchmark universe.

Private-company scoring should therefore be interpreted as benchmark-relative.

---

## 6. Survivorship and distress bias

The EDGAR training universe may underrepresent companies that failed, delisted, merged, liquidated, or stopped filing.

The weakest cluster is a distressed or near-default proxy group, not a complete default sample.

---

## 7. Point-in-time nature

The model scores a company-year observation. It is not a through-the-cycle rating and does not automatically forecast future migration.

The outlook flag is a current-position diagnostic, not a prediction.

---

## 8. Missing data and imputation

Missing values can pull low-information companies toward the center of the distribution.

High-risk companies with incomplete debt, EBITDA, CFO, capex, or interest-expense data may look less extreme than they are.

Mitigation:

- review `feature_coverage_pct`;
- inspect missing fields;
- improve manual input quality;
- treat low-coverage results cautiously.

---

## 9. Debt-capacity input sensitivity

Several features rely on correct mapping of total debt, CFO, capex, EBITDA, cash, and interest expense.

| Input issue | Potential effect |
|---|---|
| debt omitted | leverage and debt-service risk understated |
| CFO overstated | cash-flow risk understated |
| capex omitted | FCF/debt overstated |
| EBITDA unavailable | EBITDA-based diagnostics weakened |
| interest expense missing | coverage unavailable |

---

## 10. Financial companies excluded

The model targets non-financial companies. Banks, insurers, REITs, and financial institutions need a different feature framework.

---

## 11. Sector effects

The model is broad and does not fully capture sector-specific differences.

Examples:

- utilities may tolerate higher leverage;
- software companies may have low tangible assets;
- retailers may have structurally lower margins;
- manufacturers may have capital-intensive asset bases.

Peer review and sector interpretation remain necessary.

---

## 12. Structural distress score is not insolvency

`structural_distress_risk` is a gradient balance-sheet vulnerability measure. It is not a legal insolvency conclusion.

Hard flags such as negative equity and liabilities exceeding assets should be reviewed separately.

---

## 13. Guardrails are diagnostics

Guardrails qualify model output. They are not automatic credit ratings or default labels.

Analyst review remains necessary.

---

## 14. No external ground-truth validation

The model has not been directly calibrated against agency ratings, bond spreads, CDS spreads, rating transitions, or observed defaults.

Current validation relies on internal metrics, financial monotonicity, representative-company review, and sensitivity testing.

---

## 15. Summary

The model is intentionally transparent. Its main limitation is that transparent assumptions must be documented, tested, and interpreted with financial judgment.
