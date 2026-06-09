# Limitations

## Purpose of this document

This document defines the known limitations of the Corporate Credit Clustering Tool and explains how they should be mitigated or disclosed. It is intended for future development, professional project use, and academic transparency.

The model should be presented as a **domain-guided unsupervised credit-risk benchmarking tool**, not as a formal credit rating system.

---

## 1. Not a formal credit rating

The model produces relative credit-risk labels derived from KMeans cluster profiles. These labels are not certified credit opinions.

They have not been calibrated against:

- official agency rating histories;
- rating transitions;
- observed defaults;
- bankruptcy filings;
- CDS spreads;
- bond yields.

Therefore, the output must not be used as:

```text
formal credit rating
lending approval decision
investment recommendation
probability of default estimate
regulatory credit assessment
```

Correct use:

```text
screening, benchmarking, exploratory analysis, model demonstration, analyst-support diagnostic
```

---

## 2. Domain-guided, not purely unsupervised

The project uses unsupervised clustering, but the feature space is heavily shaped by credit-domain knowledge.

Domain knowledge enters through:

- financial ratio selection;
- risk transformation thresholds;
- component risk formulas;
- domain-level aggregation;
- scorecard weights used for post-hoc cluster ranking;
- guardrail rules.

This is not a weakness if disclosed correctly. It makes the model interpretable. But the correct description is:

```text
domain-guided unsupervised clustering
```

not:

```text
pure unsupervised discovery of credit risk from raw data
```

A different valid set of thresholds and weights could produce different clusters.

The current thresholds are professionally selected and financially interpretable, but they remain judgment-based. Conservative coverage thresholds, such as high “good” breakpoints for interest coverage and EBITDA interest coverage, make the scorecard stricter in distinguishing excellent debt-service capacity from merely adequate coverage.

---

## 3. Survivorship bias

The training universe contains companies available in the SEC EDGAR/XBRL data pull. Companies that defaulted, delisted, were acquired, dissolved, or stopped filing before the relevant data window may be missing.

Impact:

- The highest-risk population is likely under-represented.
- The distressed / near-default proxy cluster contains financially weak surviving companies, not necessarily actual defaulters.
- The model may understate true distress risk relative to a dataset containing historical default outcomes.

Mitigation:

- disclose survivorship bias;
- avoid probability-of-default language;
- treat the weakest cluster as a stress proxy;
- consider adding historical delisted/defaulted company data in future versions.

---

## 4. US public-company bias

The training universe is based on SEC filers. It is therefore biased toward US-listed public companies, including some ADRs and foreign issuers that file with the SEC.

Private companies, SMEs, Bulgarian companies, and IFRS-only companies may differ in:

- accounting definitions;
- disclosure granularity;
- capital structure;
- access to refinancing;
- industry mix;
- size distribution;
- survival patterns.

Impact:

Scoring a private or non-US company is useful as a benchmark, but it is partly out-of-distribution.

Mitigation:

- use feature coverage and warning flags;
- disclose that the benchmark is SEC-public-company based;
- avoid direct external rating language;
- consider future regional/private-company calibration.

---

## 5. Point-in-time model, not through-the-cycle rating

The model scores a company based on one fiscal-year financial profile. It does not smooth through economic cycles and does not explicitly model migration paths.

The current training period includes unusual macroeconomic conditions, including the COVID shock, recovery, inflation, and higher interest-rate environment.

Impact:

- The same issuer can appear in different risk clusters in different years.
- A strong year can improve the label even if long-term business risk remains high.
- A weak year can worsen the label even if the company has strategic support or temporary shock effects.

Mitigation:

- use multi-year scoring where available;
- analyze movement over time;
- present labels as point-in-time financial profile signals;
- do not present them as through-the-cycle credit ratings.

---

## 6. Missing data and imputation risk

The KMeans pipeline uses imputation for missing model features. Median imputation can pull low-coverage companies toward the center of the distribution.

Impact:

- A company with missing debt-service data may appear less risky than it truly is.
- Low-data observations may be assigned to middle clusters because imputed values resemble typical companies.
- Feature coverage becomes part of interpretation.

Mitigation:

- show `feature_coverage_pct` in outputs;
- flag low coverage in reports;
- avoid strong conclusions when coverage is weak;
- future improvement: add missingness indicators or stricter minimum coverage.

Recommended interpretation:

| Feature coverage | Interpretation |
|---:|---|
| 1.00 | Strong input completeness |
| 0.80–0.99 | Generally usable |
| 0.67–0.79 | Interpret with caution |
| below 0.67 | Weak basis for scoring; manual review needed |

---

## 7. Financial companies excluded

The model is designed for non-financial companies. It should not be applied to banks, insurers, or financial institutions.

Reason:

- high leverage is normal for banks;
- deposits and financial liabilities are operating inputs, not distress signals;
- EBITDA is not meaningful for banks or insurers;
- liquidity ratios are interpreted differently;
- regulatory capital frameworks matter.

Mitigation:

- keep SIC-based financial exclusion;
- add a separate financial-institution model in a future version;
- do not manually override this limitation for convenience.

---

## 8. EBITDA availability and reconstruction

EBITDA is not always directly reported. The project may estimate EBITDA as:

```text
operating_income + depreciation_amortization
```

If the required inputs are missing, EBITDA-dependent ratios may be unavailable and the debt-service feature may fall back to legacy coverage/FCF logic.

Impact:

- companies with direct EBITDA and companies with reconstructed EBITDA may not be perfectly comparable;
- missing EBITDA weakens debt-service interpretation;
- incorrect `operating_income` mapping can materially distort EBITDA and interest coverage.

Mitigation:

- document whether EBITDA is direct or reconstructed;
- ensure `operating_income` means EBIT, not EBT;
- disclose missing EBITDA effects in reports;
- consider separate EBITDA quality flags in future versions.

---

## 9. No qualitative credit factors

The model uses financial statement data only. It does not include:

- management quality;
- ownership support;
- corporate governance;
- customer concentration;
- supplier risk;
- covenant package;
- collateral;
- refinancing access;
- market position;
- sector outlook;
- country risk;
- legal disputes.

Impact:

The model can miss both upside and downside credit factors that a human analyst would consider.

Mitigation:

Use the model as a structured financial diagnostic, then layer qualitative analysis separately.

---

## 10. No external ground-truth validation yet

The current project focuses on internal validation:

- cluster metrics;
- financial monotonicity;
- representative company review;
- alternative clustering comparisons;
- sensitivity analysis plan;
- guardrails.

It does not yet perform supervised validation against external ratings or observed defaults.

Impact:

The labels are plausible and interpretable, but not empirically calibrated to default risk.

Mitigation:

Future versions should compare cluster labels with:

- agency ratings;
- bond yields;
- CDS spreads;
- bankruptcy events;
- delisting/default databases.

---

## 11. Guardrails are diagnostics, not automatic overrides

Guardrails detect specific red flags such as negative equity, weak coverage, high leverage, or poor liquidity. They are not a second model and do not automatically change cluster assignment.

Impact:

A company may remain assigned to a relatively strong cluster while guardrails still require caution.

Mitigation:

- show guardrails clearly in reports;
- explain the contradiction when it occurs;
- use human judgment before drawing a commercial conclusion.

---

## 12. Mitigation summary table

| Limitation | Impact | Current mitigation | Future improvement |
|---|---|---|---|
| Not a formal rating | Overclaiming risk | Explicit disclaimers and cautious labels | External rating/default calibration |
| Domain-guided features | Analyst assumptions affect output | Thresholds documented in config manual | Sensitivity testing and data-driven calibration |
| Survivorship bias | Weakest risk tail under-represented | Disclose distressed cluster as proxy | Add delisted/defaulted companies |
| US public-company bias | Private/non-US scoring is out-of-distribution | FX normalization and warnings | Regional/private benchmark model |
| Missing data | Median imputation may centralize observations | Feature coverage output | Missingness indicators / stricter coverage |
| Financial companies excluded | Model invalid for banks/insurers | SIC-based exclusion | Separate financial-sector model |
| EBITDA inconsistency | Debt-service metrics can vary in quality | EBITDA fallback logic | EBITDA quality flags |
| No qualitative factors | Incomplete credit view | Guardrails and report limitations | Analyst qualitative overlay |
| No external validation | Labels not calibrated | Internal metrics and profiles | Ratings/default/bond spread validation |

---

## 13. Correct wording for reports and notebooks

Use:

```text
model-relative credit-risk bucket
financial-risk profile
cluster-derived risk label
credit screening tool
analyst-support diagnostic
```

Avoid:

```text
credit rating
default probability
investment grade rating
bank approval model
objective PD model
```

---

## 14. Bottom line

The model is useful because it is transparent, reproducible, financially interpretable, and technically aligned with unsupervised learning. Its limitations are manageable if the output is framed correctly.

The correct conclusion is:

```text
This project provides structured credit-risk benchmarking, not a replacement for professional credit judgment.
```
