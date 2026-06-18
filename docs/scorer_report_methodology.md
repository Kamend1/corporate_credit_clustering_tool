# Scorer Report Methodology

## Purpose of this document

This document explains how Notebook 03 and the reporting utilities convert a company’s financial statements into a scored output, scenario analysis, Excel workbook, and PDF credit-risk diagnostic report.

It is project documentation for future development. The final SoftUni notebooks should still contain direct explanatory markdown for the examiner.

---

## 1. Overview

Notebook 03 applies the trained KMeans artifact from Notebook 02 to manually entered or private-company financials.

Main use cases:

- score a private company against the public-company benchmark;
- rescore a public company using updated financials;
- run scenario analysis;
- produce Excel and PDF outputs;
- demonstrate an end-to-end ML application beyond training.

The scoring pipeline is implemented mainly through:

| Component | Role |
|---|---|
| `features.py` | Builds ratios, component risks, and domain features. |
| `scoring.py` | Applies the saved model artifact and computes affinities. |
| `diagnostics.py` | Adds adjacent-bucket distances and outlook. |
| `guardrails.py` | Adds post-model analyst caution logic. |
| `credit_report_util.py` | Creates Excel / tabular reporting outputs. |
| `credit_pdf_report_util.py` | Creates the professional PDF report. |

---

## 2. End-to-end scoring flow

```text
Raw financial inputs
        ↓
Input schema alignment and missing-value handling
        ↓
FX normalization, if required
        ↓
Derived accounting values
        ↓
Base financial ratios
        ↓
Component risk scores
        ↓
Six domain-level model features
        ↓
KMeans cluster assignment
        ↓
Distance and affinity diagnostics
        ↓
Adjacent-bucket outlook
        ↓
Warning flags and guardrails
        ↓
Excel workbook and PDF report
```

This flow is important academically because it shows that the project is not only fitting a model. It also operationalizes the model into a repeatable scoring/reporting workflow.

---

## 3. Input schema

The scoring input can be a dictionary or a DataFrame row. All monetary values should use the same unit and currency before scoring.

### Minimum usable input

The minimum input for a basic score is:

| Field | Meaning |
|---|---|
| `assets` | Total assets |
| `liabilities` | Total liabilities |
| `equity` | Total equity |
| `revenue` | Revenue / sales |
| `net_income` | Profit after tax |
| `cfo` | Operating cash flow |

This minimum can produce some core profitability, cash-flow, leverage, and structural risk features.

### Recommended complete input

For a more reliable credit score, include:

| Field | Why it matters |
|---|---|
| `cash` | Cash buffer and net debt calculation |
| `current_assets` / `assets_current` | Current ratio |
| `current_liabilities` / `liabilities_current` | Current and quick ratio |
| `receivables` | Quick ratio |
| `inventory` | Future working-capital diagnostics |
| `long_term_debt` | Debt load |
| `short_term_debt` | Debt load |
| `interest_expense` | Interest coverage |
| `operating_income` | EBIT and EBITDA reconstruction |
| `depreciation_amortization` | EBITDA reconstruction |
| `ebitda` | Direct EBITDA, if available |
| `capex` | Free cash flow |
| `gross_profit` | Gross margin diagnostics |

### Missing data consequences

| Missing item | Main consequence |
|---|---|
| Debt fields | Leverage and debt-service ratios weaken or become unavailable. |
| Interest expense | Coverage ratios unavailable. |
| Current assets/liabilities | Liquidity risk weakens. |
| EBITDA / D&A / operating income | EBITDA leverage and coverage unavailable. |
| Capex | FCF/debt unavailable. |
| Cash | Cash buffer and net debt less reliable. |

The output column `feature_coverage_pct` should be reviewed whenever inputs are incomplete.

---

## 4. Currency and units

All input values must be internally consistent.

Examples of acceptable input bases:

```text
all values in USD
all values in thousand USD
all values in BGN, then converted using fx_to_model_currency
```

Ratios are mostly scale-neutral, but the following are not:

- `log_assets`;
- asset-size bands;
- assets-below-threshold warnings;
- any report text discussing absolute size.

Therefore, if the model currency is USD and the input company reports in BGN, apply the FX multiplier consistently.

---

## 5. Feature engineering during scoring

Notebook 03 calls the same `engineer_private_company_features()` logic used in training.

This ensures consistency between:

```text
training feature space
```

and:

```text
private-company scoring feature space
```

The scoring path may also apply:

| Option | Meaning |
|---|---|
| `winsor_caps` | Optional training-time caps if stored in the artifact. |
| `fx_to_model_currency` | Currency conversion multiplier. |
| `min_denominator` | SME-compatible ratio denominator threshold. |

In the current v4 scorecard model, winsorization is not the central modeling step. The model relies mainly on bounded risk transformations.

The bounded transformations use the same current `RISK_THRESHOLDS` as the training path. These thresholds are professionally selected, judgment-based breakpoints. They are not agency-rating rules. Debt-service thresholds are intentionally conservative, so the report can distinguish excellent coverage from merely adequate coverage.

---

## 6. Cluster assignment

The saved artifact contains a scikit-learn pipeline, typically:

```text
SimpleImputer → KMeans
```

Scoring uses:

```python
pipeline.predict(X_new)
```

to assign the nearest cluster, and:

```python
pipeline.transform(X_new)
```

to compute distances to all cluster centroids.

The assigned cluster is the centroid with the lowest Euclidean distance in the six-dimensional risk-feature space.

Important:

```text
The raw assigned cluster ID is not the business label.
The business label comes from the artifact's risk-rank mapping.
```

---

## 7. Soft affinity

Distances are converted into soft affinities:

```text
a_j = exp(-d_j / T) / Σ_i exp(-d_i / T)
```

where `T` is the temperature parameter.

| Temperature | Effect |
|---:|---|
| 0.3 | Very sharp; mostly assigned cluster. |
| 1.0 | Balanced default. |
| 3.0 | Flatter; more affinity spread across clusters. |

Affinity is a similarity measure. It is not probability of default and should not be presented as such.

---

## 8. Adjacent-bucket outlook

The adjacent-bucket diagnostic compares the company to the next stronger and next weaker risk buckets.

Simplified logic:

```text
if closer to stronger adjacent bucket by enough margin:
    Positive
elif closer to weaker adjacent bucket by enough margin:
    Negative
else:
    Neutral
```

The current logic uses a neutral band and boundary multipliers to avoid overreacting to small distance differences.

Interpretation:

| Outlook | Meaning |
|---|---|
| Positive | Current profile leans toward a stronger bucket. |
| Neutral | Current profile is reasonably centered in assigned bucket. |
| Negative | Current profile leans toward a weaker bucket. |

The outlook is not a forecast. It is a static cluster-position diagnostic.

---

## 9. Scenario analysis

Notebook 03 includes mechanical scenarios that modify base financial inputs and rescore the company.

Typical scenarios:

| Scenario | Description |
|---|---|
| `base` | Reported or manually entered financials. |
| `revenue_down_15pct` | Revenue, profit, operating income, and CFO stress. |
| `debt_up_25pct` | Debt and liabilities increase. |
| `cash_burn_case` | Cash falls and profitability/cash flow weaken. |
| `near_default_stress` | Balance sheet and coverage pushed toward distress. |

Scenarios are sensitivities, not forecasts.

Correct wording:

```text
Under this mechanical stress scenario, the company migrates to a weaker model-relative risk bucket.
```

Avoid:

```text
The company is expected to be downgraded.
```

---

## 10. Company vs cluster median comparison

The report compares the scored company to the median profile of its assigned cluster.

| Output field | Meaning |
|---|---|
| `metric` | Ratio or risk feature. |
| `company_value` | Company-specific value. |
| `assigned_cluster_median` | Median value in the assigned cluster. |
| `difference` | Company value minus cluster median. |
| `relative_position` | Above, below, or equal to cluster median. |

This comparison explains why a company is typical or atypical for its bucket.

---

## 11. Guardrail layer

Guardrails are added after model scoring. They detect red flags that require analyst caution.

Examples:

| Guardrail family | Example issue |
|---|---|
| Leverage | High debt/assets or debt/EBITDA. |
| Coverage | Weak interest coverage. |
| Liquidity | Current ratio below 1.0 or quick ratio below 0.5. |
| Structural distress | Negative equity or liabilities exceeding assets. |

Guardrails are important because a cluster label can be too optimistic when a company has one severe red flag but looks normal on other dimensions.

---

## 12. Excel report structure

The Excel report should be treated as the analyst workbook.

Recommended tabs:

| Tab | Purpose |
|---|---|
| `Score Summary` | Main label, risk rank, affinity, outlook, guardrails. |
| `Ratio Snapshot` | Company financial ratios and risk features. |
| `Cluster Comparison` | Company vs assigned-cluster median. |
| `Scenario Analysis` | Base and stress scenario scoring. |
| `Guardrails` | Triggered caution items and explanation. |
| `Model Metadata` | Artifact version, feature list, assumptions. |

The Excel output is useful for transparent review and debugging.

---

## 13. PDF report structure

The PDF report should be concise, professional, and safe from overclaiming.

Recommended sections:

| PDF section | Purpose |
|---|---|
| Executive summary | One-page conclusion with model-relative label. |
| Risk label scale | Explains the 1–5 labels. |
| Company snapshot | Key financial ratios and risk drivers. |
| Model diagnostics | Affinity, near-default affinity, distance, feature coverage. |
| Guardrails | Analyst caution layer. |
| Cluster comparison | How company compares with assigned bucket. |
| Scenario analysis | Mechanical stress cases. |
| Limitations | Prevents rating/PD overclaiming. |

The first page should emphasize the business label and risk scale, not the raw cluster ID. In the current v4 label scale, the third bucket is a loss-making / cash-flow weak profile and the fourth bucket is a leveraged / weak operating credit profile.

---

## 14. Reporting discipline

Use:

```text
model-relative risk bucket
financial-risk diagnostic
benchmark-relative credit profile
```

Avoid:

```text
credit rating
probability of default
bank approval
investment-grade rating
```

The report supports analyst judgment. It does not replace credit analysis.

---

## 15. Examiner / final-project relevance

Notebook 03 demonstrates that the project goes beyond model fitting. It shows:

- reusable source-code architecture;
- private-company inference;
- scenario analysis;
- explainable model outputs;
- professional reporting;
- guardrails and limitations.

This supports the SoftUni grading categories for code quality, methods, communication, and applied mathematical understanding.
