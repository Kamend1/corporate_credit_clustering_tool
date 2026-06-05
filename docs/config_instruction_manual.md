# Configuration Manual for `src/credit_clustering/config.py`

## Purpose of this manual

This document is the professional instruction manual for the current `config.py` file used in the credit clustering project.

The `config.py` module is intentionally **logic-free**. It centralizes the project’s:

- model feature lists;
- scorecard domain weights;
- denominator and materiality thresholds;
- clustering defaults;
- profiling labels;
- scoring assumptions;
- artifact metadata;
- input column definitions;
- risk transformation thresholds;
- reporting columns;
- consulting guardrail configuration.

The purpose of this design is to make the project tunable without forcing users to edit the model logic in `features.py`, `clustering.py`, `scoring.py`, `profiling.py`, `guardrails.py`, `artifacts.py`, or the notebooks.

Professionals who clone the project should treat `config.py` as the main place for customizing the credit-risk experience.

---

## Important operating principle

`config.py` defines **assumptions**, not calculations.

Changing values in `config.py` changes the behavior of downstream modules, but the actual calculations are performed elsewhere:

| Setting area | Used by |
|---|---|
| Cluster features | `clustering.py`, `scoring.py`, `artifacts.py` |
| Domain weights | `features.py`, `artifacts.py`, reports |
| Risk thresholds | `features.py` |
| Guardrails | `guardrails.py`, reports |
| Reporting columns | Notebook 03, `credit_report_util.py`, `credit_pdf_report_util.py` |
| Artifact defaults | `artifacts.py`, Notebook 02, Notebook 03 |
| Scoring defaults | Notebook 03, `scoring.py` calls |
| Profiling defaults | `profiling.py`, Notebook 02 |

A professional user should therefore tune `config.py` first, rerun the relevant notebook, and only then review whether deeper source-code changes are needed.

---

# 1. Core model feature set

## 1.1 `SCORECARD_CLUSTER_FEATURES`

Current value:

```python
SCORECARD_CLUSTER_FEATURES = [
    "structural_distress_risk",
    "earnings_risk",
    "operating_cashflow_risk",
    "liquidity_risk",
    "leverage_risk",
    "debt_service_risk",
]
```

These six variables are the **domain-level features used by the KMeans clustering model**.

They are not raw accounting ratios. They are normalized credit-risk dimensions, typically bounded between 0 and 1, where:

```text
0.00 = stronger / lower-risk characteristic
1.00 = weaker / higher-risk characteristic
```

## Professional interpretation

The current feature set reflects a traditional credit analyst’s logic:

| Feature | Credit meaning |
|---|---|
| `structural_distress_risk` | hard balance-sheet distress: negative equity or liabilities above assets |
| `earnings_risk` | weak or negative profitability relative to assets |
| `operating_cashflow_risk` | weak or negative operating cash-flow generation |
| `liquidity_risk` | insufficient current, quick, or cash liquidity |
| `leverage_risk` | excessive liabilities, debt, or weak equity buffer |
| `debt_service_risk` | weak ability to cover interest and debt burden using earnings / cash flow |

## Why these six were selected

The model is intended to be:

- explainable;
- suitable for non-financial companies;
- robust across inconsistent public-company reporting;
- usable for SME/private-company scoring after manual input mapping;
- defensible in a consulting report.

The feature set avoids highly specialized sector-specific ratios and focuses on core credit fundamentals.

## Fine-tuning guidance

A professional user may modify this list, but should be careful.

### Safe customizations

Add or remove features only if:

- the feature is available for most training observations;
- the feature is directionally consistent;
- the feature is properly engineered in `features.py`;
- the feature does not duplicate another dominant feature.

### Example: adding size risk

```python
SCORECARD_CLUSTER_FEATURES = [
    "structural_distress_risk",
    "earnings_risk",
    "operating_cashflow_risk",
    "liquidity_risk",
    "leverage_risk",
    "debt_service_risk",
    "size_risk",
]
```

But this requires `size_risk` to exist in the engineered dataframe.

### Professional warning

Adding too many features to KMeans can degrade interpretability and create distance effects driven by noisy or sparse variables. For consulting use, a compact and explainable feature set is generally better than a large black-box feature space.

---

# 2. Scorecard domain weights

## 2.1 `SCORECARD_DOMAIN_WEIGHTS`

Current value:

```python
SCORECARD_DOMAIN_WEIGHTS = {
    "leverage_risk": 0.25,
    "liquidity_risk": 0.20,
    "earnings_risk": 0.15,
    "operating_cashflow_risk": 0.20,
    "debt_service_risk": 0.15,
    "structural_distress_risk": 0.05,
}
```

These weights are used to construct the `scorecard_credit_score`.

The current weights sum to:

```text
1.00
```

The score is effectively a weighted summary of domain-level credit risk.

## Professional assumptions behind the defaults

The current weighting framework reflects a conservative credit analyst view:

| Domain | Weight | Rationale |
|---|---:|---|
| Leverage | 25% | Capital structure is central to default risk and refinancing capacity |
| Liquidity | 20% | Short-term liquidity often determines whether stress becomes default |
| Operating cash flow | 20% | Cash generation is more robust than accounting earnings |
| Earnings | 15% | Profitability matters, but accounting income can be noisy |
| Debt service | 15% | Important, but coverage data may have lower availability |
| Structural distress | 5% | Hard distress indicator, but binary and low-gradient |

## Why structural distress receives only 5%

`structural_distress_risk` is powerful but crude. It is often binary:

```text
0 = no structural distress
1 = negative equity or liabilities exceed assets
```

Because it lacks gradient, it should not dominate the score unless used as a guardrail. The model keeps it in the cluster features, but the scorecard weight is intentionally modest.

## Why leverage receives the highest weight

Leverage is the most persistent credit-risk driver. High leverage can remain dangerous even when a company has temporary liquidity or positive earnings.

For consulting use, leverage deserves a high weight because it affects:

- refinancing risk;
- covenant pressure;
- downturn resilience;
- debt-service flexibility;
- equity cushion.

## Fine-tuning guidance

### More conservative lender view

Increase leverage and debt-service weights:

```python
SCORECARD_DOMAIN_WEIGHTS = {
    "leverage_risk": 0.30,
    "liquidity_risk": 0.20,
    "earnings_risk": 0.10,
    "operating_cashflow_risk": 0.20,
    "debt_service_risk": 0.15,
    "structural_distress_risk": 0.05,
}
```

### SME working-capital view

Increase liquidity and cash flow:

```python
SCORECARD_DOMAIN_WEIGHTS = {
    "leverage_risk": 0.20,
    "liquidity_risk": 0.25,
    "earnings_risk": 0.15,
    "operating_cashflow_risk": 0.25,
    "debt_service_risk": 0.10,
    "structural_distress_risk": 0.05,
}
```

### Distress-screening view

Increase structural distress and debt service:

```python
SCORECARD_DOMAIN_WEIGHTS = {
    "leverage_risk": 0.25,
    "liquidity_risk": 0.15,
    "earnings_risk": 0.10,
    "operating_cashflow_risk": 0.20,
    "debt_service_risk": 0.20,
    "structural_distress_risk": 0.10,
}
```

## Validation checklist after changing weights

After changing weights:

1. Confirm they sum to 1.00.
2. Recompute score distributions.
3. Check cluster medians by score.
4. Inspect top and bottom scored companies.
5. Confirm the score direction remains intuitive.
6. Rerun Notebook 02 and Notebook 03 from a clean kernel.

---

# 3. Low-level component features

## 3.1 `SCORECARD_COMPONENT_FEATURES`

Current value:

```python
SCORECARD_COMPONENT_FEATURES = [
    "liabilities_risk",
    "debt_load_risk",
    "equity_buffer_risk",
    "cash_buffer_risk",
    "current_liquidity_risk",
    "quick_liquidity_risk",
    "profitability_risk",
    "cashflow_risk",
    "coverage_risk",
    "fcf_risk",
    "ebitda_margin_risk",
    "debt_to_ebitda_risk",
    "net_debt_to_ebitda_risk",
    "ebitda_coverage_risk",
    "negative_ebitda_flag",
    "negative_equity_flag",
    "liabilities_exceed_assets_flag",
]
```

These are the lower-level features used to construct the domain-level scorecard features.

They allow professionals to audit why a domain score looks high or low.

## Professional use

A credit professional should use these component features for diagnostic interpretation, not necessarily for clustering.

For example:

| Domain-level risk | Component drivers |
|---|---|
| `liquidity_risk` | `cash_buffer_risk`, `current_liquidity_risk`, `quick_liquidity_risk` |
| `leverage_risk` | `liabilities_risk`, `debt_load_risk`, `equity_buffer_risk` |
| `debt_service_risk` | `coverage_risk`, `fcf_risk`, `debt_to_ebitda_risk`, `net_debt_to_ebitda_risk`, `ebitda_coverage_risk` |
| `structural_distress_risk` | `negative_equity_flag`, `liabilities_exceed_assets_flag` |

## Fine-tuning guidance

Professionals can expand this list if they create additional sub-risk measures in `features.py`.

Examples:

- `inventory_intensity_risk`;
- `receivables_quality_risk`;
- `revenue_decline_risk`;
- `gross_margin_risk`;
- `working_capital_absorption_risk`.

But do not add a component here unless it is actually computed.

---

# 4. Guardrail / analyst interpretation columns

## 4.1 `GUARDRAIL_COLS`

Current value:

```python
GUARDRAIL_COLS = [
    "guardrail_level",
    "guardrail_flags",
    "guardrail_summary",
    "analyst_interpretation",
    "commercial_conclusion",
]
```

These columns are added by `guardrails.py` after scoring.

They are designed for consulting use. The mechanical model assigns clusters; guardrails qualify the interpretation.

## Professional purpose

The guardrail layer prevents outputs like:

```text
“Top bucket = investment-grade-like”
```

when the underlying financial profile has material caveats.

The guardrails are not intended to override the cluster mechanically. They are intended to force professional judgment into the report.

## Column explanation

| Column | Meaning |
|---|---|
| `guardrail_level` | highest severity level triggered by the guardrail rules |
| `guardrail_flags` | machine-readable list of triggered issues |
| `guardrail_summary` | short report-facing summary |
| `analyst_interpretation` | professional interpretation paragraph |
| `commercial_conclusion` | client-safe conclusion and recommended caution |

## Consulting assumption

For professional use, the model output should be presented as:

```text
relative financial-risk benchmarking
```

not as:

```text
formal external credit rating
probability of default
bank-grade approval decision
```

The guardrail columns make that distinction visible in Excel and PDF reports.

---

# 5. Guardrail severity levels

## 5.1 `GUARDRAIL_SEVERITY_ORDER`

Current value:

```python
GUARDRAIL_SEVERITY_ORDER = {
    "Clear": 0,
    "Monitor": 1,
    "Caution": 2,
    "High caution": 3,
    "Override required": 4,
}
```

The highest triggered severity becomes the final `guardrail_level`.

## Severity interpretation

| Severity | Meaning | Consulting treatment |
|---|---|---|
| Clear | no material contradiction detected | output can be presented normally with standard limitations |
| Monitor | minor weakness detected | explain but do not challenge model result |
| Caution | meaningful caveat detected | qualify the conclusion |
| High caution | material weakness detected | do not frame result as clean low risk |
| Override required | severe contradiction or distress signal | manual analyst override required |

## Professional assumptions

The severity scale is intentionally conservative. A credit consultant should be able to say:

```text
“The model assigned a strong bucket, but our guardrail layer requires caution.”
```

That makes the tool commercially safer.

---

# 6. Guardrail rule configuration

Guardrails are organized into four direct rule groups:

1. leverage;
2. debt service;
3. liquidity;
4. structural distress.

They are combined into `CREDIT_GUARDRAILS`.

Each rule has:

```python
{
    "column": "...",
    "operator": "...",
    "threshold": ...,
    "severity": "...",
}
```

## 6.1 Leverage guardrails

Current rules:

```python
LEVERAGE_GUARDRAILS = {
    "elevated_debt_to_assets": debt_to_assets > 0.45 => Caution,
    "high_debt_to_assets": debt_to_assets > 0.60 => High caution,
    "elevated_debt_to_ebitda": debt_to_ebitda > 3.5 => Caution,
    "high_debt_to_ebitda": debt_to_ebitda > 5.0 => High caution,
    "elevated_net_debt_to_ebitda": net_debt_to_ebitda > 3.0 => Caution,
    "high_net_debt_to_ebitda": net_debt_to_ebitda > 4.5 => High caution,
}
```

## Professional assumptions

These thresholds reflect a broad non-financial corporate credit framework.

### Debt / assets

| Threshold | Interpretation |
|---:|---|
| > 45% | leverage requires qualification |
| > 60% | leverage is high and materially constrains credit interpretation |

A debt/assets ratio above 45% does not mean default risk is high by itself. But it does mean the company is meaningfully debt-financed.

### Debt / EBITDA

| Threshold | Interpretation |
|---:|---|
| > 3.5x | elevated leverage |
| > 5.0x | high leverage |

These are broad corporate-credit thresholds. They are not universal. Utilities, infrastructure, real estate, telecoms, and project-finance companies may tolerate higher leverage than asset-light operating companies.

### Net debt / EBITDA

| Threshold | Interpretation |
|---:|---|
| > 3.0x | elevated net leverage |
| > 4.5x | high net leverage |

Net debt / EBITDA is often more meaningful than gross debt / EBITDA where the company has large cash balances.

## Fine-tuning guidance

### Conservative bank lending setting

Lower thresholds:

```python
"elevated_debt_to_ebitda": threshold 3.0
"high_debt_to_ebitda": threshold 4.0
"elevated_net_debt_to_ebitda": threshold 2.5
"high_net_debt_to_ebitda": threshold 4.0
```

### Infrastructure / utility setting

Raise thresholds modestly:

```python
"elevated_debt_to_ebitda": threshold 4.0
"high_debt_to_ebitda": threshold 5.5
"elevated_net_debt_to_ebitda": threshold 3.5
"high_net_debt_to_ebitda": threshold 5.0
```

But only do this if regulated cash flows, long maturities, and stable tariff frameworks are explicitly considered.

---

## 6.2 Debt-service guardrails

Current rules:

```python
DEBT_SERVICE_GUARDRAILS = {
    "weak_interest_coverage": interest_coverage < 2.0 => Caution,
    "critical_interest_coverage": interest_coverage < 1.0 => High caution,
    "weak_ebitda_interest_coverage": ebitda_interest_coverage < 3.0 => Caution,
    "weak_fcf_to_debt": fcf_to_debt < 0.10 => Caution,
}
```

## Professional assumptions

Debt service is where accounting profitability meets creditor reality.

### Interest coverage

| Threshold | Interpretation |
|---:|---|
| < 2.0x | weak coverage |
| < 1.0x | critical coverage |

An interest coverage ratio below 1.0x usually means EBIT does not cover interest expense.

Professional warning: this ratio is highly sensitive to correct input mapping.

`operating_income` should normally mean EBIT, not EBT. If EBT is used by mistake, interest coverage will be understated.

### EBITDA interest coverage

| Threshold | Interpretation |
|---:|---|
| < 3.0x | weak EBITDA-based interest cushion |

EBITDA coverage is less conservative than EBIT coverage because it excludes depreciation and amortization, but it is often more useful for debt capacity analysis.

### FCF / debt

| Threshold | Interpretation |
|---:|---|
| < 10% | weak free-cash-flow debt repayment capacity |

A company can have positive EBITDA but weak free cash flow due to capex or working-capital absorption. This guardrail captures that issue.

## Fine-tuning guidance

### Conservative lender setting

```python
weak_interest_coverage threshold = 2.5
weak_ebitda_interest_coverage threshold = 4.0
weak_fcf_to_debt threshold = 0.15
```

### Growth company setting

You may tolerate lower FCF/debt if capex is strategic and financing is stable, but the report should explicitly say so.

---

## 6.3 Liquidity guardrails

Current rules:

```python
LIQUIDITY_GUARDRAILS = {
    "current_ratio_below_1": current_ratio < 1.0 => Caution,
    "quick_ratio_below_0_5": quick_ratio < 0.5 => High caution,
    "low_cash_to_assets": cash_to_assets < 0.03 => Monitor,
}
```

## Professional assumptions

Liquidity stress often materializes before accounting insolvency.

### Current ratio

A current ratio below 1.0x means current liabilities exceed current assets.

This does not always imply distress, but it requires explanation, especially for SMEs.

### Quick ratio

A quick ratio below 0.5x is a stronger warning because it removes inventory from current assets.

### Cash / assets

A cash/assets ratio below 3% is not automatically dangerous, but it indicates limited immediate cash buffer.

## Fine-tuning guidance

For inventory-heavy businesses, quick ratio may be harsh. For trading businesses, cash turnover and receivable quality may matter more.

Professionals may adjust thresholds by sector.

---

## 6.4 Structural distress guardrails

Current rules:

```python
STRUCTURAL_GUARDRAILS = {
    "negative_equity": equity_to_assets < 0.0 => Override required,
    "liabilities_exceed_assets": liabilities_to_assets > 1.0 => Override required,
}
```

## Professional assumptions

These are hard red flags.

Negative equity or liabilities exceeding assets should not be buried inside a clustering result. They require manual review even if the model assigns a non-distressed cluster.

## Fine-tuning guidance

Usually do not relax these rules.

If a sector often reports negative book equity due to buybacks or accounting effects, treat this as an analyst override topic rather than removing the guardrail.

---

## 6.5 `CREDIT_GUARDRAILS`

Current value:

```python
CREDIT_GUARDRAILS = {
    **LEVERAGE_GUARDRAILS,
    **DEBT_SERVICE_GUARDRAILS,
    **LIQUIDITY_GUARDRAILS,
    **STRUCTURAL_GUARDRAILS,
}
```

This combines all direct ratio-based guardrails into one dictionary.

`guardrails.py` uses this combined dictionary to evaluate rules consistently.

---

## 6.6 Guardrail flag sets

Current values:

```python
LEVERAGE_GUARDRAIL_FLAGS = set(LEVERAGE_GUARDRAILS.keys())
DEBT_SERVICE_GUARDRAIL_FLAGS = set(DEBT_SERVICE_GUARDRAILS.keys())
LIQUIDITY_GUARDRAIL_FLAGS = set(LIQUIDITY_GUARDRAILS.keys())
STRUCTURAL_GUARDRAIL_FLAGS = set(STRUCTURAL_GUARDRAILS.keys())
```

These are used by `guardrails.py` to build narrative logic.

For example:

```text
if top model bucket + leverage or debt-service flag => qualify conclusion
```

## 6.7 `HIGH_CAUTION_GUARDRAIL_FLAGS`

Current value:

```python
HIGH_CAUTION_GUARDRAIL_FLAGS = {
    name
    for name, rule in CREDIT_GUARDRAILS.items()
    if rule.get("severity") == "High caution"
}
```

This automatically collects all rules with severity `"High caution"`.

This is useful because if you later add a new high-caution rule, the model-contradiction logic can detect it without separately maintaining another list.

---

# 7. Denominator and materiality thresholds

## 7.1 `SME_MIN_DENOMINATOR`

Current value:

```python
SME_MIN_DENOMINATOR = 1_000
```

This is used for SME-compatible ratio construction.

## Professional assumption

Small private-company financial statements can contain small denominators that make ratios unstable. At the same time, using a very high denominator threshold would suppress valid SME ratios.

The current value of `1,000` is a practical compromise:

- avoids division by tiny meaningless values;
- still allows SME-scale companies to be scored;
- is much less restrictive than public-company materiality thresholds.

## Fine-tuning guidance

### Microbusiness setting

```python
SME_MIN_DENOMINATOR = 100
```

### More conservative professional setting

```python
SME_MIN_DENOMINATOR = 10_000
```

### Larger corporate-only setting

```python
SME_MIN_DENOMINATOR = 100_000
```

---

## 7.2 `PUBLIC_COMPANY_MIN_ASSETS`

Current value:

```python
PUBLIC_COMPANY_MIN_ASSETS = 1_000_000
```

This is used as a training/materiality reference for public-company datasets.

## Professional assumption

Public-company observations with extremely small asset bases may represent:

- shells;
- pre-revenue entities;
- distressed microcaps;
- incomplete data;
- reverse mergers;
- non-operating entities.

The threshold helps reduce noise in training.

## Fine-tuning guidance

### Broader universe

```python
PUBLIC_COMPANY_MIN_ASSETS = 500_000
```

### Higher-quality public-company universe

```python
PUBLIC_COMPANY_MIN_ASSETS = 5_000_000
```

### Larger corporate benchmarking

```python
PUBLIC_COMPANY_MIN_ASSETS = 50_000_000
```

---

## 7.3 Size-band thresholds

Current values:

```python
SMALL_COMPANY_ASSET_LIMIT = 5_000_000
MID_COMPANY_ASSET_LIMIT = 50_000_000
LARGE_COMPANY_ASSET_LIMIT = 100_000_000_000
```

These are diagnostic only. They are not clustering features.

## Professional assumptions

The thresholds create broad asset-size bands for reporting and interpretation.

| Band logic | Meaning |
|---|---|
| below 5m assets | small company |
| 5m to 50m assets | mid-sized company |
| above 50m assets | larger company |
| 100bn cap | practical upper boundary / outlier control |

## Fine-tuning guidance

These thresholds can be localized.

For Bulgaria / SME consulting, a professional may choose lower bands. For US public-company analysis, the current values are modest.

---

# 8. Clustering defaults

## 8.1 `DEFAULT_SEGMENT_COL`

Current value:

```python
DEFAULT_SEGMENT_COL = "financial_flag"
```

This is the column used to separate financial and non-financial companies.

## Professional assumption

Financial institutions have fundamentally different balance sheets:

- banks have financial assets and liabilities by design;
- insurers have policy liabilities;
- leverage ratios are not comparable to industrial companies;
- working capital ratios are not interpreted the same way.

Therefore, this project currently targets:

```text
Non-financial companies only
```

## 8.2 `DEFAULT_TARGET_SEGMENTS`

Current value:

```python
DEFAULT_TARGET_SEGMENTS = ("Non-financial",)
```

Only the non-financial segment is clustered by default.

## Fine-tuning guidance

Do not add `"Financial"` unless you build a separate financial-institution feature framework.

---

## 8.3 `DEFAULT_N_CLUSTERS`

Current value:

```python
DEFAULT_N_CLUSTERS = 5
```

The model uses 5 clusters.

## Professional assumption

Five clusters align with a practical consulting risk scale:

1. strong profile;
2. good profile;
3. leveraged/elevated risk;
4. weak profile;
5. distressed/near-default proxy.

This is granular enough for business interpretation without pretending to replicate rating-agency notches.

## Fine-tuning guidance

### 3 clusters

Simpler management reporting:

```python
DEFAULT_N_CLUSTERS = 3
```

Example interpretation:

- low risk;
- medium risk;
- high risk.

### 5 clusters

Current recommended default.

### 7+ clusters

Not recommended for first consulting use. More clusters may improve mathematical segmentation but can make business interpretation fragile.

---

## 8.4 `DEFAULT_MIN_ROWS_PER_SEGMENT`

Current value:

```python
DEFAULT_MIN_ROWS_PER_SEGMENT = 500
```

A segment must have at least 500 rows to be clustered.

## Professional assumption

KMeans cluster profiles become unstable when trained on small samples. A minimum of 500 issuer-years is a reasonable floor for a broad unsupervised model.

## Fine-tuning guidance

For smaller curated datasets, a professional may reduce this to 200, but should then inspect cluster stability carefully.

---

## 8.5 Feature coverage defaults

Current values:

```python
DEFAULT_MIN_FEATURES = 4
DEFAULT_ROW_FEATURE_COVERAGE = 0.60
DEFAULT_MIN_FEATURE_COVERAGE = 0.0
```

## Interpretation

| Setting | Meaning |
|---|---|
| `DEFAULT_MIN_FEATURES = 4` | each row should have at least 4 usable model features |
| `DEFAULT_ROW_FEATURE_COVERAGE = 0.60` | row should have at least 60% feature availability |
| `DEFAULT_MIN_FEATURE_COVERAGE = 0.0` | do not globally drop features based on availability |

## Professional assumption

The project accepts partial feature availability because public filings and private-company inputs are not always complete.

This is especially relevant for:

- EBITDA;
- interest expense;
- debt;
- FCF;
- coverage ratios.

## Fine-tuning guidance

### Stricter model

```python
DEFAULT_MIN_FEATURES = 5
DEFAULT_ROW_FEATURE_COVERAGE = 0.80
```

### More inclusive SME model

```python
DEFAULT_MIN_FEATURES = 3
DEFAULT_ROW_FEATURE_COVERAGE = 0.50
```

Professional warning: loosening coverage increases sample size but may create clusters partly driven by missing-data patterns.

---

## 8.6 `DEFAULT_RANDOM_STATE`

Current value:

```python
DEFAULT_RANDOM_STATE = 42
```

This ensures reproducibility.

## 8.7 `DEFAULT_N_INIT`

Current value:

```python
DEFAULT_N_INIT = 500
```

KMeans is run with many initializations.

## Professional assumption

A high `n_init` improves stability and reduces the risk of a poor local optimum.

For a final project or consulting prototype, `500` is defensible even if slower.

## Fine-tuning guidance

For speed:

```python
DEFAULT_N_INIT = 50
```

For final model fitting:

```python
DEFAULT_N_INIT = 500
```

For production-grade stability checks, run several random states and compare cluster profiles.

---

# 9. Profiling defaults

## 9.1 `DEFAULT_RATING_STYLE_LABELS`

Current value:

```python
DEFAULT_RATING_STYLE_LABELS = {
    1: "1 - Strong relative credit profile",
    2: "2 - Good credit profile",
    3: "3 - Leveraged / elevated risk profile",
    4: "4 - Weak credit profile",
    5: "5 - Distressed / near-default proxy",
}
```

## Professional assumption

The labels are intentionally **not external credit ratings**.

The wording avoids:

```text
investment grade
BB equivalent
default probability
rating agency grade
```

and uses:

```text
relative credit profile
model-relative risk
distressed proxy
```

This is important for consulting defensibility.

## Why this change matters

A model may assign a company to the strongest relative cluster even when external ratings, leverage, or coverage metrics require caution.

Therefore, label 1 is now:

```text
Strong relative credit profile
```

not:

```text
Investment-grade-like
```

## Fine-tuning guidance

Professionals may rename labels, but should avoid rating-agency language unless the model is formally calibrated to rating outcomes.

Safe alternatives:

```python
{
    1: "1 - Strongest model-relative bucket",
    2: "2 - Sound model-relative bucket",
    3: "3 - Elevated model-relative risk",
    4: "4 - Weak model-relative profile",
    5: "5 - Distress-like model-relative profile",
}
```

---

## 9.2 `DEFAULT_EXTREME_QUANTILES`

Current value:

```python
DEFAULT_EXTREME_QUANTILES = (
    0.001,
    0.01,
    0.05,
    0.50,
    0.95,
    0.99,
    0.999,
)
```

These quantiles are used for feature profiling and extreme value inspection.

## Professional assumption

Credit data is heavy-tailed. Extreme quantiles help identify:

- outliers;
- distressed observations;
- data errors;
- unusual leverage;
- extreme profitability;
- abnormal liquidity.

## Fine-tuning guidance

For a simpler report:

```python
DEFAULT_EXTREME_QUANTILES = (0.01, 0.05, 0.50, 0.95, 0.99)
```

For data-quality review:

```python
DEFAULT_EXTREME_QUANTILES = (0.001, 0.005, 0.01, 0.05, 0.50, 0.95, 0.99, 0.995, 0.999)
```

---

# 10. Scoring defaults

## 10.1 `DEFAULT_SCORING_SEGMENT`

Current value:

```python
DEFAULT_SCORING_SEGMENT = "Non-financial"
```

Manual/private-company scoring defaults to the non-financial model.

## 10.2 `DEFAULT_SCORING_TEMPERATURE`

Current value:

```python
DEFAULT_SCORING_TEMPERATURE = 1.0
```

This controls soft cluster affinities.

Lower temperature makes affinities sharper. Higher temperature makes them smoother.

## Professional assumption

A value of `1.0` is neutral and interpretable.

## Fine-tuning guidance

### Sharper affinity

```python
DEFAULT_SCORING_TEMPERATURE = 0.5
```

Useful if you want more decisive cluster closeness.

### Smoother affinity

```python
DEFAULT_SCORING_TEMPERATURE = 2.0
```

Useful if you want to avoid overconfidence in cluster membership.

Professional warning: cluster affinity is not probability of default. Do not present it as a probability.

---

## 10.3 `DEFAULT_FX_TO_MODEL_CURRENCY`

Current value:

```python
DEFAULT_FX_TO_MODEL_CURRENCY = 1.0
```

This multiplier converts input monetary values into the model currency.

## Professional assumption

The default assumes the input currency is already aligned with the model/reporting currency.

## Fine-tuning guidance

If input is in BGN and the model expects USD, use a conversion factor. If all ratios are scale-independent, the effect may be limited, but size features like `log_assets` will change.

Always document the currency conversion used in the report.

---

## 10.4 `DEFAULT_SCORING_MIN_DENOMINATOR`

Current value:

```python
DEFAULT_SCORING_MIN_DENOMINATOR = SME_MIN_DENOMINATOR
```

This means:

```python
DEFAULT_SCORING_MIN_DENOMINATOR = 1_000
```

Manual scoring uses SME-compatible ratio handling.

---

# 11. Artifact defaults

## 11.1 `DEFAULT_ARTIFACT_VERSION`

Current value:

```python
DEFAULT_ARTIFACT_VERSION = "v3_scorecard_ebitda"
```

This identifies the model artifact version.

## Professional assumption

The version name communicates that the current version includes EBITDA-aware features.

## Fine-tuning guidance

Update this whenever you make a model-breaking or interpretation-breaking change.

Examples:

```python
DEFAULT_ARTIFACT_VERSION = "v4_scorecard_guardrails"
DEFAULT_ARTIFACT_VERSION = "v4_sector_adjusted"
DEFAULT_ARTIFACT_VERSION = "v5_supervised_overlay"
```

## 11.2 `DEFAULT_PRIMARY_SEGMENT`

Current value:

```python
DEFAULT_PRIMARY_SEGMENT = "Non-financial"
```

This is used when saving and loading artifacts.

---

# 12. Input column groups

## 12.1 `REQUIRED_OR_OPTIONAL_FINANCIAL_COLUMNS`

Current value:

```python
REQUIRED_OR_OPTIONAL_FINANCIAL_COLUMNS = [
    "assets",
    "liabilities",
    "equity",
    "cash",
    "net_income",
    "cfo",
    "revenue",
    "long_term_debt",
    "short_term_debt",
    "assets_current",
    "current_assets",
    "liabilities_current",
    "current_liabilities",
    "receivables",
    "inventory",
    "capex",
    "operating_income",
    "gross_profit",
    "interest_expense",
    "depreciation_amortization",
    "ebitda",
]
```

These are the expected financial statement input columns.

They are called “required or optional” because the feature engineering layer can derive some metrics when direct fields are missing.

## Professional mapping guidance

| Input field | Recommended meaning |
|---|---|
| `assets` | total assets |
| `liabilities` | total liabilities |
| `equity` | total equity |
| `cash` | cash and cash equivalents |
| `net_income` | net profit after tax |
| `cfo` | cash flow from operating activities |
| `revenue` | revenue / sales |
| `long_term_debt` | long-term debt / borrowings |
| `short_term_debt` | short-term debt / current debt |
| `current_assets` | current assets |
| `current_liabilities` | current liabilities |
| `receivables` | trade and other operating receivables |
| `inventory` | inventories |
| `capex` | capital expenditures |
| `operating_income` | EBIT / operating profit before finance costs |
| `gross_profit` | gross profit |
| `interest_expense` | finance cost / interest expense |
| `depreciation_amortization` | depreciation and amortization |
| `ebitda` | EBITDA, if directly available |

## Important professional warning

`operating_income` should normally be EBIT, not EBT.

If a user maps profit before tax into `operating_income`, then interest coverage will be understated.

Recommended hierarchy:

```text
If EBITDA and depreciation/amortization are available:
    EBIT = EBITDA - depreciation_amortization

If EBIT / operating profit is explicitly disclosed:
    use it directly

Do not use EBT as operating income
```

## 12.2 `MONETARY_COLUMNS`

Current value:

```python
MONETARY_COLUMNS = REQUIRED_OR_OPTIONAL_FINANCIAL_COLUMNS.copy()
```

This means all financial input columns are treated as monetary values for FX conversion.

Professional warning: ratio columns should not be included in `MONETARY_COLUMNS`.

---

# 13. Risk thresholds

## 13.1 `RISK_THRESHOLDS`

Current value:

```python
RISK_THRESHOLDS = {
    "liabilities_to_assets": {"low": 0.45, "high": 1.00},
    "debt_to_assets": {"low": 0.25, "high": 0.85},
    "equity_to_assets": {"good": 0.40, "bad": 0.00},
    "cash_to_assets": {"good": 0.10, "bad": 0.01},
    "current_ratio": {"good": 2.00, "bad": 0.75},
    "quick_ratio": {"good": 1.00, "bad": 0.25},
    "net_income_to_assets": {"good": 0.05, "bad": -0.05},
    "cfo_to_assets": {"good": 0.08, "bad": -0.03},
    "interest_coverage": {"good": 3.00, "bad": 1.00},
    "fcf_to_debt": {"good": 0.15, "bad": -0.10},
    "ebitda_margin": {"good": 0.20, "bad": 0.00},
    "debt_to_ebitda": {"low": 2.0, "high": 6.0},
    "net_debt_to_ebitda": {"low": 1.5, "high": 5.0},
    "ebitda_interest_coverage": {"good": 4.0, "bad": 1.5},
}
```

These thresholds are used to convert raw ratios into bounded risk indicators.

## Professional assumption

The thresholds are not legal rules or rating-agency standards. They are practical credit-analysis breakpoints.

The transformation typically works like this:

- good value => lower risk score;
- bad value => higher risk score;
- values between good and bad => interpolated risk score.

## Threshold interpretation

### Leverage

| Ratio | Good/low | Bad/high |
|---|---:|---:|
| liabilities/assets | 0.45 | 1.00 |
| debt/assets | 0.25 | 0.85 |
| debt/EBITDA | 2.0x | 6.0x |
| net debt/EBITDA | 1.5x | 5.0x |

These reflect broad non-financial corporate credit assumptions.

### Liquidity

| Ratio | Good | Bad |
|---|---:|---:|
| cash/assets | 10% | 1% |
| current ratio | 2.0x | 0.75x |
| quick ratio | 1.0x | 0.25x |

### Profitability and cash flow

| Ratio | Good | Bad |
|---|---:|---:|
| net income/assets | 5% | -5% |
| CFO/assets | 8% | -3% |
| EBITDA margin | 20% | 0% |
| FCF/debt | 15% | -10% |

### Coverage

| Ratio | Good | Bad |
|---|---:|---:|
| interest coverage | 3.0x | 1.0x |
| EBITDA interest coverage | 4.0x | 1.5x |

## Fine-tuning guidance

### Conservative bank mode

Increase good thresholds and tighten bad thresholds:

```python
"interest_coverage": {"good": 4.00, "bad": 1.50}
"debt_to_ebitda": {"low": 1.5, "high": 5.0}
"net_debt_to_ebitda": {"low": 1.0, "high": 4.0}
```

### SME flexible mode

Use softer thresholds:

```python
"current_ratio": {"good": 1.50, "bad": 0.75}
"interest_coverage": {"good": 2.50, "bad": 1.00}
"debt_to_ebitda": {"low": 2.5, "high": 6.5}
```

### Regulated utility / infrastructure mode

Do not simply relax leverage thresholds. Instead, create a sector-specific model or add sector-context narrative. Infrastructure companies can carry higher leverage, but only with stable regulated cash flows, long maturities, and predictable tariff regimes.

---

# 14. Diagnostic and reporting columns

## 14.1 `RATIO_COLS`

This list controls ratio and diagnostic output in reports.

Current values include:

```python
"log_assets",
"liabilities_to_assets",
"debt_to_assets",
"debt_to_equity",
"equity_to_assets",
"cash_to_assets",
"net_income_to_assets",
"cfo_to_assets",
"revenue_to_assets",
"current_ratio",
"quick_ratio",
"interest_coverage",
"fcf_to_debt",
"operating_margin",
"gross_margin",
"cfo_to_liabilities",
"capex_to_revenue",
"total_debt",
"net_debt",
"ebitda",
"ebitda_margin",
"debt_to_ebitda",
"net_debt_to_ebitda",
"ebitda_interest_coverage",
"leverage_risk",
"liquidity_risk",
"earnings_risk",
"operating_cashflow_risk",
"debt_service_risk",
"debt_service_risk_legacy",
"structural_distress_risk",
"scorecard_credit_score",
```

## Professional purpose

These columns are not all clustering inputs. They are used to explain the model result.

The list combines:

- raw ratios;
- derived debt metrics;
- risk domain scores;
- scorecard output.

## Fine-tuning guidance

Professionals can add more columns if they are computed in `features.py`.

Useful additions may include:

```python
"receivables_to_revenue"
"inventory_to_revenue"
"working_capital_to_revenue"
"capex_to_cfo"
"gross_debt_to_cfo"
```

But do not add columns that are not created by the scoring pipeline.

---

## 14.2 `INTERPRET_FEATURES`

These are used for cluster profiling and interpretation.

They overlap with `RATIO_COLS` but are not necessarily identical.

Professional use:

```text
RATIO_COLS = what to report for one company
INTERPRET_FEATURES = what to profile by cluster
```

Fine-tuning should keep them aligned enough that reported ratios explain cluster behavior.

---

## 14.3 `SUMMARY_COLS`

Current base scoring summary columns:

```python
SUMMARY_COLS = [
    "company_name",
    "fiscal_year",
    "assigned_cluster",
    "cluster_label",
    "risk_rank",
    "cluster_affinity",
    "near_default_affinity",
    "distance_to_assigned_cluster",
    "scorecard_credit_score",
    "feature_coverage_pct",
    "warning_flags",
]
```

This is the compact output of a base scoring run.

## Professional meaning

| Column | Meaning |
|---|---|
| `assigned_cluster` | KMeans cluster assigned to the company |
| `cluster_label` | business label from cluster risk ranking |
| `risk_rank` | ordered cluster rank; lower is stronger |
| `cluster_affinity` | soft similarity to assigned cluster |
| `near_default_affinity` | soft similarity to distressed proxy cluster |
| `distance_to_assigned_cluster` | model-space distance to assigned cluster centroid |
| `scorecard_credit_score` | weighted risk score |
| `feature_coverage_pct` | share of model features available |
| `warning_flags` | mechanical warning flags from scoring |

Professional warning: `cluster_affinity` and `near_default_affinity` are not probabilities.

---

## 14.4 `SUMMARY_COLS_WITH_OUTLOOK`

This expands `SUMMARY_COLS` with adjacent-bucket and outlook diagnostics, then appends `GUARDRAIL_COLS`.

Current additional fields include:

```python
"upper_bucket_cluster",
"upper_bucket_label",
"distance_to_upper_bucket",
"lower_bucket_cluster",
"lower_bucket_label",
"distance_to_lower_bucket",
"outlook_flag",
"outlook_reason",
*GUARDRAIL_COLS,
```

## Professional assumption

The outlook is a **cluster-position diagnostic**, not a forecast.

For example:

```text
Positive = closer to a stronger adjacent bucket
Negative = closer to a weaker adjacent bucket
Neutral = not near a boundary
```

It should not be presented as management guidance or rating outlook.

## Guardrail integration

Appending `*GUARDRAIL_COLS` ensures the consulting interpretation layer flows into:

- Notebook displays;
- Excel reports;
- PDF reports;
- CSV exports.

---

## 14.5 `SCENARIO_SUMMARY_COLS`

This is the scenario version of the summary output.

It includes:

```python
"scenario",
...
*GUARDRAIL_COLS
```

## Professional assumption

Scenarios should be interpreted as mechanical sensitivities, not forecasts.

Scenario guardrails are useful because a scenario may migrate into a weaker cluster or trigger structural distress even when the base case looks acceptable.

---

# 15. Recommended tuning workflow for professionals

Professionals cloning the project should follow this workflow:

## Step 1 — Define intended use

Choose one:

```text
SME credit diagnostic
lender screening tool
supplier risk screening
transaction due diligence support
portfolio monitoring
academic demonstration
```

The intended use determines how conservative the settings should be.

## Step 2 — Decide whether labels are client-facing

If the report will be shown to clients, avoid rating-agency language.

Use:

```text
Strong relative credit profile
```

not:

```text
Investment grade
```

## Step 3 — Tune denominator thresholds

For SMEs, keep `SME_MIN_DENOMINATOR` low enough to avoid suppressing valid ratios.

For public companies, use stronger materiality filters.

## Step 4 — Tune risk thresholds

Modify `RISK_THRESHOLDS` only after reviewing score distributions and examples.

## Step 5 — Tune guardrails

Guardrails should be stricter than the cluster model. Their job is to prevent over-optimistic interpretation.

## Step 6 — Retrain and inspect clusters

After changing feature weights, thresholds, or clustering settings, rerun Notebook 02.

Inspect:

- cluster sizes;
- cluster medians;
- cluster labels;
- feature availability;
- representative companies;
- extreme cases.

## Step 7 — Test manual scoring

Rerun Notebook 03 on:

- a strong company;
- a weak company;
- a leveraged but cash-generative company;
- a distressed company;
- a company with missing data.

## Step 8 — Review PDF narrative

Make sure the PDF does not overclaim.

---

# 16. Common customization scenarios

## 16.1 SME advisory mode

Recommended changes:

```python
SME_MIN_DENOMINATOR = 1_000
DEFAULT_SCORING_MIN_DENOMINATOR = SME_MIN_DENOMINATOR

DEFAULT_RATING_STYLE_LABELS = {
    1: "1 - Strong SME financial profile",
    2: "2 - Sound SME financial profile",
    3: "3 - Elevated SME financial risk",
    4: "4 - Weak SME financial profile",
    5: "5 - Distress-like SME profile",
}
```

Keep guardrails conservative.

## 16.2 Bank-style conservative mode

Recommended changes:

```python
"weak_interest_coverage": threshold 2.5
"elevated_debt_to_ebitda": threshold 3.0
"elevated_net_debt_to_ebitda": threshold 2.5
"weak_fcf_to_debt": threshold 0.15
```

Increase the importance of debt service in domain weights.

## 16.3 Infrastructure / utility mode

Do not just relabel the model.

Recommended approach:

- keep base model;
- use guardrails;
- add sector narrative;
- consider training a sector-specific model;
- consider adding regulatory cash-flow indicators.

Possible changes:

```python
"elevated_debt_to_ebitda": threshold 4.0
"elevated_net_debt_to_ebitda": threshold 3.5
```

But this should be paired with stronger requirements for cash-flow stability.

## 16.4 Distress screening mode

Recommended changes:

```python
SCORECARD_DOMAIN_WEIGHTS = {
    "leverage_risk": 0.25,
    "liquidity_risk": 0.20,
    "earnings_risk": 0.10,
    "operating_cashflow_risk": 0.20,
    "debt_service_risk": 0.15,
    "structural_distress_risk": 0.10,
}
```

Keep structural guardrails as `Override required`.

---

# 17. Quality-control checklist

Before using the project professionally, verify:

## Input mapping

- `operating_income` is EBIT, not EBT.
- `interest_expense` is interest/finance cost, consistently defined.
- `ebitda` is direct EBITDA or correctly derived.
- `depreciation_amortization` is included if EBITDA must be derived.
- `debt` includes interest-bearing debt, not all liabilities.
- `cash` is unrestricted cash unless otherwise disclosed.
- `capex` is capital expenditures, not total investing cash flow unless adjusted.

## Model consistency

- KMeans labels are reviewed after each retraining.
- `near_default_cluster` is inferred from risk rank, not hardcoded.
- Cluster labels do not imply formal credit ratings.
- Feature coverage is inspected.

## Report consistency

- Excel and PDF show the same guardrail conclusions.
- Guardrail flags are explained.
- Scenario outputs are labelled as sensitivities, not forecasts.
- Disclaimers remain in the PDF.

## Commercial defensibility

- Do not call the output a credit rating.
- Do not present affinity as probability of default.
- Do not present cluster assignment as lending approval.
- Use the model as a structured diagnostic and benchmarking tool.

---

# 18. Suggested future improvements

Professionals may extend `config.py` over time with:

## Sector-specific profiles

```python
SECTOR_RISK_THRESHOLDS = {
    "Utilities": {...},
    "Manufacturing": {...},
    "Software": {...},
}
```

## Country-specific SME thresholds

```python
COUNTRY_SME_DENOMINATORS = {
    "BG": 1_000,
    "DE": 10_000,
    "US": 10_000,
}
```

## External-rating reconciliation

```python
EXTERNAL_RATING_GUARDRAILS = {
    ...
}
```

## Report style configuration

```python
PDF_BRAND_NAME = "KSB Analytica"
PDF_REPORT_TITLE = "Credit Risk Diagnostic Report"
```

## Model version registry

```python
MODEL_VERSION_NOTES = {
    "v3_scorecard_ebitda": "Adds EBITDA-based leverage and coverage features."
}
```

---

# 19. Summary of current default philosophy

The current `config.py` is designed around this professional philosophy:

```text
Use unsupervised clustering as a relative credit-risk benchmarking tool.
Use scorecard weights to create a transparent risk score.
Use guardrails to prevent over-optimistic interpretation.
Use consulting-style labels, not rating-agency claims.
Use reports to support analyst judgment, not replace it.
```

The defaults are intentionally balanced:

- leverage is important but not the only driver;
- liquidity and operating cash flow receive strong weight;
- structural distress is present but not over-weighted in the score;
- guardrails are stricter than the model labels;
- scoring is SME-compatible;
- clustering remains non-financial-company focused;
- reports are built for professional explanation and customization.

This is the correct posture for a consulting-grade credit diagnostic prototype.
