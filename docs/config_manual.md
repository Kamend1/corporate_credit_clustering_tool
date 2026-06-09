# Configuration Manual for `src/credit_clustering/config.py`

## Purpose and status of this document

This is the **canonical configuration manual** for the Corporate Credit Clustering Tool.

It replaces the two overlapping configuration guides previously maintained as:

- `user_guide_config.md`
- `config_instruction_manual.md`

The project should keep **one authoritative configuration document** going forward. This file combines the practical quick-reference value of the shorter guide with the professional tuning logic of the longer manual.

The `config.py` module is intentionally logic-free. It centralizes assumptions, defaults, thresholds, labels, reporting columns, and model metadata. The calculations are implemented elsewhere in the project.

| Configuration area | Main consumer modules / notebooks |
|---|---|
| Model feature lists | `features.py`, `clustering.py`, `scoring.py`, `artifacts.py`, Notebook 02, Notebook 03 |
| Risk thresholds | `features.py` |
| Scorecard weights | `features.py`, `profiling.py`, reports |
| Guardrail rules | `guardrails.py`, `credit_report_util.py`, `credit_pdf_report_util.py` |
| Reporting columns | Notebook 03, Excel output, PDF output |
| Artifact defaults | `artifacts.py`, Notebook 02, Notebook 03 |
| Scoring defaults | `scoring.py`, Notebook 03 |
| Label definitions | `profiling.py`, `scoring.py`, reports |

The central design principle is separation of concerns:

```text
config.py defines assumptions.
features.py, clustering.py, scoring.py, diagnostics.py, guardrails.py and reporting utilities perform calculations.
notebooks orchestrate the workflow and explain the project.
```

For the SoftUni final project, this structure demonstrates that assumptions, model logic, and explanatory notebooks are deliberately separated.

---

## 1. What can be changed safely?

Not every setting has the same impact. Some changes only affect reporting; others require full model retraining.

| Change | Requires retraining? | Why |
|---|---:|---|
| `SCORECARD_CLUSTER_FEATURES` | Yes | KMeans centroids are fitted in exactly this feature space. |
| `RISK_THRESHOLDS` | Yes | Raw ratios are transformed into different risk features. |
| Domain/sub-component weights used to build model features | Yes if they affect model inputs | Feature values and cluster structure change. |
| `DEFAULT_N_CLUSTERS` | Yes | The number of centroids changes. |
| `DEFAULT_N_INIT` / `DEFAULT_RANDOM_STATE` | Yes, if producing a new artifact | These affect fitted centroids. |
| `DEFAULT_RATING_STYLE_LABELS` | No | Reporting label text only. |
| Guardrail thresholds | No, but rerun scoring/reporting | Guardrails are post-model diagnostics. |
| `SUMMARY_COLS` / report columns | No | Output formatting only. |
| `DEFAULT_SCORING_TEMPERATURE` | No, but comparability changes | It changes affinity sharpness, not cluster assignment. |
| `DEFAULT_FX_TO_MODEL_CURRENCY` | No, but rerun scoring | It changes monetary normalization for a scored company. |

Professional rule: if a setting changes the numerical values of the six model input features, retrain the model and regenerate the artifact.

---

## 2. Core model feature set

### `SCORECARD_CLUSTER_FEATURES`

Current recommended value:

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

These six features are the **direct KMeans inputs**. Each company-year becomes a vector:

```text
x_i = [structural_distress, earnings, operating_cashflow, liquidity, leverage, debt_service]
```

The intended range of each domain feature is approximately `[0, 1]`, where:

```text
0.00 = stronger / lower-risk characteristic
1.00 = weaker / higher-risk characteristic
```

### Credit interpretation

| Feature | Credit meaning |
|---|---|
| `structural_distress_risk` | Hard balance-sheet distress: negative equity or liabilities above assets. |
| `earnings_risk` | Weak or negative profitability relative to asset base. |
| `operating_cashflow_risk` | Weak or negative operating cash generation. |
| `liquidity_risk` | Insufficient current, quick, or cash liquidity. |
| `leverage_risk` | Excessive liabilities, debt, or weak equity buffer. |
| `debt_service_risk` | Weak ability to cover interest and debt burden using earnings / cash flow. |

### Why this feature design is defensible

The feature set is compact, financial, explainable, and appropriate for non-financial companies. It avoids throwing dozens of noisy ratios into KMeans and instead builds a low-dimensional credit-risk space that a human analyst can understand.

This is important because KMeans uses Euclidean distance. If many raw ratios with different scales were used directly, the model could be dominated by scale artifacts or sparse variables. The scorecard transformation makes the feature space more coherent.

### Tuning warning

Do not add or remove features from this list unless you also:

1. create the feature in `features.py`;
2. check feature coverage;
3. retrain Notebook 02;
4. regenerate the artifact;
5. rerun cluster profiling;
6. update methodology and interpretation documents.

---

## 3. Component risk features

### `SCORECARD_COMPONENT_FEATURES`

The component features explain how domain-level risks are built. They are diagnostic and report-facing, not direct KMeans inputs.

Typical current components:

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

Professional use:

| Domain risk | Typical component drivers |
|---|---|
| `liquidity_risk` | `current_liquidity_risk`, `quick_liquidity_risk`, `cash_buffer_risk` |
| `leverage_risk` | `liabilities_risk`, `debt_load_risk`, `equity_buffer_risk` |
| `debt_service_risk` | `coverage_risk`, `fcf_risk`, `debt_to_ebitda_risk`, `net_debt_to_ebitda_risk`, `ebitda_coverage_risk` |
| `structural_distress_risk` | `negative_equity_flag`, `liabilities_exceed_assets_flag` |

These component features make the model explainable. A user should be able to trace a poor risk label back to specific financial weaknesses.

---

## 4. Scorecard domain weights

### `SCORECARD_DOMAIN_WEIGHTS`

Recommended default:

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

These weights are used to build `scorecard_credit_score`, a post-hoc continuous credit-risk index on `[0, 100]`.

Important: this score is used for **cluster ranking and interpretation**, not as a KMeans input if the model uses the six domain features directly.

### Professional assumptions behind the default weights

| Domain | Weight | Rationale |
|---|---:|---|
| Leverage | 25% | Capital structure is a persistent driver of default and refinancing risk. |
| Liquidity | 20% | Liquidity stress often turns financial weakness into immediate distress. |
| Operating cash flow | 20% | Cash generation is harder to manipulate than accounting earnings. |
| Earnings | 15% | Profitability matters, but net income can be noisy. |
| Debt service | 15% | Coverage is highly relevant, but source data availability can be weaker. |
| Structural distress | 5% | Powerful but crude; better handled as both feature and guardrail. |

### Tuning examples

Conservative lender view:

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

SME working-capital view:

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

Validation checklist after changing weights:

1. Confirm weights sum to 1.00.
2. Recompute score distributions.
3. Check cluster median scores.
4. Confirm risk rank ordering is intuitive.
5. Inspect representative companies.
6. Rerun Notebook 02 and Notebook 03.

---

## 5. Risk transformation thresholds

### `RISK_THRESHOLDS`

These thresholds convert raw financial ratios into bounded risk components.

Recommended default structure:

```python
RISK_THRESHOLDS = {
    # --- Leverage ---
    "liabilities_to_assets":    {"low": 0.30,  "high": 0.95},
    "debt_to_assets":           {"low": 0.10,  "high": 0.75},
    "equity_to_assets":         {"good": 0.65, "bad": 0.00},

    # --- Liquidity ---
    "cash_to_assets":           {"good": 0.15, "bad": 0.005},
    "current_ratio":            {"good": 2.50, "bad": 0.50},
    "quick_ratio":              {"good": 1.50, "bad": 0.25},

    # --- Profitability / cash generation ---
    "net_income_to_assets":     {"good": 0.10, "bad": -0.05},
    "cfo_to_assets":            {"good": 0.20, "bad": -0.05},
    "cfo_to_debt":              {"good": 0.60, "bad": 0.02},

    # --- Debt service ---
    "interest_coverage":        {"good": 12.0, "bad": 0.80},
    "fcf_to_debt":              {"good": 0.40, "bad": -0.10},
    "debt_repayment_capacity":  {"good": 0.40, "bad": -0.08},

    # --- EBITDA-based ---
    "ebitda_margin":            {"good": 0.35, "bad": -0.05},
    "debt_to_ebitda":           {"low": 1.0,   "high": 6.0},
    "net_debt_to_ebitda":       {"low": 1.0,   "high": 5.0},
    "ebitda_interest_coverage": {"good": 20.0, "bad": 1.00},
}
```

### Mathematical logic

For ratios where higher values are worse:

```text
risk = clip((x - low) / (high - low), 0, 1)
```

For ratios where lower values are worse:

```text
risk = clip((good - x) / (good - bad), 0, 1)
```

The thresholds are not external credit-rating rules. They are practical credit-analysis breakpoints used to structure an interpretable feature space.

The current calibration is designed to preserve useful gradient while remaining economically interpretable for non-financial corporate credit analysis. Profitability and cash-flow thresholds are anchored in realistic corporate ranges, while interest-coverage and EBITDA-interest-coverage thresholds are intentionally conservative: they distinguish companies with excellent debt-service capacity from companies that are merely able to cover interest.

### Tuning guidance

| Use case | Possible adjustment |
|---|---|
| Conservative bank screening | Increase coverage requirements and lower leverage tolerance. |
| SME advisory | Slightly softer liquidity and coverage thresholds may be acceptable. |
| Distress screening | Keep hard structural thresholds strict. |
| Infrastructure / utilities | Do not simply relax leverage; consider a sector-specific model. |

Changing these thresholds requires retraining because the six KMeans features will change.

---

## 6. Denominator and materiality thresholds

### `SME_MIN_DENOMINATOR`

Recommended default:

```python
SME_MIN_DENOMINATOR = 1_000
```

This protects ratios from tiny denominators while still allowing SME-scale companies to be scored. A threshold that is too high will suppress valid small-company ratios, especially interest coverage and debt-service metrics.

| Setting | Use case |
|---:|---|
| `100` | Microbusiness experimentation |
| `1_000` | Current SME-compatible default |
| `10_000` | Conservative professional setting |
| `100_000` | Larger corporate-only setting |

### `PUBLIC_COMPANY_MIN_ASSETS`

Recommended default:

```python
PUBLIC_COMPANY_MIN_ASSETS = 1_000_000
```

This filters the public-company training universe to reduce shell companies, inactive microcaps, incomplete data, and extreme noise.

Do not treat a company below this threshold as unscoreable, but do flag it as outside the normal model calibration range.

### Size bands

Recommended diagnostic thresholds:

```python
SMALL_COMPANY_ASSET_LIMIT = 5_000_000
MID_COMPANY_ASSET_LIMIT = 50_000_000
LARGE_COMPANY_ASSET_LIMIT = 100_000_000_000
```

These are for reporting only. They are not clustering features.

---

## 7. Clustering defaults

Recommended defaults:

```python
DEFAULT_SEGMENT_COL = "financial_flag"
DEFAULT_TARGET_SEGMENTS = ("Non-financial",)
DEFAULT_PRIMARY_SEGMENT = "Non-financial"
DEFAULT_N_CLUSTERS = 5
DEFAULT_MIN_ROWS_PER_SEGMENT = 500
DEFAULT_MIN_FEATURES = 4
DEFAULT_ROW_FEATURE_COVERAGE = 0.60
DEFAULT_MIN_FEATURE_COVERAGE = 0.0
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_INIT = 500
```

### Why only non-financial companies?

Banks, insurers, and financial institutions have structurally different balance sheets. Leverage, liquidity, EBITDA, and working-capital ratios mean different things for them. The current model is intentionally limited to non-financial companies.

### Why five clusters?

Five clusters provide enough granularity for credit interpretation without pretending to replicate agency rating notches:

1. strong relative profile;
2. good profile;
3. leveraged / elevated risk;
4. weak profile;
5. distressed / near-default proxy.

### Why `n_init = 500`?

KMeans can converge to local optima. A high number of initializations improves centroid stability. This is slower than library defaults, but defensible for a final model artifact and an academic project.

---

## 8. Label definitions

Recommended labels:

```python
DEFAULT_RATING_STYLE_LABELS = {
    1: "1 - Strong relative credit profile",
    2: "2 - Good credit profile",
    3: "3 - Leveraged / elevated risk profile",
    4: "4 - Weak credit profile",
    5: "5 - Distressed / near-default proxy",
}
```

Avoid rating-agency language such as:

```text
investment grade
BB equivalent
rating
probability of default
```

The labels are model-relative. They are not formal credit ratings.

Critical principle:

```text
Raw KMeans cluster IDs are arbitrary.
Risk rank and label are post-hoc interpretations derived from cluster profiles.
```

---

## 9. Scoring defaults

Recommended defaults:

```python
DEFAULT_SCORING_SEGMENT = "Non-financial"
DEFAULT_SCORING_TEMPERATURE = 1.0
DEFAULT_FX_TO_MODEL_CURRENCY = 1.0
DEFAULT_SCORING_MIN_DENOMINATOR = SME_MIN_DENOMINATOR
```

### Temperature

The temperature parameter controls soft cluster affinity:

| Temperature | Effect |
|---:|---|
| `0.3` | Sharp assignment; affinity concentrated on nearest cluster. |
| `1.0` | Balanced default. |
| `3.0` | Smooth assignment; affinities spread across clusters. |

Affinity is not probability of default. It is a distance-based similarity score.

### FX conversion

All monetary inputs should be in a consistent currency and unit before scoring. If scoring non-USD statements, apply `fx_to_model_currency` consistently.

Ratios are mostly scale-independent, but `log_assets`, size bands, and model-scope warnings are not.

---

## 10. Input columns and mapping discipline

Expected financial input columns:

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

Professional mapping notes:

| Field | Recommended meaning |
|---|---|
| `assets` | Total assets |
| `liabilities` | Total liabilities |
| `equity` | Total equity |
| `cash` | Cash and cash equivalents |
| `net_income` | Profit after tax |
| `cfo` | Operating cash flow |
| `revenue` | Sales / revenue |
| `long_term_debt` | Long-term interest-bearing debt and leases |
| `short_term_debt` | Short-term borrowings and current portion of long-term debt |
| `current_assets` / `assets_current` | Current assets |
| `current_liabilities` / `liabilities_current` | Current liabilities |
| `receivables` | Trade and operating receivables |
| `inventory` | Inventories |
| `capex` | Capital expenditures |
| `operating_income` | EBIT / operating profit before finance costs |
| `interest_expense` | Interest expense / finance cost |
| `depreciation_amortization` | Depreciation and amortization |
| `ebitda` | EBITDA, if directly available |

Important warning:

```text
operating_income should normally mean EBIT, not EBT.
```

If profit before tax is mapped into `operating_income`, interest coverage and EBITDA reconstruction can be distorted.

---

## 11. Guardrail configuration

Guardrails are post-model analyst diagnostics. They do not change the KMeans assignment, but they qualify interpretation.

Recommended severity scale:

```python
GUARDRAIL_SEVERITY_ORDER = {
    "Clear": 0,
    "Monitor": 1,
    "Caution": 2,
    "High caution": 3,
    "Override required": 4,
}
```

Recommended direct rule families:

| Family | Examples |
|---|---|
| Leverage | high debt/assets, high debt/EBITDA, high net debt/EBITDA |
| Debt service | weak interest coverage, weak EBITDA coverage, weak FCF/debt |
| Liquidity | current ratio below 1.0, quick ratio below 0.5, low cash/assets |
| Structural distress | negative equity, liabilities exceed assets |

The model output should be presented as:

```text
relative financial-risk benchmarking
```

not as:

```text
formal credit rating
lending approval
probability of default
investment recommendation
```

Guardrails make this professional caution visible in Excel and PDF reporting.

---

## 12. Reporting columns

The reporting column lists should distinguish three purposes:

| List | Purpose |
|---|---|
| `RATIO_COLS` | Company-level ratio and diagnostic output. |
| `INTERPRET_FEATURES` | Cluster profile and median comparison. |
| `SUMMARY_COLS` / `SUMMARY_COLS_WITH_OUTLOOK` | Compact scoring tables and report summaries. |

Recommended summary fields include:

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

For client-facing or student-facing communication, do not overemphasize `assigned_cluster`. It is an unordered technical ID.

---

## 13. Artifact defaults

Recommended artifact metadata:

```python
DEFAULT_ARTIFACT_VERSION = "v3_scorecard_ebitda"
DEFAULT_PRIMARY_SEGMENT = "Non-financial"
```

Update the artifact version when a change affects model behavior or interpretation.

Examples:

```python
"v4_scorecard_guardrails"
"v4_sector_adjusted"
"v5_supervised_overlay"
```

The saved artifact should contain enough metadata for Notebook 03 to score a company without re-running Notebook 02.

Minimum expected artifact contents:

```text
pipeline
feature_cols
cluster_labels
risk_rank
cluster_profile_ranked
scorecard_domain_weights
winsor_caps
artifact_version
training metadata
```

---

## 14. Recommended professional tuning workflow

1. Define intended use: academic demo, SME diagnostic, lender screening, supplier risk monitoring, or portfolio surveillance.
2. Decide whether labels are external-facing.
3. Avoid rating-agency terminology unless externally calibrated.
4. Tune denominator thresholds only if company scale requires it.
5. Tune risk thresholds only with re-training and profile review.
6. Keep guardrails conservative.
7. Retrain the KMeans artifact.
8. Inspect cluster sizes, profiles, monotonicity, representative companies, and outliers.
9. Test private-company scoring on several constructed examples.
10. Review Excel/PDF output for overclaiming.

---

## 15. Quality-control checklist

Before final submission or professional use, verify:

### Input mapping

- `assets` and `liabilities` are total balance-sheet values.
- `equity` reconciles broadly with assets minus liabilities.
- `operating_income` is EBIT, not EBT.
- `interest_expense` is finance cost / interest expense.
- `debt` fields are interest-bearing debt, not all liabilities.
- `cash` is unrestricted cash unless disclosed otherwise.
- `capex` is capital expenditure, not total investing cash flow without adjustment.

### Model consistency

- KMeans labels are reviewed after every retraining.
- Near-default cluster is inferred from risk rank, not hardcoded.
- Cluster IDs are not interpreted directly.
- Feature coverage is inspected.
- Cluster profile medians show plausible financial progression.

### Report consistency

- Excel and PDF show the same labels and guardrail conclusions.
- Scenario outputs are clearly marked as sensitivities, not forecasts.
- Affinities are not described as probabilities.
- Limitations are visible.

---

## 16. Suggested future configuration extensions

Possible future additions:

```python
SECTOR_RISK_THRESHOLDS = {
    "Utilities": {...},
    "Manufacturing": {...},
    "Software": {...},
}
```

```python
COUNTRY_SME_DENOMINATORS = {
    "BG": 1_000,
    "DE": 10_000,
    "US": 10_000,
}
```

```python
MODEL_VERSION_NOTES = {
    "v3_scorecard_ebitda": "Adds EBITDA-based leverage and coverage features."
}
```

```python
PDF_BRAND_NAME = "KSB Analytica"
PDF_REPORT_TITLE = "Credit Risk Diagnostic Report"
```

---

## 17. Summary philosophy

The current configuration philosophy is:

```text
Use unsupervised clustering as relative credit-risk benchmarking.
Use domain-guided features to make KMeans financially interpretable.
Use scorecard weights to rank clusters transparently.
Use guardrails to prevent over-optimistic interpretation.
Use reports to support analyst judgment, not replace it.
```

This is the correct posture for a final ML course project and for a realistic consulting-grade prototype.
