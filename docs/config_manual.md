# Configuration Manual

This document explains the current `src/credit_clustering/config.py` design for the Corporate Credit Clustering Tool.

The configuration module is intentionally logic-free. It centralizes feature lists, scorecard weights, thresholds, clustering defaults, guardrails, scoring defaults, artifact metadata, and reporting columns used by Notebook 02, Notebook 03, and the reusable source modules.

---

## 1. Core model feature set

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

These six domain-level features are the direct KMeans inputs. They are designed to be numeric, directionally consistent, interpretable, and broadly bounded between 0 and 1.

```text
0.00 = stronger / lower-risk characteristic
1.00 = weaker / higher-risk characteristic
```

The model clusters companies on credit-risk concepts, not directly on dozens of raw accounting ratios.

---

## 2. Component risk features

The lower-level component risks explain how the six domain features are constructed.

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
    "cfo_to_debt_risk",
    "coverage_risk",
    "fcf_risk",
    "debt_repayment_risk",
    "ebitda_margin_risk",
    "debt_to_ebitda_risk",
    "net_debt_to_ebitda_risk",
    "ebitda_coverage_risk",
    "negative_ebitda_flag",
    "negative_equity_flag",
    "liabilities_exceed_assets_flag",
]
```

| Component | Meaning |
|---|---|
| `liabilities_risk` | risk from high liabilities/assets |
| `debt_load_risk` | risk from high debt/assets |
| `equity_buffer_risk` | risk from weak equity/assets |
| `cash_buffer_risk` | risk from weak cash/assets |
| `current_liquidity_risk` | risk from weak current ratio |
| `quick_liquidity_risk` | risk from weak quick ratio |
| `profitability_risk` | risk from weak net income/assets |
| `cashflow_risk` | risk from weak CFO/assets |
| `cfo_to_debt_risk` | risk from weak CFO relative to total debt |
| `coverage_risk` | risk from weak EBIT/interest coverage |
| `fcf_risk` | risk from weak FCF/debt |
| `debt_repayment_risk` | risk from weak free-cash-flow debt repayment capacity |
| `ebitda_margin_risk` | risk from weak EBITDA margin |
| `debt_to_ebitda_risk` | risk from high debt/EBITDA |
| `net_debt_to_ebitda_risk` | risk from high net debt/EBITDA |
| `ebitda_coverage_risk` | risk from weak EBITDA/interest coverage |
| `negative_ebitda_flag` | hard diagnostic flag for non-positive EBITDA |
| `negative_equity_flag` | hard diagnostic flag for negative equity |
| `liabilities_exceed_assets_flag` | hard diagnostic flag for liabilities above assets |

---

## 3. Domain feature formulas

### `leverage_risk`

```text
leverage_risk =
    0.30 × liabilities_risk
  + 0.25 × debt_load_risk
  + 0.20 × equity_buffer_risk
  + 0.25 × net_debt_to_ebitda_risk
```

This captures balance-sheet leverage, equity cushion, and net debt burden relative to EBITDA.

### `liquidity_risk`

```text
liquidity_risk =
    0.35 × current_liquidity_risk
  + 0.30 × quick_liquidity_risk
  + 0.20 × debt_repayment_risk
  + 0.15 × cash_buffer_risk
```

Liquidity risk includes short-term liquidity and internal debt repayment capacity from free cash flow.

### `earnings_risk`

```text
earnings_risk = profitability_risk
```

### `operating_cashflow_risk`

```text
operating_cashflow_risk =
    0.50 × cashflow_risk
  + 0.50 × cfo_to_debt_risk
```

Operating cash flow is evaluated relative to both assets and total debt.

### `debt_service_risk`

```text
debt_service_risk =
    0.35 × coverage_risk
  + 0.25 × fcf_risk
  + 0.25 × debt_to_ebitda_risk
  + 0.15 × ebitda_coverage_risk
```

Fallback when EBITDA diagnostics are unavailable:

```text
debt_service_risk_legacy =
    0.60 × coverage_risk
  + 0.40 × fcf_risk
```

### `structural_distress_risk`

```text
structural_distress_risk =
    0.60 × equity_buffer_risk
  + 0.40 × liabilities_risk
```

This is a gradient balance-sheet vulnerability score. It is not a legal insolvency conclusion. Hard structural flags remain separate: `negative_equity_flag` and `liabilities_exceed_assets_flag`.

---

## 4. Scorecard domain weights

```python
SCORECARD_DOMAIN_WEIGHTS = {
    "leverage_risk": 0.25,
    "liquidity_risk": 0.10,
    "earnings_risk": 0.15,
    "operating_cashflow_risk": 0.20,
    "debt_service_risk": 0.25,
    "structural_distress_risk": 0.05,
}
```

These weights calculate `scorecard_credit_score` and support cluster ranking and report interpretation. They do not directly change KMeans assignments unless explicitly used as feature weights in the clustering feature space.

| Domain | Weight | Rationale |
|---|---:|---|
| Leverage | 25% | capital structure and net debt burden are core credit drivers |
| Debt service | 25% | coverage and repayment capacity are direct creditor signals |
| Operating cash flow | 20% | cash generation is central to credit quality |
| Earnings | 15% | profitability matters but can be accounting-noisy |
| Liquidity | 10% | short-term liquidity matters but is not the dominant long-term credit driver |
| Structural vulnerability | 5% | hard distress is handled separately by guardrails |

---

## 5. Risk thresholds

Risk scores are calculated using two transformation types.

For bad-high metrics:

```text
risk = clip((x - low) / (high - low), 0, 1)
```

For bad-low metrics:

```text
risk = clip((good - x) / (good - bad), 0, 1)
```

Current thresholds:

```python
RISK_THRESHOLDS = {
    "liabilities_to_assets": {"low": 0.15, "high": 2.00},
    "debt_to_assets": {"low": 0.15, "high": 0.95},
    "equity_to_assets": {"good": 0.80, "bad": 0.00},

    "cash_to_assets": {"good": 0.15, "bad": 0.005},
    "current_ratio": {"good": 2.50, "bad": 0.50},
    "quick_ratio": {"good": 1.50, "bad": 0.25},

    "net_income_to_assets": {"good": 0.45, "bad": -0.45},
    "cfo_to_assets": {"good": 0.80, "bad": -0.30},
    "cfo_to_debt": {"good": 0.90, "bad": -0.35},

    "interest_coverage": {"good": 4.00, "bad": 0.75},
    "fcf_to_debt": {"good": 0.85, "bad": -0.85},
    "debt_repayment_capacity": {"good": 0.45, "bad": -0.45},

    "ebitda_margin": {"good": 0.35, "bad": -0.05},
    "debt_to_ebitda": {"low": 1.0, "high": 6.0},
    "net_debt_to_ebitda": {"low": 1.0, "high": 5.0},
    "ebitda_interest_coverage": {"good": 4.0, "bad": 0.5},
}
```

The broad threshold ranges preserve gradient and reduce excessive saturation at exact 0 or exact 1.

---

## 6. Guardrails

Guardrails are separate from KMeans. They identify analyst-level red flags such as high leverage, weak coverage, weak liquidity, negative equity, or liabilities exceeding assets.

Severity levels:

| Severity | Meaning |
|---|---|
| Clear | no material issue |
| Monitor | mild issue |
| Caution | meaningful weakness |
| High caution | serious weakness |
| Override required | hard structural issue requiring manual review |

Guardrails qualify interpretation; they do not become the KMeans label.

---

## 7. Clustering defaults

```python
DEFAULT_N_CLUSTERS = 5
DEFAULT_N_INIT = 500
DEFAULT_RANDOM_STATE = 42
DEFAULT_MIN_ROWS_PER_SEGMENT = 500
DEFAULT_MIN_FEATURES = 4
DEFAULT_ROW_FEATURE_COVERAGE = 0.60
DEFAULT_MIN_FEATURE_COVERAGE = 0.0
```

Five clusters are used because they create a practical 1–5 credit-risk scale without pretending to replicate rating-agency notches.

---

## 8. Artifact defaults

```python
DEFAULT_ARTIFACT_VERSION = "v3_scorecard_ebitda"
DEFAULT_PRIMARY_SEGMENT = "Non-financial"
```

The artifact version should remain synchronized across Notebook 02, Notebook 03, saved model filenames, README examples, and report metadata.

---

## 9. What requires retraining?

| Change | Retrain Notebook 02? | Reason |
|---|---:|---|
| `SCORECARD_CLUSTER_FEATURES` | Yes | KMeans centroids depend on feature dimensions |
| Domain formulas in `features.py` | Yes | Six KMeans inputs change |
| `RISK_THRESHOLDS` | Yes | Component and domain values change |
| `DEFAULT_N_CLUSTERS` | Yes | Cluster structure changes |
| `SCORECARD_DOMAIN_WEIGHTS` | Yes for full consistency | Scorecard ranking and interpretation change |
| Guardrail thresholds | No model retraining; rerun scoring/reporting | Post-model diagnostics change |
| Reporting columns | No | Output formatting only |
| Label text | No model retraining; rerun reports | Presentation only |

---

## 10. Summary

The current configuration defines a debt-capacity-aware KMeans credit clustering model. Leverage and debt service carry the strongest scorecard emphasis. Liquidity includes both short-term liquidity and FCF/debt repayment capacity. Operating cash flow is evaluated relative to assets and debt. Structural distress is a gradient balance-sheet vulnerability score, while hard distress flags remain guardrails.
