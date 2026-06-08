# Scorer Report Methodology

This document explains the private-company scoring workflow and the interpretation of Excel/PDF report outputs.

---

## 1. Purpose

The scorer is designed to:

- score a manually entered or private company against the public-company benchmark;
- produce a model-relative risk label;
- show distance and affinity diagnostics;
- compare the company to cluster medians;
- run scenario analysis;
- highlight warning flags and guardrails;
- generate Excel and PDF report outputs.

The output is analytical, not a formal credit rating.

---

## 2. Scoring flow

```text
Raw company financials
        ↓
Column standardization and FX conversion
        ↓
Derived accounting values
        ↓
Ratios and EBITDA diagnostics
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

---

## 3. Input schema

Minimum practical inputs:

| Field | Meaning |
|---|---|
| `assets` | total assets |
| `liabilities` | total liabilities |
| `equity` | total equity |
| `revenue` | revenue |
| `net_income` | net income |
| `cfo` | operating cash flow |

Recommended inputs:

| Field | Meaning |
|---|---|
| `cash` | cash and equivalents |
| `current_assets` / `assets_current` | current assets |
| `current_liabilities` / `liabilities_current` | current liabilities |
| `receivables` | receivables for quick ratio |
| `inventory` | inventory |
| `long_term_debt` | long-term debt |
| `short_term_debt` | short-term debt |
| `interest_expense` | interest / finance cost |
| `operating_income` | EBIT / operating profit |
| `depreciation_amortization` | D&A |
| `ebitda` | direct EBITDA if available |
| `capex` | capital expenditure |
| `gross_profit` | gross profit |

Debt, CFO, FCF, EBITDA, interest expense, and capex are high-impact inputs because they influence several model dimensions.

---

## 4. Derived values

```text
total_debt = long_term_debt + short_term_debt
net_debt = total_debt - cash
fcf = CFO - |capex|
EBITDA = direct EBITDA or operating_income + depreciation_amortization
```

These values drive CFO/debt, FCF/debt, debt/EBITDA, net debt/EBITDA, and EBITDA interest coverage.

---

## 5. Domain feature drivers

| Domain | Main drivers |
|---|---|
| leverage risk | liabilities/assets, debt/assets, equity/assets, net debt/EBITDA |
| liquidity risk | current ratio, quick ratio, FCF/debt repayment capacity, cash/assets |
| earnings risk | net income/assets |
| operating cash-flow risk | CFO/assets and CFO/debt |
| debt-service risk | interest coverage, FCF/debt, debt/EBITDA, EBITDA coverage |
| structural distress risk | equity buffer and liabilities/assets |

---

## 6. Main scoring outputs

| Output | Meaning |
|---|---|
| `assigned_cluster` | nearest raw KMeans cluster ID |
| `risk_rank` | ordered rank from strongest to weakest |
| `cluster_label` | human-readable risk label |
| `distance_to_assigned_cluster` | distance to assigned centroid |
| `cluster_affinity` | distance-based similarity to assigned cluster |
| `near_default_affinity` | distance-based similarity to weakest cluster |
| `outlook_flag` | adjacent-bucket position signal |
| `warning_flags` | mechanical red flags |
| `guardrail_level` | analyst caution level |

---

## 7. Scenario analysis

Scenario analysis modifies selected inputs and reruns the same scoring process.

Common scenario types:

- base case;
- revenue decline;
- debt increase;
- cash burn;
- near-default stress.

Debt-up scenarios may affect multiple domains simultaneously because debt influences debt/assets, net debt/EBITDA, CFO/debt, FCF/debt, and debt-service diagnostics.

Scenarios are mechanical sensitivities, not forecasts.

---

## 8. Excel report

Recommended tabs:

| Tab | Purpose |
|---|---|
| Score summary | headline outputs |
| Input financials | raw/manual inputs |
| Ratio snapshot | derived ratios and diagnostics |
| Cluster comparison | company vs assigned-cluster median |
| Scenarios | scenario outputs |
| Guardrails | flags and analyst interpretation |
| Methodology notes | concise model explanation |

---

## 9. PDF report

Recommended sections:

| Section | Purpose |
|---|---|
| Executive summary | concise conclusion |
| Risk label scale | explains the 1–5 labels |
| Company scorecard | label, score, affinity, feature coverage |
| Key ratios | financial diagnostics |
| Cluster comparison | benchmark-relative position |
| Guardrails | caution items |
| Scenarios | sensitivity results |
| Limitations | prevents overclaiming |

The PDF should clearly state that the result is not a formal credit rating.

---

## 10. Input quality warnings

Incorrect mapping can materially affect outputs.

| Input issue | Potential effect |
|---|---|
| debt omitted | leverage and debt-service risk understated |
| CFO overstated | operating cash-flow and FCF/debt risk understated |
| capex omitted | FCF/debt overstated |
| EBITDA unavailable | EBITDA-based diagnostics weakened |
| interest expense missing | coverage risk unavailable |

---

## 11. Summary

The scorer report connects model output with credit-analysis interpretation. It should show the assigned label, the financial drivers, the distance/affinity diagnostics, the guardrails, the scenario sensitivity, and the limitations.
