# Limitations

This document catalogs the known limitations of the Corporate Credit Clustering Tool. Understanding these boundaries is essential for appropriate use of the model outputs.

---

## 1. This is not a formal credit rating

The cluster labels (e.g. "3 — Elevated risk / leveraged") are **interpretive analogies**, not certified credit opinions. They have not been calibrated against agency rating transitions, validated against historical default events, or reviewed by a credit rating analyst. They must not be used as a substitute for formal due diligence, regulated credit assessment, or investment advice.

---

## 2. Survivorship bias

The training universe consists of companies that filed annual reports on SEC EDGAR between 2020 and 2025 and remained in the XBRL database throughout the scraping window. Companies that:

- defaulted and were delisted before the data pull;
- were acquired and ceased to file independently;
- were voluntarily dissolved or went private;

are **not represented** in the training data. This systematically under-populates the highest-risk end of the credit quality spectrum. The "Distressed / near-default proxy" cluster is, in practice, a cluster of financially stressed but still-operating companies — not a cluster of companies that actually defaulted.

**Practical implication:** The model may assign a company a lower risk rank than a model trained on a complete historical panel that includes eventual defaulters would produce. Stress scores should be read with this in mind.

---

## 3. Domain-guided, not purely unsupervised

The feature transformation pipeline in `features.py` encodes credit-domain knowledge through:

- **Monotone risk thresholds** in `RISK_THRESHOLDS` (e.g. `interest_coverage good = 3.0, bad = 1.0`) that determine how each ratio is mapped to a [0, 1] risk score;
- **Sub-component weights** within each domain feature (e.g. `leverage_risk = 0.40 × liabilities_risk + 0.35 × debt_load_risk + 0.25 × equity_buffer_risk`);
- **Domain weights** in `SCORECARD_DOMAIN_WEIGHTS` (leverage 25%, liquidity 20%, etc.).

These choices embed the analyst's prior about what constitutes credit risk **before** any clustering algorithm runs. KMeans then groups companies in this pre-structured space. The model is best described as **domain-guided clustering**, not pure unsupervised discovery. A different valid set of thresholds and weights would produce different clusters.

---

## 4. Fixed scorecard weights are unvalidated

The domain weights and sub-component weights are set by judgment and have not been:

- empirically optimised against a labelled dataset;
- validated through sensitivity analysis demonstrating label stability under weight perturbation;
- compared against a data-driven dimensionality reduction (e.g. PCA-derived weights).

See [Sensitivity Analysis](sensitivity_analysis.md) for guidance on how to test weight sensitivity.

---

## 5. US-listed, large-to-mid-cap bias

The training universe is constrained to SEC EDGAR filers, which are predominantly US-domiciled companies. The minimum assets filter (`PUBLIC_COMPANY_MIN_ASSETS = $1 000 000`) further skews the training population toward established, operationally stable companies. Specifically:

- Micro-cap companies (below ~$10M assets) are sparse in the training data.
- Non-US companies with US ADR listings are included but may follow different accounting conventions.
- Private companies, non-US companies, and companies in IFRS jurisdictions are **outside the training distribution**.

When scoring a non-US or private company using Notebook 03, the FX conversion flag (`fx_to_model_currency`) can normalise monetary units, but the underlying thresholds and cluster centroids remain calibrated to the US public-company population.

---

## 6. Point-in-time labels, not through-the-cycle

The model produces a **point-in-time** credit quality label for a single fiscal year's financial data. It is not a through-the-cycle rating that smooths over economic cycles.

The training panel spans 2020–2025, which includes:
- The COVID-19 shock (2020–2021) and subsequent distress spike;
- The recovery and interest rate normalisation period (2022–2024).

A company appearing in the training data in 2020 and again in 2023 may contribute to different clusters in different years. The model does not track migration paths; it only observes cross-sectional financial position at a given date.

---

## 7. Missing-value imputation may distort low-data rows

For rows with fewer available features, missing values are imputed to the segment median before clustering. This pulls low-coverage rows toward the centre of the feature distribution, regardless of their true risk profile. A company with almost no available data will appear at an average risk level rather than being flagged as unclassifiable.

The `feature_coverage_pct` output column quantifies this risk per row. Rows with `feature_coverage_pct < 0.67` (fewer than 4 of 6 model features available) should be interpreted with caution.

---

## 8. Financial companies are excluded

Companies with SIC codes classified as Financial / Insurance / Real Estate (`financial_flag = "Financial"`) are excluded from both model training and scoring. The risk thresholds and leverage-based domain features are not appropriate for banks, insurers, or REITs, where high leverage is structurally expected and EBITDA is not a meaningful concept.

Applying the model to a financial company will produce misleading results.

---

## 9. EBITDA availability varies

EBITDA is calculated as `operating_income + depreciation_amortization` when the direct EBITDA concept is not reported. If both `operating_income` and `depreciation_amortization` are missing, the EBITDA-dependent sub-components of `debt_service_risk` fall back to the legacy formula (`0.60 × coverage_risk + 0.40 × fcf_risk`), which excludes the EBITDA-based risk factors entirely. Companies with this fallback receive a slightly different effective feature weighting.

---

## 10. No external ground-truth validation

The cluster labels have not been validated against:

- known agency ratings for the same companies in the same fiscal years;
- subsequent default or distress events;
- market-implied credit signals (CDS spreads, bond yields).

The only validation in the project is internal: the cluster profiles are inspected for monotone progression of key ratios across the five risk tiers, and representative tickers are reviewed for plausibility. This is a qualitative sanity check, not a statistical validation.

---

*See also: [Methodology](methodology.md) | [Sensitivity Analysis](sensitivity_analysis.md)*
