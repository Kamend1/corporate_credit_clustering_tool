"""
Feature engineering for the credit clustering project.

This module is the single source of truth for converting raw accounting data
into financial ratios, EBITDA diagnostics, scorecard risk components, and the
six domain-level risk features used by the KMeans clustering model.
"""

import numpy as np
import pandas as pd

from .config import (
    LARGE_COMPANY_ASSET_LIMIT,
    MID_COMPANY_ASSET_LIMIT,
    MONETARY_COLUMNS,
    REQUIRED_OR_OPTIONAL_FINANCIAL_COLUMNS,
    RISK_THRESHOLDS,
    SCORECARD_DOMAIN_WEIGHTS,
    SMALL_COMPANY_ASSET_LIMIT,
    SME_MIN_DENOMINATOR,
)


def ensure_columns(df, columns, default=np.nan):
    """Return a copy of df with all requested columns present."""
    out = df.copy()

    for col in columns:
        if col not in out.columns:
            out[col] = default

    return out


# Backward-compatible private alias for existing code that may import it.
_ensure_columns = ensure_columns


def safe_divide(numerator, denominator, min_abs_denominator=None):
    """
    Division helper that returns NaN when denominator is zero, tiny, or missing.

    min_abs_denominator is optional. For private-company manual scoring,
    interest coverage should still be calculated when interest expense is
    economically valid but below a public-company materiality threshold.
    """
    numerator = pd.Series(numerator).astype(float)
    denominator = pd.Series(denominator).astype(float)

    denominator_safe = denominator.copy()
    denominator_safe = denominator_safe.mask(denominator_safe == 0)

    if min_abs_denominator is not None:
        denominator_safe = denominator_safe.mask(
            denominator_safe.abs() < min_abs_denominator
        )

    out = numerator / denominator_safe

    return out.replace([np.inf, -np.inf], np.nan)


def clip01(x):
    """Clip numeric values to the [0, 1] risk-score range."""
    return np.clip(x, 0, 1)


# Backward-compatible private alias.
_clip01 = clip01


def linear_risk_bad_high(x, low, high):
    """0 when x <= low, 1 when x >= high."""
    return clip01((x - low) / (high - low))


# Backward-compatible private alias.
_linear_risk_bad_high = linear_risk_bad_high


def linear_risk_bad_low(x, good, bad):
    """0 when x >= good, 1 when x <= bad."""
    return clip01((good - x) / (good - bad))


# Backward-compatible private alias.
_linear_risk_bad_low = linear_risk_bad_low


def _weighted_risk(out, weighted_components):
    """
    Combine bounded [0, 1] risk components into a domain-level risk feature,
    renormalizing over the weights of the components that are actually present
    on each row.

    weighted_components : list of (column_name, weight) tuples.

    Why this exists
    ---------------
    The domain features previously used raw weighted sums, e.g.

        leverage_risk = 0.30 * liabilities_risk + ... + 0.25 * net_debt_to_ebitda_risk

    Because NaN propagates through addition, a single missing component (for
    example net_debt_to_ebitda_risk for a negative-EBITDA issuer, or any
    debt-relative component for a debt-free issuer) collapsed the *entire*
    domain feature to NaN. The clustering pipeline then median-imputed that
    NaN, silently pulling distressed and debt-free issuers toward the universe
    average.

    This helper mirrors the missing-weight handling already used in
    ``_add_scorecard_credit_score``: present components are weighted and the
    result is divided by the weight that was actually available, so the feature
    is NaN only when *every* component is missing for a row. When all components
    are present the output is identical to the original weighted sum (the
    weights in each domain feature sum to 1.0), so this change is backward
    compatible for complete data.
    """
    weighted_sum = pd.Series(0.0, index=out.index)
    available_weight = pd.Series(0.0, index=out.index)

    for col, weight in weighted_components:
        if col in out.columns:
            valid = out[col].notna()
            weighted_sum = weighted_sum + out[col].fillna(0.0) * weight
            available_weight = available_weight + valid.astype(float) * weight

    # NaN only when no component was available for the row.
    return weighted_sum / available_weight.replace(0.0, np.nan)


def _coerce_and_convert_monetary_columns(out, fx_to_model_currency):
    for col in MONETARY_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Convert only absolute monetary values. Ratios are calculated afterward.
    for col in MONETARY_COLUMNS:
        out[col] = out[col] * fx_to_model_currency

    return out


def _standardize_current_column_names(out):
    # Accept both EDGAR-style and business-friendly names.
    if out["assets_current"].isna().all() and "current_assets" in out.columns:
        out["assets_current"] = out["current_assets"]

    if out["liabilities_current"].isna().all() and "current_liabilities" in out.columns:
        out["liabilities_current"] = out["current_liabilities"]

    return out


def _add_derived_accounting_values(out):
    debt_missing_mask = out[["long_term_debt", "short_term_debt"]].isna().all(axis=1)

    out["long_term_debt"] = out["long_term_debt"].fillna(0)
    out["short_term_debt"] = out["short_term_debt"].fillna(0)

    out["total_debt"] = out["long_term_debt"] + out["short_term_debt"]
    out.loc[debt_missing_mask, "total_debt"] = np.nan

    # Backward-compatible alias used by older Notebook 03 code.
    out["debt"] = out["total_debt"]

    out["capex"] = out["capex"].fillna(0)
    out["fcf"] = out["cfo"] - out["capex"].abs()

    # Backward-compatible alias used by older Notebook 03 code.
    out["free_cash_flow"] = out["fcf"]

    out["net_debt"] = out["total_debt"] - out["cash"]
    out.loc[
        out[["total_debt", "cash"]].isna().any(axis=1),
        "net_debt",
    ] = np.nan

    calculated_ebitda = out["operating_income"] + out["depreciation_amortization"]
    calculated_ebitda = calculated_ebitda.where(
        out[["operating_income", "depreciation_amortization"]].notna().all(axis=1)
    )

    # If direct EBITDA is supplied, keep it; otherwise calculate it strictly as
    # Operating Income + Depreciation & Amortization when both are available.
    out["ebitda"] = out["ebitda"].combine_first(calculated_ebitda)

    return out


def _add_base_ratios(out, min_denominator=SME_MIN_DENOMINATOR):
    out["log_assets"] = np.log1p(out["assets"].clip(lower=0))

    out["liabilities_to_assets"] = safe_divide(
        out["liabilities"],
        out["assets"],
        min_denominator,
    )

    out["equity_to_assets"] = safe_divide(
        out["equity"],
        out["assets"],
        min_denominator,
    )

    out["cash_to_assets"] = safe_divide(
        out["cash"],
        out["assets"],
        min_denominator,
    )

    out["revenue_to_assets"] = safe_divide(
        out["revenue"],
        out["assets"],
        min_denominator,
    )

    out["net_income_to_assets"] = safe_divide(
        out["net_income"],
        out["assets"],
        min_denominator,
    )

    out["cfo_to_assets"] = safe_divide(
        out["cfo"],
        out["assets"],
        min_denominator,
    )

    out["debt_to_assets"] = safe_divide(
        out["total_debt"],
        out["assets"],
        min_denominator,
    )

    out["debt_to_equity"] = safe_divide(
        out["total_debt"],
        out["equity"],
        min_denominator,
    )

    out["current_ratio"] = safe_divide(
        out["assets_current"],
        out["liabilities_current"],
        min_denominator,
    )

    out["quick_ratio"] = safe_divide(
        out["cash"].fillna(0) + out["receivables"].fillna(0),
        out["liabilities_current"],
        min_denominator,
    )

    out.loc[
        out[["cash", "receivables"]].isna().all(axis=1),
        "quick_ratio",
    ] = np.nan

    return out


def _add_diagnostic_ratios(out, min_denominator=SME_MIN_DENOMINATOR):
    out["gross_margin"] = safe_divide(
        out["gross_profit"],
        out["revenue"],
        min_denominator,
    )

    out["operating_margin"] = safe_divide(
        out["operating_income"],
        out["revenue"],
        min_denominator,
    )

    # No minimum denominator here. Private-company scoring must calculate valid
    # coverage ratios for smaller absolute interest-expense amounts.
    out["interest_coverage"] = safe_divide(
        out["operating_income"],
        out["interest_expense"].abs(),
    )

    out["cfo_to_liabilities"] = safe_divide(
        out["cfo"],
        out["liabilities"],
        min_denominator,
    )

    out["fcf_to_debt"] = safe_divide(
        out["fcf"],
        out["total_debt"],
        min_denominator,
    )

    out["capex_to_revenue"] = safe_divide(
        out["capex"].abs(),
        out["revenue"],
        min_denominator,
    )

    return out


def _add_ebitda_metrics(out, min_denominator=SME_MIN_DENOMINATOR):
    out["ebitda_margin"] = safe_divide(
        out["ebitda"],
        out["revenue"],
        min_denominator,
    )

    out["debt_to_ebitda"] = safe_divide(
        out["total_debt"],
        out["ebitda"],
        min_denominator,
    )

    out["net_debt_to_ebitda"] = safe_divide(
        out["net_debt"],
        out["ebitda"],
        min_denominator,
    )

    # No minimum denominator for coverage. See interest_coverage above.
    out["ebitda_interest_coverage"] = safe_divide(
        out["ebitda"],
        out["interest_expense"].abs(),
    )

    # Negative EBITDA should not create misleading negative leverage ratios.
    out.loc[
        out["ebitda"] <= 0,
        ["debt_to_ebitda", "net_debt_to_ebitda"],
    ] = np.nan

    # EBITDA is generally not meaningful for banks, insurers, and financial firms.
    if "financial_flag" in out.columns:
        financial_mask = out["financial_flag"].astype(str).str.lower().isin(
            ["financial", "1", "true"]
        )

        ebitda_cols = [
            "ebitda",
            "ebitda_margin",
            "debt_to_ebitda",
            "net_debt_to_ebitda",
            "ebitda_interest_coverage",
        ]

        out.loc[financial_mask, ebitda_cols] = np.nan

    return out


def _apply_winsor_caps(out, winsor_caps):
    if winsor_caps is None:
        return out

    for col, bounds in winsor_caps.items():
        if col in out.columns and bounds is not None:
            lower, upper = bounds
            out[col] = out[col].clip(lower=lower, upper=upper)

    return out


def _add_size_diagnostics(out):
    out["small_company_flag"] = (out["assets"] < SMALL_COMPANY_ASSET_LIMIT).astype(int)

    out["mid_company_flag"] = out["assets"].between(
        SMALL_COMPANY_ASSET_LIMIT,
        MID_COMPANY_ASSET_LIMIT,
        inclusive="left",
    ).astype(int)

    out["large_company_flag"] = (out["assets"] >= MID_COMPANY_ASSET_LIMIT).astype(int)

    out["asset_size_band"] = pd.cut(
        out["assets"],
        bins=[
            0,
            SMALL_COMPANY_ASSET_LIMIT,
            MID_COMPANY_ASSET_LIMIT,
            LARGE_COMPANY_ASSET_LIMIT,
            np.inf,
        ],
        labels=["small", "mid", "large", "mega"],
    )

    return out


def _add_component_risks(out):
    t = RISK_THRESHOLDS

    out["liabilities_risk"] = linear_risk_bad_high(
        out["liabilities_to_assets"],
        low=t["liabilities_to_assets"]["low"],
        high=t["liabilities_to_assets"]["high"],
    )

    out["debt_load_risk"] = linear_risk_bad_high(
        out["debt_to_assets"],
        low=t["debt_to_assets"]["low"],
        high=t["debt_to_assets"]["high"],
    )

    out["equity_buffer_risk"] = linear_risk_bad_low(
        out["equity_to_assets"],
        good=t["equity_to_assets"]["good"],
        bad=t["equity_to_assets"]["bad"],
    )

    out["cash_buffer_risk"] = linear_risk_bad_low(
        out["cash_to_assets"],
        good=t["cash_to_assets"]["good"],
        bad=t["cash_to_assets"]["bad"],
    )

    out["current_liquidity_risk"] = linear_risk_bad_low(
        out["current_ratio"],
        good=t["current_ratio"]["good"],
        bad=t["current_ratio"]["bad"],
    )

    out["quick_liquidity_risk"] = linear_risk_bad_low(
        out["quick_ratio"],
        good=t["quick_ratio"]["good"],
        bad=t["quick_ratio"]["bad"],
    )

    out["profitability_risk"] = linear_risk_bad_low(
        out["net_income_to_assets"],
        good=t["net_income_to_assets"]["good"],
        bad=t["net_income_to_assets"]["bad"],
    )

    out["cashflow_risk"] = linear_risk_bad_low(
        out["cfo_to_assets"],
        good=t["cfo_to_assets"]["good"],
        bad=t["cfo_to_assets"]["bad"],
    )

    out["coverage_risk"] = linear_risk_bad_low(
        out["interest_coverage"],
        good=t["interest_coverage"]["good"],
        bad=t["interest_coverage"]["bad"],
    )

    out["fcf_risk"] = linear_risk_bad_low(
        out["fcf_to_debt"],
        good=t["fcf_to_debt"]["good"],
        bad=t["fcf_to_debt"]["bad"],
    )

    # Корекция 1: CFO спрямо дълга (не спрямо активите) за operating_cashflow_risk.
    # cfo_to_assets е размерно-зависим и се занулява при капиталоинтензивни компании
    # с голяма asset base. cfo_to_debt директно измерва debt-repayment capacity от
    # оперативния кеш поток.
    cfo_to_debt = safe_divide(out["cfo"], out["total_debt"].abs())
    out["cfo_to_debt_risk"] = linear_risk_bad_low(
        cfo_to_debt,
        good=t["cfo_to_debt"]["good"],
        bad=t["cfo_to_debt"]["bad"],
    )

    out["ebitda_margin_risk"] = linear_risk_bad_low(
        out["ebitda_margin"],
        good=t["ebitda_margin"]["good"],
        bad=t["ebitda_margin"]["bad"],
    )

    out["debt_to_ebitda_risk"] = linear_risk_bad_high(
        out["debt_to_ebitda"],
        low=t["debt_to_ebitda"]["low"],
        high=t["debt_to_ebitda"]["high"],
    )

    out["net_debt_to_ebitda_risk"] = linear_risk_bad_high(
        out["net_debt_to_ebitda"],
        low=t["net_debt_to_ebitda"]["low"],
        high=t["net_debt_to_ebitda"]["high"],
    )

    out["ebitda_coverage_risk"] = linear_risk_bad_low(
        out["ebitda_interest_coverage"],
        good=t["ebitda_interest_coverage"]["good"],
        bad=t["ebitda_interest_coverage"]["bad"],
    )

    # Корекция 5: FCF/debt като мярка за способността за изплащане на дълга
    # от свободния кеш поток — използва се в liquidity_risk формулата.
    out["debt_repayment_risk"] = linear_risk_bad_low(
        out["fcf_to_debt"],
        good=t["debt_repayment_capacity"]["good"],
        bad=t["debt_repayment_capacity"]["bad"],
    )

    out["negative_ebitda_flag"] = np.where(
        out["ebitda"].notna(),
        (out["ebitda"] <= 0).astype(int),
        np.nan,
    )

    return out


def _add_structural_distress_flags(out):
    out["negative_equity_flag"] = (out["equity_to_assets"] < 0).astype(int)

    out["liabilities_exceed_assets_flag"] = (
        out["liabilities_to_assets"] > 1
    ).astype(int)

    # Корекция 4: structural_distress_risk е заменен с градиентен композит.
    # Оригиналният max(binary, binary) дава 0 за всяка компания без negative equity
    # или liabilities > assets — т.е. не разграничава equity_ratio 5% от 50%.
    # Новата формула комбинира вече-изчислените градиентни компоненти:
    #   - equity_buffer_risk:  0 при equity/assets ≥ 0.40, 1 при equity/assets ≤ 0.00
    #   - liabilities_risk:    0 при liabilities/assets ≤ 0.45, 1 при ≥ 1.00
    # Бинарните flags се запазват за guardrails и warning_flags (scoring.py ги ползва).
    # Когато компонентите все още не са изчислени (ранен извик), пада до legacy логика.
    if "equity_buffer_risk" in out.columns and "liabilities_risk" in out.columns:
        out["structural_distress_risk"] = (
            0.60 * out["equity_buffer_risk"]
            + 0.40 * out["liabilities_risk"]
        )
    else:
        out["structural_distress_risk"] = np.maximum(
            out["negative_equity_flag"],
            out["liabilities_exceed_assets_flag"],
        )

    return out


def _add_domain_risk_features(out):
    # All domain features are combined with _weighted_risk, which renormalizes
    # over the components actually present on each row. This prevents a single
    # missing component from collapsing the whole feature to NaN (which the
    # clustering imputer would then backfill to the universe median). For rows
    # with every component present the output equals the original weighted sum.

    # Корекция 3: leverage_risk включва net_debt_to_ebitda_risk като четвърти компонент.
    # Оригиналната формула (liabilities + debt_load + equity_buffer) не съдържа
    # пряка EBITDA-relative leverage метрика. net_debt_to_ebitda_risk вече е изчислен
    # в _add_component_risks и носи директен сигнал за debt capacity.
    # net_debt_to_ebitda_risk е NaN при отрицателна/липсваща EBITDA — в такъв случай
    # leverage_risk се преизчислява само върху наличните три компонента, вместо да
    # стане NaN и да бъде заменен с медианата.
    out["leverage_risk"] = _weighted_risk(
        out,
        [
            ("liabilities_risk", 0.30),
            ("debt_load_risk", 0.25),
            ("equity_buffer_risk", 0.20),
            ("net_debt_to_ebitda_risk", 0.25),
        ],
    )

    # Корекция 5: liquidity_risk добавя debt_repayment_risk като трети компонент.
    # Оригиналната формула (current + quick + cash_buffer) измерва само краткосрочна
    # ликвидност. debt_repayment_risk (FCF/debt) добавя способността за изплащане
    # на дълга от свободния кеш — критичен сигнал за капиталоинтензивни компании.
    # debt_repayment_risk е NaN при нулев/липсващ дълг — тогава ликвидността се
    # оценява само по текущите ликвидни компоненти (renormalization).
    out["liquidity_risk"] = _weighted_risk(
        out,
        [
            ("current_liquidity_risk", 0.35),
            ("quick_liquidity_risk", 0.30),
            ("debt_repayment_risk", 0.20),
            ("cash_buffer_risk", 0.15),
        ],
    )

    out["earnings_risk"] = out["profitability_risk"]

    # Корекция 1: operating_cashflow_risk е заменен с 50/50 комбинация между
    # cashflow_risk (cfo_to_assets — размерна метрика) и cfo_to_debt_risk
    # (cfo_to_debt — debt-relative метрика). Чистото cfo_to_assets се занулява
    # при капиталоинтензивни компании с голяма asset base и умерен FCF margin,
    # давайки false-positive сигнал за нулев риск.
    # При нулев дълг cfo_to_debt_risk е NaN и рискът пада обратно върху cashflow_risk.
    out["operating_cashflow_risk"] = _weighted_risk(
        out,
        [
            ("cashflow_risk", 0.50),
            ("cfo_to_debt_risk", 0.50),
        ],
    )

    # Legacy debt-service risk, kept as a final fallback when the EBITDA-enhanced
    # version has no available components at all.
    out["debt_service_risk_legacy"] = _weighted_risk(
        out,
        [
            ("coverage_risk", 0.60),
            ("fcf_risk", 0.40),
        ],
    )

    # EBITDA-enhanced debt-service risk. Missing EBITDA components (e.g. negative
    # EBITDA) are dropped and the remaining weights renormalized rather than
    # nulling the whole feature.
    out["debt_service_risk"] = _weighted_risk(
        out,
        [
            ("coverage_risk", 0.35),
            ("fcf_risk", 0.25),
            ("debt_to_ebitda_risk", 0.25),
            ("ebitda_coverage_risk", 0.15),
        ],
    )

    out["debt_service_risk"] = out["debt_service_risk"].fillna(
        out["debt_service_risk_legacy"]
    )

    return out


def _add_scorecard_credit_score(out):
    weighted_sum = pd.Series(0.0, index=out.index)
    available_weight = pd.Series(0.0, index=out.index)

    for col, weight in SCORECARD_DOMAIN_WEIGHTS.items():
        if col in out.columns:
            valid = out[col].notna()
            weighted_sum = weighted_sum + out[col].fillna(0) * weight
            available_weight = available_weight + valid.astype(float) * weight

    out["scorecard_credit_score"] = 100 * safe_divide(
        weighted_sum,
        available_weight,
    )

    return out


def engineer_private_company_features(
    input_df,
    winsor_caps=None,
    fx_to_model_currency=1.0,
    min_denominator=SME_MIN_DENOMINATOR,
):
    """
    Convert raw private-company financials into model-ready features.

    This function creates:
    - base balance-sheet, earnings, cash-flow, and liquidity ratios;
    - EBITDA diagnostics;
    - bounded directional risk factors;
    - six domain-level clustering features;
    - diagnostic scorecard credit score.

    The function keeps diagnostic ratios in the returned dataframe so Notebook 03
    can report them even when they are not direct KMeans inputs.
    """
    out = input_df.copy()

    out = ensure_columns(out, REQUIRED_OR_OPTIONAL_FINANCIAL_COLUMNS)
    out = _standardize_current_column_names(out)
    out = _coerce_and_convert_monetary_columns(out, fx_to_model_currency)
    out = _add_derived_accounting_values(out)
    out = _add_base_ratios(out, min_denominator=min_denominator)
    out = _add_diagnostic_ratios(out, min_denominator=min_denominator)
    out = _add_ebitda_metrics(out, min_denominator=min_denominator)
    out = _apply_winsor_caps(out, winsor_caps)
    out = _add_size_diagnostics(out)
    out = _add_component_risks(out)
    out = _add_structural_distress_flags(out)
    out = _add_domain_risk_features(out)
    out = _add_scorecard_credit_score(out)

    return out


# Optional neutral alias for Notebook 02, where the same function is used for
# both training-data feature engineering and private-company serving.
engineer_credit_features = engineer_private_company_features
