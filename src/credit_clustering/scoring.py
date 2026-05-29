import numpy as np
import pandas as pd


def _ensure_columns(df, columns, default=np.nan):
    out = df.copy()

    for col in columns:
        if col not in out.columns:
            out[col] = default

    return out


def safe_divide(numerator, denominator, min_abs_denominator=None):
    numerator = pd.Series(numerator).astype(float)
    denominator = pd.Series(denominator).astype(float)

    out = numerator / denominator

    if min_abs_denominator is not None:
        out = out.where(denominator.abs() >= min_abs_denominator)

    return out.replace([np.inf, -np.inf], np.nan)


def engineer_private_company_features(
    input_df,
    winsor_caps=None,
    fx_to_model_currency=1.0,
):
    """
    Convert raw private-company financials into model-ready features.

    This function creates:
    1. Raw financial ratios used for diagnostics and interpretation.
    2. Engineered risk features required by the trained clustering model:
       - structural_distress_risk
       - earnings_risk
       - operating_cashflow_risk
       - liquidity_risk
       - leverage_risk
       - debt_service_risk

    fx_to_model_currency:
        Multiplier used to convert absolute monetary values into the model currency.
        If the model was trained on USD and input is EUR, use EUR/USD.
    """

    out = input_df.copy()

    required_or_optional = [
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
    ]

    out = _ensure_columns(out, required_or_optional)

    # Accept both EDGAR-style and business-friendly names.
    if out["assets_current"].isna().all() and "current_assets" in out.columns:
        out["assets_current"] = out["current_assets"]

    if out["liabilities_current"].isna().all() and "current_liabilities" in out.columns:
        out["liabilities_current"] = out["current_liabilities"]

    monetary_cols = [
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
    ]

    for col in monetary_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Convert only absolute monetary values.
    # Ratios are calculated after conversion.
    for col in monetary_cols:
        out[col] = out[col] * fx_to_model_currency

    out["long_term_debt"] = out["long_term_debt"].fillna(0)
    out["short_term_debt"] = out["short_term_debt"].fillna(0)
    out["debt"] = out["long_term_debt"] + out["short_term_debt"]

    out["capex"] = out["capex"].fillna(0)
    out["free_cash_flow"] = out["cfo"] - out["capex"]

    out["log_assets"] = np.log1p(out["assets"].clip(lower=0))

    out["liabilities_to_assets"] = safe_divide(
        out["liabilities"],
        out["assets"],
        10_000_000,
    )

    out["debt_to_assets"] = safe_divide(
        out["debt"],
        out["assets"],
        10_000_000,
    )

    out["equity_to_assets"] = safe_divide(
        out["equity"],
        out["assets"],
        10_000_000,
    )

    out["cash_to_assets"] = safe_divide(
        out["cash"],
        out["assets"],
        10_000_000,
    )

    out["net_income_to_assets"] = safe_divide(
        out["net_income"],
        out["assets"],
        10_000_000,
    )

    out["cfo_to_assets"] = safe_divide(
        out["cfo"],
        out["assets"],
        10_000_000,
    )

    out["revenue_to_assets"] = safe_divide(
        out["revenue"],
        out["assets"],
        10_000_000,
    )

    out["current_ratio"] = safe_divide(
        out["assets_current"],
        out["liabilities_current"],
        1_000_000,
    )

    out["quick_ratio"] = safe_divide(
        out["cash"].fillna(0) + out["receivables"].fillna(0),
        out["liabilities_current"],
        1_000_000,
    )

    # Diagnostic ratios.
    # These can be used for interpretation even if not used directly by KMeans.
    out["operating_margin"] = safe_divide(
        out["operating_income"],
        out["revenue"],
        10_000_000,
    )

    out["gross_margin"] = safe_divide(
        out["gross_profit"],
        out["revenue"],
        10_000_000,
    )

    out["interest_coverage"] = safe_divide(
        out["operating_income"],
        out["interest_expense"],
        1_000_000,
    )

    out["fcf_to_debt"] = safe_divide(
        out["free_cash_flow"],
        out["debt"],
        1_000_000,
    )

    # Apply training-time winsorization/capping.
    if winsor_caps is not None:
        for col, bounds in winsor_caps.items():
            if col in out.columns and bounds is not None:
                lower, upper = bounds
                out[col] = out[col].clip(lower=lower, upper=upper)

    # ------------------------------------------------------------------
    # Engineered risk features required by the trained clustering model
    # ------------------------------------------------------------------

    out["liabilities_risk"] = (
        (out["liabilities_to_assets"] - 0.50) / (1.00 - 0.50)
    ).clip(0, 1)

    out["debt_load_risk"] = (
        (out["debt_to_assets"] - 0.25) / (0.75 - 0.25)
    ).clip(0, 1)

    out["equity_buffer_risk"] = (
        (0.40 - out["equity_to_assets"]) / (0.40 - 0.00)
    ).clip(0, 1)

    out["leverage_risk"] = out[
        [
            "liabilities_risk",
            "debt_load_risk",
            "equity_buffer_risk",
        ]
    ].mean(axis=1)

    out["cash_buffer_risk"] = (
        (0.10 - out["cash_to_assets"]) / 0.10
    ).clip(0, 1)

    out["current_liquidity_risk"] = (
        (1.50 - out["current_ratio"]) / (1.50 - 0.75)
    ).clip(0, 1)

    out["quick_liquidity_risk"] = (
        (1.00 - out["quick_ratio"]) / (1.00 - 0.50)
    ).clip(0, 1)

    out["liquidity_risk"] = out[
        [
            "cash_buffer_risk",
            "current_liquidity_risk",
            "quick_liquidity_risk",
        ]
    ].mean(axis=1)

    out["profitability_risk"] = (
        (0.03 - out["net_income_to_assets"]) / (0.03 - (-0.10))
    ).clip(0, 1)

    out["earnings_risk"] = out["profitability_risk"]

    out["cashflow_risk"] = (
        (0.03 - out["cfo_to_assets"]) / (0.03 - (-0.10))
    ).clip(0, 1)

    out["operating_cashflow_risk"] = out["cashflow_risk"]

    out["coverage_risk"] = (
        (3.0 - out["interest_coverage"]) / (3.0 - 1.0)
    ).clip(0, 1)

    out["fcf_risk"] = (
        (0.05 - out["fcf_to_debt"]) / (0.05 - (-0.25))
    ).clip(0, 1)

    out["debt_service_risk"] = out[
        [
            "coverage_risk",
            "fcf_risk",
        ]
    ].mean(axis=1)

    out["negative_equity_flag"] = (
        out["equity_to_assets"] < 0
    ).astype(float)

    out["liabilities_exceed_assets_flag"] = (
        out["liabilities_to_assets"] > 1
    ).astype(float)

    out["structural_distress_risk"] = out[
        [
            "negative_equity_flag",
            "liabilities_exceed_assets_flag",
        ]
    ].max(axis=1)

    return out


def soft_cluster_scores(distances, temperature=1.0):
    distances = np.asarray(distances, dtype=float)
    similarities = np.exp(-distances / temperature)

    return similarities / similarities.sum(axis=1, keepdims=True)


def make_warning_flags(row):
    flags = []

    if pd.notna(row.get("assets")) and row.get("assets") <= 0:
        flags.append("invalid_assets")

    if pd.notna(row.get("assets")) and row.get("assets") < 10_000_000:
        flags.append("assets_below_model_threshold")

    if pd.notna(row.get("liabilities_to_assets")) and row.get("liabilities_to_assets") > 1:
        flags.append("liabilities_exceed_assets")

    if pd.notna(row.get("equity_to_assets")) and row.get("equity_to_assets") < 0:
        flags.append("negative_equity")

    if pd.notna(row.get("debt_to_assets")) and row.get("debt_to_assets") > 0.75:
        flags.append("high_debt_to_assets")

    if pd.notna(row.get("current_ratio")) and row.get("current_ratio") < 1:
        flags.append("current_ratio_below_1")

    if pd.notna(row.get("quick_ratio")) and row.get("quick_ratio") < 0.5:
        flags.append("quick_ratio_below_0_5")

    if pd.notna(row.get("interest_coverage")) and row.get("interest_coverage") < 1:
        flags.append("interest_coverage_below_1")

    if pd.notna(row.get("cfo_to_assets")) and row.get("cfo_to_assets") < 0:
        flags.append("negative_cfo_to_assets")

    return ", ".join(flags) if flags else "none"


def score_companies(
    input_df,
    artifact,
    segment="Non-financial",
    temperature=1.0,
    fx_to_model_currency=1.0,
):
    """
    Score private companies using the trained KMeans clustering artifact.

    Main output:
    - assigned_cluster
    - cluster_label
    - risk_rank
    - cluster_affinity
    - distance_to_assigned_cluster
    - feature_coverage_pct
    - warning_flags

    Note:
    The segment argument is currently reserved for future use.
    """

    scored = engineer_private_company_features(
        input_df,
        winsor_caps=artifact.get("winsor_caps"),
        fx_to_model_currency=fx_to_model_currency,
    )

    pipe = artifact["pipeline"]
    feature_cols = artifact["feature_cols"]
    cluster_labels = artifact.get("cluster_labels", {})
    risk_rank = artifact.get("risk_rank", {})

    missing_features = [col for col in feature_cols if col not in scored.columns]

    if missing_features:
        raise KeyError(
            "The scored dataframe is missing model feature columns: "
            f"{missing_features}"
        )

    X_new = scored[feature_cols]

    assigned = pipe.predict(X_new)

    X_prepared = pipe[:-1].transform(X_new)
    distances = pipe.named_steps["cluster"].transform(X_prepared)
    affinities = soft_cluster_scores(distances, temperature=temperature)

    scored["assigned_cluster"] = assigned

    scored["distance_to_assigned_cluster"] = distances[
        np.arange(len(assigned)),
        assigned,
    ]

    scored["cluster_label"] = scored["assigned_cluster"].map(cluster_labels)
    scored["risk_rank"] = scored["assigned_cluster"].map(risk_rank)
    scored["cluster_affinity"] = affinities.max(axis=1)

    # Cluster 4 is the current near-default / distressed proxy.
    # Keep this only if your trained artifact uses this convention.
    scored["near_default_affinity"] = (
        affinities[:, 4] if affinities.shape[1] > 4 else np.nan
    )

    for i in range(affinities.shape[1]):
        scored[f"cluster_{i}_affinity"] = affinities[:, i]
        scored[f"cluster_{i}_distance"] = distances[:, i]

    scored["feature_coverage_pct"] = scored[feature_cols].notna().mean(axis=1)
    scored["warning_flags"] = scored.apply(make_warning_flags, axis=1)

    return scored