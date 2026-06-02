"""
Cluster profiling utilities for the credit clustering project.

This module contains reusable interpretation/reporting logic used by Notebook 02:
cluster profiles, median tables, industry mix, rating-style labels, label maps,
and representative companies.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_EXTREME_QUANTILES,
    DEFAULT_RATING_STYLE_LABELS,
    DEFAULT_SEGMENT_COL,
    INTERPRET_FEATURES,
    SCORECARD_CLUSTER_FEATURES,
)

from .clustering import get_features_for_segment


def existing_columns(df: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Return requested columns that exist in df, preserving order."""
    return [col for col in columns if col in df.columns]


def numeric_existing_columns(df: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Return requested columns that exist and are numeric in df."""
    return [
        col
        for col in columns
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]


def build_interpret_features(
    df: pd.DataFrame,
    interpret_features: Sequence[str] | None = None,
    numeric_only: bool = False,
) -> list[str]:
    """Return the usable interpretation feature list for a dataframe."""
    requested = list(interpret_features or INTERPRET_FEATURES)
    if numeric_only:
        return numeric_existing_columns(df, requested)
    return existing_columns(df, requested)


def build_cluster_profile(
    clustered_panel: pd.DataFrame,
    segment_col: str = DEFAULT_SEGMENT_COL,
    interpret_features: Sequence[str] | None = None,
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """
    Build the main cluster profile table with issuer counts and median features.

    This replaces the Notebook 02 `cluster_profile` groupby cell.
    """
    required = [segment_col, "cluster"]
    missing = [col for col in required if col not in clustered_panel.columns]
    if missing:
        raise KeyError(f"clustered_panel is missing required columns: {missing}")

    features = build_interpret_features(
        clustered_panel,
        interpret_features=interpret_features,
        numeric_only=True,
    )

    agg_spec = {}
    if ticker_col in clustered_panel.columns:
        agg_spec["issuer_years"] = (ticker_col, "size")
        agg_spec["issuers"] = (ticker_col, "nunique")
    else:
        # Fallback for datasets without ticker.
        agg_spec["issuer_years"] = ("cluster", "size")
        agg_spec["issuers"] = ("cluster", "size")

    agg_spec.update({f"median_{col}": (col, "median") for col in features})

    return (
        clustered_panel.groupby([segment_col, "cluster"])
        .agg(**agg_spec)
        .reset_index()
        .sort_values([segment_col, "issuer_years"], ascending=[True, False])
    )


def build_cluster_medians(
    clustered_panel: pd.DataFrame,
    segment_col: str = DEFAULT_SEGMENT_COL,
    profile_cols: Sequence[str] | None = None,
    decimals: int = 3,
) -> pd.DataFrame:
    """Build a compact median table indexed by segment and cluster."""
    if profile_cols is None:
        profile_cols = build_interpret_features(
            clustered_panel,
            interpret_features=INTERPRET_FEATURES,
            numeric_only=True,
        )
    else:
        profile_cols = numeric_existing_columns(clustered_panel, profile_cols)

    return (
        clustered_panel.groupby([segment_col, "cluster"])[list(profile_cols)]
        .median()
        .round(decimals)
    )


def build_feature_extremes(
    df: pd.DataFrame,
    profile_cols: Sequence[str] | None = None,
    quantiles: Sequence[float] = DEFAULT_EXTREME_QUANTILES,
    decimals: int = 3,
) -> pd.DataFrame:
    """Build quantile diagnostics for selected numeric profile columns."""
    if profile_cols is None:
        profile_cols = build_interpret_features(
            df,
            interpret_features=INTERPRET_FEATURES,
            numeric_only=True,
        )
    else:
        profile_cols = numeric_existing_columns(df, profile_cols)

    if not profile_cols:
        return pd.DataFrame()

    return df[list(profile_cols)].quantile(list(quantiles)).T.round(decimals)


def build_industry_cluster_mix(
    clustered_panel: pd.DataFrame,
    segment_col: str = DEFAULT_SEGMENT_COL,
    sector_col: str = "major_sector",
) -> pd.DataFrame:
    """Build industry/sector mix by segment and cluster."""
    required = [segment_col, "cluster", sector_col]
    missing = [col for col in required if col not in clustered_panel.columns]
    if missing:
        raise KeyError(f"clustered_panel is missing required columns: {missing}")

    mix = (
        clustered_panel.groupby([segment_col, "cluster", sector_col])
        .size()
        .reset_index(name="row_count")
    )

    mix["cluster_total"] = mix.groupby([segment_col, "cluster"])["row_count"].transform("sum")
    mix["pct_of_cluster"] = mix["row_count"] / mix["cluster_total"]

    return mix.sort_values(
        [segment_col, "cluster", "pct_of_cluster"],
        ascending=[True, True, False],
    )


def add_rating_style_labels(
    profile: pd.DataFrame,
    segment_col: str = DEFAULT_SEGMENT_COL,
    score_col: str = "median_scorecard_credit_score",
    label_by_rank: Mapping[int, str] | None = None,
) -> pd.DataFrame:
    """
    Rank clusters within each segment and add rating-style labels.

    Lower median scorecard_credit_score means lower risk.
    """
    p = profile.copy()
    label_by_rank = dict(label_by_rank or DEFAULT_RATING_STYLE_LABELS)

    if score_col not in p.columns:
        p["rating_style_rank"] = np.nan
        p["rating_style_label"] = "Unranked"
        return p

    p["rating_style_rank"] = p.groupby(segment_col)[score_col].rank(
        ascending=True,
        method="first",
    )

    p["rating_style_rank"] = p["rating_style_rank"].astype("Int64")
    p["rating_style_label"] = p["rating_style_rank"].map(label_by_rank).fillna("Unranked")

    return p.sort_values([segment_col, "rating_style_rank"])


def build_rating_label_maps(
    cluster_profile_ranked: pd.DataFrame,
    segment_name: str = "Non-financial",
    segment_col: str = DEFAULT_SEGMENT_COL,
) -> tuple[dict[int, str], dict[int, int]]:
    """
    Build production artifact mappings:
    - {cluster_id: rating_style_label}
    - {cluster_id: rating_style_rank}
    """
    required = [segment_col, "cluster", "rating_style_label", "rating_style_rank"]
    missing = [col for col in required if col not in cluster_profile_ranked.columns]
    if missing:
        raise KeyError(f"cluster_profile_ranked is missing required columns: {missing}")

    ranked = cluster_profile_ranked[
        cluster_profile_ranked[segment_col] == segment_name
    ].copy()

    labels_by_cluster = {
        int(row["cluster"]): row["rating_style_label"]
        for _, row in ranked.dropna(subset=["cluster"]).iterrows()
    }

    risk_rank_by_cluster = {
        int(row["cluster"]): int(row["rating_style_rank"])
        for _, row in ranked.dropna(subset=["cluster", "rating_style_rank"]).iterrows()
    }

    return labels_by_cluster, risk_rank_by_cluster


def representatives(
    clustered_panel: pd.DataFrame,
    segment_col: str = DEFAULT_SEGMENT_COL,
    max_names: int = 10,
    features_by_fin_flag: Mapping[str, Sequence[str]] | None = None,
    fallback_features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Pick representative companies close to each cluster median in feature space.

    This mirrors the Notebook 02 representative ticker helper.
    """
    required = [segment_col, "cluster"]
    missing = [col for col in required if col not in clustered_panel.columns]
    if missing:
        raise KeyError(f"clustered_panel is missing required columns: {missing}")

    rows = []
    fallback_features = list(fallback_features or SCORECARD_CLUSTER_FEATURES)

    for (segment_name, cluster_id), group in clustered_panel.groupby([segment_col, "cluster"]):
        features = get_features_for_segment(
            segment_name,
            segment_col=segment_col,
            features_by_fin_flag=features_by_fin_flag,
        )
        if not features:
            features = fallback_features

        features = [col for col in features if col in group.columns and group[col].notna().any()]
        if not features:
            continue

        medians = group[features].median(numeric_only=True)
        X = group[features].copy().fillna(medians)
        distance = ((X - medians) ** 2).sum(axis=1) ** 0.5

        ordered = group.loc[distance.sort_values().index].head(max_names)

        row = {
            segment_col: segment_name,
            "cluster": cluster_id,
        }

        if "ticker" in ordered.columns:
            row["representative_tickers"] = ", ".join(
                ordered["ticker"].astype(str).unique()[:max_names]
            )
        else:
            row["representative_tickers"] = ""

        if "company_name" in ordered.columns:
            row["sample_companies"] = " | ".join(
                ordered["company_name"].dropna().astype(str).unique()[:5]
            )
        else:
            row["sample_companies"] = ""

        if "fiscal_year" in ordered.columns:
            row["sample_years"] = ", ".join(
                ordered["fiscal_year"].dropna().astype(str).unique()[:max_names]
            )

        rows.append(row)

    return pd.DataFrame(rows)


def merge_profile_with_representatives(
    cluster_profile_ranked: pd.DataFrame,
    cluster_representatives: pd.DataFrame,
    segment_col: str = DEFAULT_SEGMENT_COL,
) -> pd.DataFrame:
    """Attach representative companies to the ranked cluster profile."""
    return cluster_profile_ranked.merge(
        cluster_representatives,
        on=[segment_col, "cluster"],
        how="left",
    )


__all__ = [
    "DEFAULT_RATING_STYLE_LABELS",
    "DEFAULT_EXTREME_QUANTILES",
    "existing_columns",
    "numeric_existing_columns",
    "build_interpret_features",
    "build_cluster_profile",
    "build_cluster_medians",
    "build_feature_extremes",
    "build_industry_cluster_mix",
    "add_rating_style_labels",
    "build_rating_label_maps",
    "representatives",
    "merge_profile_with_representatives",
]
