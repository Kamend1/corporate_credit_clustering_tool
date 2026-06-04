"""
Clustering utilities for the credit clustering project.

This module contains the reusable training/evaluation logic used by Notebook 02.
It deliberately does not contain feature engineering. Feature construction should
come from src.credit_clustering.features so Notebook 02 and Notebook 03 use the
same financial-ratio and scorecard logic.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline

from .config import (
    RATIO_COLS,
    SCORECARD_CLUSTER_FEATURES,
    SCORECARD_COMPONENT_FEATURES,
    SCORECARD_DOMAIN_WEIGHTS,
)

from .config import (
    SCORECARD_CLUSTER_FEATURES,
    DEFAULT_SEGMENT_COL,
    DEFAULT_TARGET_SEGMENTS,
    DEFAULT_N_CLUSTERS,
    DEFAULT_MIN_ROWS_PER_SEGMENT,
    DEFAULT_MIN_FEATURES,
    DEFAULT_ROW_FEATURE_COVERAGE,
    DEFAULT_MIN_FEATURE_COVERAGE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_N_INIT,
)


def default_features_by_financial_flag() -> dict[str, list[str]]:
    """Return the default segment-to-feature mapping for the v3 scorecard model."""
    return {
        "Financial": [],
        "Non-financial": list(SCORECARD_CLUSTER_FEATURES),
        "Unknown": [],
    }


def get_features_for_segment(
    segment_name: str,
    segment_col: str = DEFAULT_SEGMENT_COL,
    features_by_fin_flag: Mapping[str, Sequence[str]] | None = None,
    cluster_features: Sequence[str] | None = None,
) -> list[str]:
    """
    Return the intended clustering features for a segment.

    The v3 model clusters only non-financial issuers using bounded scorecard
    domain features. Financial issuers are skipped because EBITDA and standard
    industrial leverage metrics are not appropriate for banks/insurers.
    """
    cluster_features = list(cluster_features or SCORECARD_CLUSTER_FEATURES)

    if segment_col == "financial_flag":
        mapping = features_by_fin_flag or default_features_by_financial_flag()
        return list(mapping.get(segment_name, []))

    if segment_name == "Finance / Insurance / Real Estate":
        return []

    return cluster_features


def select_scorecard_features(
    df: pd.DataFrame,
    requested_features: Sequence[str],
    min_feature_coverage: float = DEFAULT_MIN_FEATURE_COVERAGE,
) -> tuple[list[str], pd.Series]:
    """
    Select available scorecard features for a segment.

    Governance rule:
    - keep the intended rating-style feature set;
    - let the pipeline imputer handle row-level missing values;
    - drop a feature only when it is completely unavailable, unless a stricter
      min_feature_coverage is explicitly supplied.
    """
    features = [feature for feature in requested_features if feature in df.columns]

    if not features:
        return [], pd.Series(dtype=float)

    availability = df[features].notna().mean().sort_values(ascending=False)

    if min_feature_coverage and min_feature_coverage > 0:
        selected = availability[availability >= min_feature_coverage].index.tolist()
    else:
        selected = availability[availability > 0].index.tolist()

    return selected, availability


def minimum_non_null_features(
    feature_count: int,
    min_features: int = DEFAULT_MIN_FEATURES,
    row_feature_coverage: float = DEFAULT_ROW_FEATURE_COVERAGE,
) -> int:
    """Return the row-level minimum number of observed model features."""
    if feature_count <= 0:
        return min_features
    return max(min_features, int(np.ceil(feature_count * row_feature_coverage)))


def filter_rows_by_feature_coverage(
    df: pd.DataFrame,
    features: Sequence[str],
    min_features: int = DEFAULT_MIN_FEATURES,
    row_feature_coverage: float = DEFAULT_ROW_FEATURE_COVERAGE,
) -> tuple[pd.DataFrame, int]:
    """Keep rows with enough non-null model features for meaningful assignment."""
    out = df.copy()
    min_non_null = minimum_non_null_features(
        len(features),
        min_features=min_features,
        row_feature_coverage=row_feature_coverage,
    )

    if features:
        out = out[out[list(features)].notna().sum(axis=1) >= min_non_null].copy()

    return out, min_non_null


def make_kmeans_pipeline(
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_init: int = DEFAULT_N_INIT,
) -> Pipeline:
    """
    Build the v3 KMeans pipeline.

    The scorecard risk factors are already bounded 0-1 and directionally
    comparable, so the pipeline intentionally imputes but does not scale.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "cluster",
                KMeans(
                    n_clusters=n_clusters,
                    init="k-means++",
                    n_init=n_init,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _safe_cluster_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute clustering metrics, returning NaN when not applicable."""
    if len(set(labels)) <= 1:
        return {
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }

    return {
        "silhouette": silhouette_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
    }


def _profile_columns_for_artifact(df: pd.DataFrame) -> list[str]:
    """Return numeric profile columns stored inside each segment artifact."""
    candidate_cols = [
        "scorecard_credit_score",
        *SCORECARD_CLUSTER_FEATURES,
        *SCORECARD_COMPONENT_FEATURES,
        "log_assets",
        "small_company_flag",
        "mid_company_flag",
        "large_company_flag",
        "total_debt",
        "net_debt",
        "ebitda",
        *RATIO_COLS,
    ]

    # Preserve order and remove duplicates.
    seen = set()
    ordered_unique = []
    for col in candidate_cols:
        if col not in seen:
            seen.add(col)
            ordered_unique.append(col)

    return [
        col
        for col in ordered_unique
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]


def cluster_segment(
    df: pd.DataFrame,
    model: Pipeline,
    segment_name: str,
    n_clusters = DEFAULT_N_CLUSTERS,
    segment_col: str = DEFAULT_SEGMENT_COL,
    min_rows: int = DEFAULT_MIN_ROWS_PER_SEGMENT,
    cluster_only_segments: Iterable[str] | None = DEFAULT_TARGET_SEGMENTS,
    min_feature_coverage: float = DEFAULT_MIN_FEATURE_COVERAGE,
    min_features: int = DEFAULT_MIN_FEATURES,
    row_feature_coverage: float = DEFAULT_ROW_FEATURE_COVERAGE,
    features_by_fin_flag: Mapping[str, Sequence[str]] | None = None,
) -> tuple[pd.DataFrame | None, dict, dict | None]:
    """
    Cluster one segment and return (clustered_df, metrics, segment_artifact).

    This is the direct src replacement for the Notebook 02 `cluster_segment`
    method cell.
    """
    if segment_col not in df.columns:
        raise KeyError(f"segment_col '{segment_col}' not found in dataframe.")

    if cluster_only_segments and segment_name not in set(cluster_only_segments):
        metrics = {
            "segment": segment_name,
            "status": "skipped_not_target_segment",
            "rows": int((df[segment_col] == segment_name).sum()),
            "features": 0,
            "feature_list": [],
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }
        return None, metrics, None

    requested_features = get_features_for_segment(
        segment_name,
        segment_col=segment_col,
        features_by_fin_flag=features_by_fin_flag,
    )

    use = df[df[segment_col] == segment_name].copy()

    if {"ticker", "fiscal_year"}.issubset(use.columns):
        use = use.dropna(subset=["ticker", "fiscal_year"])

    features, availability = select_scorecard_features(
        use,
        requested_features,
        min_feature_coverage=min_feature_coverage,
    )

    use, min_non_null_features = filter_rows_by_feature_coverage(
        use,
        features,
        min_features=min_features,
        row_feature_coverage=row_feature_coverage,
    )

    if len(use) < min_rows or len(features) < min_features:
        metrics = {
            "segment": segment_name,
            "status": "skipped",
            "rows": len(use),
            "features": len(features),
            "feature_list": features,
            "feature_availability": availability.to_dict(),
            "min_non_null_features": min_non_null_features,
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }
        return None, metrics, None

    X = use[features]
    pipe = model

    labels = pipe.fit_predict(X)

    use["cluster"] = labels
    use["cluster_key"] = use[segment_col].astype(str) + "__" + use["cluster"].astype(str)

    X_prepared = pipe[:-1].transform(X)
    metric_values = _safe_cluster_metrics(X_prepared, labels)

    metrics = {
        "segment": segment_name,
        "status": "clustered",
        "rows": len(use),
        "features": len(features),
        "feature_list": features,
        "feature_availability": availability.to_dict(),
        "min_non_null_features": min_non_null_features,
        **metric_values,
    }

    profile_cols = _profile_columns_for_artifact(use)
    cluster_profile = use.groupby("cluster")[profile_cols].median().round(6)

    if "ticker" in use.columns:
        cluster_sizes = (
            use.groupby("cluster")
            .agg(
                issuer_years=("ticker", "size"),
                issuers=("ticker", "nunique"),
            )
            .reset_index()
        )
    else:
        cluster_sizes = use.groupby("cluster").size().reset_index(name="issuer_years")
        cluster_sizes["issuers"] = cluster_sizes["issuer_years"]

    model_artifact = {
        "segment_name": segment_name,
        "segment_col": segment_col,
        "n_clusters": n_clusters,
        "pipeline": pipe,
        "feature_cols": features,
        "availability": availability.to_dict(),
        "min_non_null_features": min_non_null_features,
        "metrics": metrics,
        "cluster_profile": cluster_profile,
        "cluster_sizes": cluster_sizes,
        "scorecard_domain_weights": SCORECARD_DOMAIN_WEIGHTS,
        "scorecard_cluster_features": list(SCORECARD_CLUSTER_FEATURES),
        "scorecard_component_features": list(SCORECARD_COMPONENT_FEATURES),
        "ratio_cols": list(RATIO_COLS),
        "notes": (
            "V3 model clusters on bounded rating-style risk factors. "
            "Absolute size is diagnostic only."
        ),
    }

    return use, metrics, model_artifact


def cluster_segments(
    df: pd.DataFrame,
    model: Pipeline,
    segment_col: str = DEFAULT_SEGMENT_COL,
    segment_names: Sequence[str] | None = None,
    **cluster_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    """
    Cluster all requested segments and return:
    - clustered_panel
    - metrics_df
    - segment_artifacts
    """
    if segment_col not in df.columns:
        raise KeyError(f"segment_col '{segment_col}' not found in dataframe.")

    if segment_names is None:
        segment_names = sorted(df[segment_col].dropna().unique())

    clustered_parts = []
    metrics = []
    segment_artifacts = {}

    for segment_name in segment_names:
        clustered, metric_row, artifact = cluster_segment(
            df,
            model,
            segment_name,
            segment_col=segment_col,
            **cluster_kwargs,
        )

        metrics.append(metric_row)

        if clustered is not None:
            clustered_parts.append(clustered)

        if artifact is not None:
            segment_artifacts[segment_name] = artifact

    clustered_panel = (
        pd.concat(clustered_parts, ignore_index=True)
        if clustered_parts
        else pd.DataFrame()
    )

    metrics_df = pd.DataFrame(metrics)

    return clustered_panel, metrics_df, segment_artifacts


def evaluate_k_range(
    df: pd.DataFrame,
    segment_name: str,
    model: Pipeline,
    segment_col: str = DEFAULT_SEGMENT_COL,
    k_values: Iterable[int] = range(2, 9),
    cluster_only_segments: Iterable[str] | None = DEFAULT_TARGET_SEGMENTS,
    min_rows: int = DEFAULT_MIN_ROWS_PER_SEGMENT,
    min_features: int = DEFAULT_MIN_FEATURES,
    row_feature_coverage: float = DEFAULT_ROW_FEATURE_COVERAGE,
    min_feature_coverage: float = DEFAULT_MIN_FEATURE_COVERAGE,
    features_by_fin_flag: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Evaluate KMeans quality metrics across a range of k values using the supplied model pipeline."""

    if cluster_only_segments and segment_name not in set(cluster_only_segments):
        return pd.DataFrame()

    requested_features = get_features_for_segment(
        segment_name,
        segment_col=segment_col,
        features_by_fin_flag=features_by_fin_flag,
    )

    use = df[df[segment_col] == segment_name].copy()

    features, _ = select_scorecard_features(
        use,
        requested_features,
        min_feature_coverage=min_feature_coverage,
    )

    use, min_non_null_features = filter_rows_by_feature_coverage(
        use,
        features,
        min_features=min_features,
        row_feature_coverage=row_feature_coverage,
    )

    if len(use) < min_rows or len(features) < min_features:
        return pd.DataFrame()

    X = use[features]

    rows = []

    for k in k_values:
        pipe = clone(model)

        if "cluster" not in pipe.named_steps:
            raise KeyError("The supplied model pipeline must contain a step named 'cluster'.")

        pipe.set_params(cluster__n_clusters=k)

        labels = pipe.fit_predict(X)

        # Use all preprocessing steps before the final cluster step
        # for distance-based metric calculation.
        X_prepared = pipe[:-1].transform(X)

        rows.append(
            {
                "segment": segment_name,
                "k": k,
                "rows": len(use),
                "features": len(features),
                "feature_list": features,
                "min_non_null_features": min_non_null_features,
                "silhouette": silhouette_score(X_prepared, labels),
                "calinski_harabasz": calinski_harabasz_score(X_prepared, labels),
                "davies_bouldin": davies_bouldin_score(X_prepared, labels),
            }
        )

    return pd.DataFrame(rows)


def evaluate_segments_k_range(
    df: pd.DataFrame,
    model: Pipeline,
    segment_col: str = DEFAULT_SEGMENT_COL,
    segment_names: Sequence[str] | None = None,
    **evaluate_kwargs,
) -> pd.DataFrame:
    """Evaluate k ranges for all requested segments using the supplied model pipeline."""

    if segment_col not in df.columns:
        raise KeyError(f"segment_col '{segment_col}' not found in dataframe.")

    if segment_names is None:
        segment_names = sorted(df[segment_col].dropna().unique())

    frames = []

    for segment_name in segment_names:
        result = evaluate_k_range(
            df=df,
            segment_name=segment_name,
            model=model,
            segment_col=segment_col,
            **evaluate_kwargs,
        )

        if not result.empty:
            frames.append(result)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


__all__ = [
    "DEFAULT_SEGMENT_COL",
    "DEFAULT_TARGET_SEGMENTS",
    "DEFAULT_N_CLUSTERS",
    "DEFAULT_MIN_ROWS_PER_SEGMENT",
    "DEFAULT_MIN_FEATURES",
    "DEFAULT_ROW_FEATURE_COVERAGE",
    "DEFAULT_MIN_FEATURE_COVERAGE",
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_N_INIT",
    "default_features_by_financial_flag",
    "get_features_for_segment",
    "select_scorecard_features",
    "minimum_non_null_features",
    "filter_rows_by_feature_coverage",
    "make_kmeans_pipeline",
    "cluster_segment",
    "cluster_segments",
    "evaluate_k_range",
    "evaluate_segments_k_range",
]
