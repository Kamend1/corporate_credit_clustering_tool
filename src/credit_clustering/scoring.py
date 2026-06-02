"""
Company scoring / serving utilities for the credit clustering project.

This module intentionally does not contain feature-engineering logic.
Feature construction lives in features.py, so Notebook 02 training and
Notebook 03 private-company scoring use the same definitions.
"""

import numpy as np
import pandas as pd

from .config import PUBLIC_COMPANY_MIN_ASSETS
from .features import engineer_private_company_features


def _resolve_artifact(artifact, segment=None):
    """
    Return the concrete segment artifact used for scoring.

    Supports both artifact layouts:
    1. Single-segment artifact:
       artifact["pipeline"], artifact["feature_cols"], ...
    2. Multi-segment wrapper:
       artifact["segment_artifacts"][segment]
    """
    if not isinstance(artifact, dict):
        raise TypeError("artifact must be a dictionary-like object.")

    if "segment_artifacts" not in artifact:
        return artifact

    if segment is None:
        raise KeyError(
            "artifact contains 'segment_artifacts', but no segment was provided."
        )

    segment_artifacts = artifact["segment_artifacts"]

    if segment not in segment_artifacts:
        available_segments = list(segment_artifacts.keys())
        raise KeyError(
            f"Segment {segment!r} not found in artifact['segment_artifacts']. "
            f"Available segments: {available_segments}"
        )

    return segment_artifacts[segment]


def infer_near_default_cluster_from_artifact(artifact, segment):
    """
    Infer the near-default / most distressed cluster from the loaded artifact.

    Preferred source:
    - risk_rank mapping: highest risk_rank = worst cluster.

    Fallback source:
    - cluster_profile_ranked table with highest risk_rank.

    Returns None if no reliable inference is possible.
    """

    # Handle multi-segment artifact.
    if "segment_artifacts" in artifact:
        segment_artifact = artifact["segment_artifacts"].get(segment, {})
    else:
        segment_artifact = artifact

    risk_rank = segment_artifact.get("risk_rank")

    if isinstance(risk_rank, dict) and len(risk_rank) > 0:
        return int(max(risk_rank, key=risk_rank.get))

    profile = segment_artifact.get("cluster_profile_ranked")

    if profile is None:
        profile = artifact.get("cluster_profile_ranked")

    if profile is not None and {"cluster", "risk_rank"}.issubset(profile.columns):
        profile_use = profile.copy()

        if "segment" in profile_use.columns:
            profile_use = profile_use.loc[
                profile_use["segment"].eq(segment)
            ].copy()

        if len(profile_use) > 0:
            worst_row = profile_use.sort_values("risk_rank").iloc[-1]
            return int(worst_row["cluster"])

    return None


def soft_cluster_scores(distances, temperature=1.0):
    """
    Convert distances to soft cluster affinities using an exponential kernel.

    Smaller distance -> higher affinity. The affinities are row-normalized.
    """
    distances = np.asarray(distances, dtype=float)

    if distances.ndim != 2:
        raise ValueError("distances must be a 2D array.")

    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    similarities = np.exp(-distances / temperature)
    denominator = similarities.sum(axis=1, keepdims=True)

    return similarities / denominator


def make_warning_flags(row):
    """
    Build human-readable diagnostic flags from a scored company row.

    These flags are not part of KMeans. They are reporting diagnostics for
    Notebook 03 and exported score reports.
    """
    flags = []

    if pd.notna(row.get("assets")) and row.get("assets") <= 0:
        flags.append("invalid_assets")

    if pd.notna(row.get("assets")) and row.get("assets") < PUBLIC_COMPANY_MIN_ASSETS:
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

    if pd.notna(row.get("ebitda_interest_coverage")) and row.get("ebitda_interest_coverage") < 1.5:
        flags.append("ebitda_interest_coverage_below_1_5")

    if pd.notna(row.get("debt_to_ebitda")) and row.get("debt_to_ebitda") > 6:
        flags.append("debt_to_ebitda_above_6")

    if pd.notna(row.get("net_debt_to_ebitda")) and row.get("net_debt_to_ebitda") > 5:
        flags.append("net_debt_to_ebitda_above_5")

    if pd.notna(row.get("ebitda")) and row.get("ebitda") <= 0:
        flags.append("negative_or_zero_ebitda")

    if pd.notna(row.get("cfo_to_assets")) and row.get("cfo_to_assets") < 0:
        flags.append("negative_cfo_to_assets")

    return ", ".join(flags) if flags else "none"


def _get_pipeline_distances(pipe, X_new):
    """
    Return distances from rows to all KMeans cluster centers.

    The preferred path mirrors the existing implementation:
    preprocessors are applied via pipe[:-1], then distances are obtained from
    the final KMeans step named 'cluster'. A fallback uses pipeline.transform().
    """
    if "cluster" in getattr(pipe, "named_steps", {}):
        X_prepared = pipe[:-1].transform(X_new)
        return pipe.named_steps["cluster"].transform(X_prepared)

    if hasattr(pipe, "transform"):
        return pipe.transform(X_new)

    raise AttributeError(
        "Could not compute cluster distances. Expected a sklearn Pipeline "
        "with a final 'cluster' step or a transform() method."
    )


def score_companies(
    input_df,
    artifact,
    segment,
    temperature,
    fx_to_model_currency,
    min_denominator,
    near_default_cluster,
):
    """
    Score private companies using a trained KMeans clustering artifact.

    The function:
    1. resolves the correct segment artifact;
    2. engineers features using features.py;
    3. predicts the assigned KMeans cluster;
    4. calculates cluster distances and soft affinities;
    5. adds labels, risk ranks, feature coverage, and warning flags.

    Parameters
    ----------
    input_df : pd.DataFrame
        Raw company financials.
    artifact : dict
        Either a single-segment artifact or a wrapper with 'segment_artifacts'.
    segment : str, default "Non-financial"
        Segment key when artifact contains multiple segment artifacts.
    temperature : float, default 1.0
        Soft-affinity temperature. Lower values make affinities sharper.
    fx_to_model_currency : float, default 1.0
        Multiplier for converting input monetary values to model currency.
    min_denominator : float or None, default None
        Optional denominator threshold passed to feature engineering. If None,
        features.py uses its configured default.
    near_default_cluster : int, default 4
        Cluster id used as near-default proxy, if present.
    """
    segment_artifact = _resolve_artifact(artifact, segment=segment)

    feature_kwargs = {
        "winsor_caps": segment_artifact.get("winsor_caps"),
        "fx_to_model_currency": fx_to_model_currency,
    }

    if min_denominator is not None:
        feature_kwargs["min_denominator"] = min_denominator

    scored = engineer_private_company_features(input_df, **feature_kwargs)

    if "pipeline" not in segment_artifact:
        raise KeyError("artifact['pipeline'] not found.")

    if "feature_cols" not in segment_artifact:
        raise KeyError("artifact['feature_cols'] not found.")

    pipe = segment_artifact["pipeline"]
    feature_cols = segment_artifact["feature_cols"]
    cluster_labels = segment_artifact.get("cluster_labels", {})
    risk_rank = segment_artifact.get("risk_rank", {})

    missing_features = [col for col in feature_cols if col not in scored.columns]

    if missing_features:
        raise KeyError(
            "The scored dataframe is missing model feature columns: "
            f"{missing_features}"
        )

    X_new = scored[feature_cols]

    assigned = pipe.predict(X_new)
    distances = _get_pipeline_distances(pipe, X_new)
    affinities = soft_cluster_scores(distances, temperature=temperature)

    scored["assigned_cluster"] = assigned
    scored["distance_to_assigned_cluster"] = distances[
        np.arange(len(assigned)),
        assigned,
    ]

    scored["cluster_label"] = scored["assigned_cluster"].map(cluster_labels)
    scored["risk_rank"] = scored["assigned_cluster"].map(risk_rank)
    scored["cluster_affinity"] = affinities.max(axis=1)

    if near_default_cluster is not None and affinities.shape[1] > near_default_cluster:
        scored["near_default_affinity"] = affinities[:, near_default_cluster]
    else:
        scored["near_default_affinity"] = np.nan

    for i in range(affinities.shape[1]):
        scored[f"cluster_{i}_affinity"] = affinities[:, i]
        scored[f"cluster_{i}_distance"] = distances[:, i]

    scored["feature_coverage_pct"] = scored[feature_cols].notna().mean(axis=1)
    scored["warning_flags"] = scored.apply(make_warning_flags, axis=1)

    return scored
