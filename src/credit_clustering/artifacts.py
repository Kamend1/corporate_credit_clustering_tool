"""
Artifact utilities for the credit clustering project.

This module owns the model-artifact schema and persistence helpers used by
Notebook 02 and Notebook 03.  It keeps Notebook 02 focused on orchestration:
train/evaluate/profile clusters, then build and save one clean artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import joblib
import pandas as pd

from .config import (
    DEFAULT_ARTIFACT_VERSION,
    DEFAULT_PRIMARY_SEGMENT,
    DEFAULT_SEGMENT_COL,
    INTERPRET_FEATURES,
    RATIO_COLS,
    SCORECARD_CLUSTER_FEATURES,
    SCORECARD_COMPONENT_FEATURES,
    SCORECARD_DOMAIN_WEIGHTS,
)
from .profiling import build_rating_label_maps


def _as_path(path: str | Path) -> Path:
    """Return a pathlib Path and create the parent folder if needed."""
    path = Path(path)
    if path.parent and str(path.parent) != ".":
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _copy_if_dataframe(value: Any) -> Any:
    """Copy pandas objects to avoid accidental mutation inside artifact builders."""
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.copy()
    return value


def _normalize_mapping_keys_to_int(mapping: Mapping | None) -> dict[int, Any]:
    """Normalize cluster-id mapping keys to int where possible."""
    if mapping is None:
        return {}

    out: dict[int, Any] = {}
    for key, value in dict(mapping).items():
        try:
            out[int(key)] = value
        except (TypeError, ValueError):
            out[key] = value
    return out


def get_segment_artifact(
    artifact: Mapping[str, Any],
    segment: str = DEFAULT_PRIMARY_SEGMENT,
) -> Mapping[str, Any]:
    """
    Return the segment-specific artifact regardless of artifact shape.

    Supports both schemas:
    1. multi-segment artifact:
       artifact["segment_artifacts"][segment]
    2. single-segment artifact:
       artifact itself contains "pipeline" and "feature_cols".
    """
    if "segment_artifacts" in artifact:
        segment_artifacts = artifact.get("segment_artifacts") or {}
        if segment not in segment_artifacts:
            available = sorted(segment_artifacts.keys())
            raise KeyError(
                f"Segment '{segment}' not found in artifact['segment_artifacts']. "
                f"Available segments: {available}"
            )
        return segment_artifacts[segment]

    if "pipeline" in artifact and "feature_cols" in artifact:
        return artifact

    raise KeyError(
        "Artifact schema not recognized. Expected either a multi-segment "
        "artifact with key 'segment_artifacts' or a single-segment artifact "
        "with keys 'pipeline' and 'feature_cols'."
    )


def enrich_segment_artifact(
    segment_artifact: Mapping[str, Any],
    cluster_profile_ranked: pd.DataFrame | None = None,
    segment_name: str | None = None,
    segment_col: str = DEFAULT_SEGMENT_COL,
    cluster_labels: Mapping[int, str] | None = None,
    risk_rank: Mapping[int, int] | None = None,
    winsor_caps: Mapping[str, tuple[float, float]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Return a production-ready segment artifact.

    Adds rating labels/risk ranks when a ranked cluster profile is available,
    while preserving the fitted pipeline, feature columns, metrics, cluster sizes,
    and profile tables produced by clustering.py.
    """
    out = {key: _copy_if_dataframe(value) for key, value in dict(segment_artifact).items()}

    if segment_name is None:
        segment_name = out.get("segment_name", DEFAULT_PRIMARY_SEGMENT)

    if cluster_profile_ranked is not None:
        out["cluster_profile_ranked"] = cluster_profile_ranked.copy()

        if cluster_labels is None or risk_rank is None:
            labels_from_profile, rank_from_profile = build_rating_label_maps(
                cluster_profile_ranked,
                segment_name=segment_name,
                segment_col=segment_col,
            )
            cluster_labels = cluster_labels or labels_from_profile
            risk_rank = risk_rank or rank_from_profile

    out["cluster_labels"] = _normalize_mapping_keys_to_int(cluster_labels or out.get("cluster_labels"))
    out["risk_rank"] = _normalize_mapping_keys_to_int(risk_rank or out.get("risk_rank"))

    if winsor_caps is not None:
        out["winsor_caps"] = dict(winsor_caps)
    else:
        out.setdefault("winsor_caps", None)

    out.setdefault("segment_name", segment_name)
    out.setdefault("segment_col", segment_col)
    out.setdefault("feature_cols", list(out.get("feature_cols", SCORECARD_CLUSTER_FEATURES)))
    out.setdefault("scorecard_domain_weights", dict(SCORECARD_DOMAIN_WEIGHTS))
    out.setdefault("scorecard_cluster_features", list(SCORECARD_CLUSTER_FEATURES))
    out.setdefault("scorecard_component_features", list(SCORECARD_COMPONENT_FEATURES))
    out.setdefault("ratio_cols", list(RATIO_COLS))
    out.setdefault("interpret_features", list(INTERPRET_FEATURES))

    if extra:
        out.update(dict(extra))

    return out


def build_credit_clustering_artifact(
    segment_artifacts: Mapping[str, Mapping[str, Any]],
    cluster_profile_ranked: pd.DataFrame | None = None,
    primary_segment: str = DEFAULT_PRIMARY_SEGMENT,
    segment_col: str = DEFAULT_SEGMENT_COL,
    metrics_df: pd.DataFrame | None = None,
    cluster_profile: pd.DataFrame | None = None,
    cluster_medians: pd.DataFrame | None = None,
    feature_extremes: pd.DataFrame | None = None,
    industry_cluster_mix: pd.DataFrame | None = None,
    winsor_caps: Mapping[str, tuple[float, float]] | None = None,
    artifact_version: str = DEFAULT_ARTIFACT_VERSION,
    notes: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the full multi-segment artifact saved by Notebook 02.

    The output intentionally contains both:
    - segment_artifacts: future-proof multi-segment structure;
    - top-level primary-segment aliases: backward compatibility with older
      Notebook 03 code that expects artifact['pipeline'] and artifact['feature_cols'].
    """
    if not segment_artifacts:
        raise ValueError("segment_artifacts is empty; nothing to persist.")

    if primary_segment not in segment_artifacts:
        if len(segment_artifacts) == 1:
            primary_segment = next(iter(segment_artifacts.keys()))
        else:
            available = sorted(segment_artifacts.keys())
            raise KeyError(
                f"Primary segment '{primary_segment}' not present in segment_artifacts. "
                f"Available segments: {available}"
            )

    enriched_segments: dict[str, dict[str, Any]] = {}
    for segment_name, seg_artifact in segment_artifacts.items():
        profile_for_segment = cluster_profile_ranked
        enriched_segments[segment_name] = enrich_segment_artifact(
            seg_artifact,
            cluster_profile_ranked=profile_for_segment,
            segment_name=segment_name,
            segment_col=segment_col,
            winsor_caps=winsor_caps,
        )

    primary = enriched_segments[primary_segment]

    artifact: dict[str, Any] = {
        "artifact_version": artifact_version,
        "primary_segment": primary_segment,
        "segment_col": segment_col,
        "segment_artifacts": enriched_segments,
        "metrics_df": _copy_if_dataframe(metrics_df),
        "cluster_profile": _copy_if_dataframe(cluster_profile),
        "cluster_profile_ranked": _copy_if_dataframe(cluster_profile_ranked),
        "cluster_medians": _copy_if_dataframe(cluster_medians),
        "feature_extremes": _copy_if_dataframe(feature_extremes),
        "industry_cluster_mix": _copy_if_dataframe(industry_cluster_mix),
        "scorecard_domain_weights": dict(SCORECARD_DOMAIN_WEIGHTS),
        "scorecard_cluster_features": list(SCORECARD_CLUSTER_FEATURES),
        "scorecard_component_features": list(SCORECARD_COMPONENT_FEATURES),
        "ratio_cols": list(RATIO_COLS),
        "interpret_features": list(INTERPRET_FEATURES),
        "winsor_caps": dict(winsor_caps) if winsor_caps is not None else None,
        "notes": notes or (
            "V3 scorecard EBITDA artifact. KMeans clusters use bounded "
            "domain-level credit risk factors; absolute size is diagnostic only."
        ),
    }

    # Backward-compatible aliases for single-segment Notebook 03 scoring.
    alias_keys = [
        "pipeline",
        "feature_cols",
        "cluster_labels",
        "risk_rank",
        "cluster_sizes",
        "cluster_profile",
        "cluster_profile_ranked",
        "availability",
        "min_non_null_features",
        "metrics",
    ]
    for key in alias_keys:
        if key in primary:
            artifact[key] = _copy_if_dataframe(primary[key])

    # Prefer the full ranked profile at top level when supplied.
    if cluster_profile_ranked is not None:
        artifact["cluster_profile_ranked"] = cluster_profile_ranked.copy()

    if extra_metadata:
        artifact.update(dict(extra_metadata))

    return artifact


def build_single_segment_artifact(
    segment_artifact: Mapping[str, Any],
    cluster_profile_ranked: pd.DataFrame | None = None,
    segment_name: str = DEFAULT_PRIMARY_SEGMENT,
    segment_col: str = DEFAULT_SEGMENT_COL,
    winsor_caps: Mapping[str, tuple[float, float]] | None = None,
    artifact_version: str = DEFAULT_ARTIFACT_VERSION,
    notes: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a compact single-segment artifact.

    Use this if Notebook 02 only trains one segment and you prefer the older,
    simple artifact shape consumed by Notebook 03.
    """
    artifact = enrich_segment_artifact(
        segment_artifact,
        cluster_profile_ranked=cluster_profile_ranked,
        segment_name=segment_name,
        segment_col=segment_col,
        winsor_caps=winsor_caps,
    )

    artifact["artifact_version"] = artifact_version
    artifact["primary_segment"] = segment_name
    artifact["notes"] = notes or (
        "Single-segment V3 scorecard EBITDA artifact. KMeans clusters use "
        "bounded domain-level credit risk factors."
    )

    if extra_metadata:
        artifact.update(dict(extra_metadata))

    return artifact


def validate_artifact(
    artifact: Mapping[str, Any],
    segment: str = DEFAULT_PRIMARY_SEGMENT,
    require_labels: bool = True,
) -> bool:
    """
    Validate that an artifact has the fields required by scoring.py.

    Raises informative errors when something important is missing.
    """
    seg = get_segment_artifact(artifact, segment=segment)

    required = ["pipeline", "feature_cols"]
    missing = [key for key in required if key not in seg or seg[key] is None]
    if missing:
        raise KeyError(f"Segment artifact is missing required keys: {missing}")

    if not seg["feature_cols"]:
        raise ValueError("Segment artifact has empty feature_cols.")

    if require_labels:
        missing_label_keys = [
            key
            for key in ["cluster_labels", "risk_rank"]
            if key not in seg or not seg.get(key)
        ]
        if missing_label_keys:
            raise KeyError(
                "Segment artifact is missing label/risk-rank mappings: "
                f"{missing_label_keys}. Run add_rating_style_labels() and "
                "build the artifact with cluster_profile_ranked."
            )

    return True


def save_artifact(
    artifact: Mapping[str, Any],
    path: str | Path,
    validate: bool = True,
    segment: str = DEFAULT_PRIMARY_SEGMENT,
) -> Path:
    """Validate and save an artifact with joblib."""
    if validate:
        validate_artifact(artifact, segment=segment, require_labels=False)

    path = _as_path(path)
    joblib.dump(dict(artifact), path)
    return path


def load_artifact(
    path: str | Path) -> dict[str, Any]:
    """Load a joblib artifact from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")

    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise TypeError(f"Expected artifact to be dict, got {type(artifact)!r}")

    return artifact


def summarize_artifact(
    artifact: Mapping[str, Any],
    segment: str = DEFAULT_PRIMARY_SEGMENT,
) -> dict[str, Any]:
    """Return a lightweight summary useful for Notebook 02 sanity checks."""
    seg = get_segment_artifact(artifact, segment=segment)

    cluster_sizes = seg.get("cluster_sizes")
    if isinstance(cluster_sizes, pd.DataFrame):
        if {"cluster", "issuer_years"}.issubset(cluster_sizes.columns):
            sizes = dict(
                zip(
                    cluster_sizes["cluster"].astype(int),
                    cluster_sizes["issuer_years"].astype(int),
                )
            )
        else:
            sizes = cluster_sizes.to_dict()
    else:
        sizes = cluster_sizes

    metrics = seg.get("metrics", {}) or {}

    return {
        "artifact_version": artifact.get("artifact_version"),
        "primary_segment": artifact.get("primary_segment", segment),
        "segment": seg.get("segment_name", segment),
        "feature_cols": list(seg.get("feature_cols", [])),
        "n_clusters": seg.get("n_clusters"),
        "cluster_labels": seg.get("cluster_labels"),
        "risk_rank": seg.get("risk_rank"),
        "cluster_sizes": sizes,
        "silhouette": metrics.get("silhouette"),
        "calinski_harabasz": metrics.get("calinski_harabasz"),
        "davies_bouldin": metrics.get("davies_bouldin"),
    }


__all__ = [
    "DEFAULT_ARTIFACT_VERSION",
    "DEFAULT_PRIMARY_SEGMENT",
    "DEFAULT_SEGMENT_COL",
    "get_segment_artifact",
    "enrich_segment_artifact",
    "build_credit_clustering_artifact",
    "build_single_segment_artifact",
    "validate_artifact",
    "save_artifact",
    "load_artifact",
    "summarize_artifact",
]
