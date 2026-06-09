"""
Alternative clustering utilities for the corporate credit clustering project.

This module keeps Notebook 04 clean. It provides reusable helpers for loading
Notebook 02 outputs, defining unfitted alternative clustering pipelines,
running Agglomerative and DBSCAN grids, computing internal metrics, comparing
labels against the selected KMeans model, and preparing PCA visualization data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class PreparedMatrix:
    """Container for the feature matrix used in alternative clustering."""

    X: np.ndarray
    feature_cols: list[str]
    row_index: pd.Index
    imputer: SimpleImputer


def load_table(path: Path, required_cols: Sequence[str] | None = None, label: str = "table") -> pd.DataFrame:
    """Load a CSV or Parquet table from an explicit path and validate columns."""
    required_cols = list(required_cols or [])
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type for {label}: {suffix}. Expected .csv or .parquet.")

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}\n"
            f"Loaded file: {path}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    return df


def validate_required_paths(required_paths: Mapping[str, Path]) -> None:
    """Raise a clear error if any required path does not exist."""
    missing = {name: Path(path) for name, path in required_paths.items() if not Path(path).exists()}
    if missing:
        raise FileNotFoundError(
            "Missing required project artifacts:\n"
            + "\n".join(f"- {name}: {path}" for name, path in missing.items())
        )


def prepare_analysis_frame(
    df: pd.DataFrame,
    segment_col: str,
    target_segment: str,
    feature_cols: Sequence[str],
    baseline_cluster_col: str = "cluster",
    max_rows: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Filter the clustered panel to the target segment and optionally sample rows."""
    required = [segment_col, baseline_cluster_col, *feature_cols]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    out = df.loc[df[segment_col].astype(str) == str(target_segment)].copy()

    if out.empty:
        raise ValueError(f"No rows found for target segment: {target_segment}")

    if max_rows is not None and len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=random_state).copy()

    out = out.reset_index(drop=True)
    return out


def make_prepared_matrix(df: pd.DataFrame, feature_cols: Sequence[str]) -> PreparedMatrix:
    """Median-impute the selected feature columns and return a numeric matrix."""
    feature_cols = list(feature_cols)
    X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_df)

    return PreparedMatrix(
        X=X,
        feature_cols=feature_cols,
        row_index=df.index,
        imputer=imputer,
    )


def define_alternative_clustering_pipelines(
    agglomerative_k_values: Iterable[int],
    dbscan_eps_values: Iterable[float],
    dbscan_min_samples_values: Iterable[int],
    linkage: str = "ward",
) -> dict[str, Pipeline]:
    """Return unfitted pipelines for teacher-visible model definitions."""
    pipelines: dict[str, Pipeline] = {}

    for k in agglomerative_k_values:
        pipelines[f"agglomerative_k_{k}"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("clusterer", AgglomerativeClustering(n_clusters=int(k), linkage=linkage)),
            ]
        )

    for eps in dbscan_eps_values:
        for min_samples in dbscan_min_samples_values:
            pipelines[f"dbscan_eps_{eps}_min_{min_samples}"] = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clusterer", DBSCAN(eps=float(eps), min_samples=int(min_samples))),
                ]
            )

    return pipelines


def _valid_metric_labels(labels: np.ndarray) -> bool:
    unique = np.unique(labels)
    return len(unique) >= 2 and len(unique) < len(labels)


def safe_internal_metrics(X: np.ndarray, labels: Sequence[int]) -> dict[str, float]:
    """Compute internal clustering metrics safely; return NaN when invalid."""
    labels = np.asarray(labels)

    if not _valid_metric_labels(labels):
        return {
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }

    return {
        "silhouette": float(silhouette_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }


def dbscan_cluster_diagnostics(labels: Sequence[int]) -> dict[str, float | int]:
    """Return DBSCAN-specific diagnostics including noise and dominant cluster shares."""
    labels = np.asarray(labels)
    n = len(labels)
    non_noise = labels != -1
    labels_non_noise = labels[non_noise]
    unique_non_noise = np.unique(labels_non_noise)

    noise_share = float((labels == -1).mean()) if n else np.nan

    if labels_non_noise.size:
        counts = pd.Series(labels_non_noise).value_counts(normalize=True)
        largest_cluster_share = float(counts.max())
    else:
        largest_cluster_share = np.nan

    return {
        "clusters_ex_noise": int(len(unique_non_noise)),
        "noise_share": noise_share,
        "largest_cluster_share": largest_cluster_share,
    }


def evaluate_agglomerative_grid(
    X: np.ndarray,
    k_values: Iterable[int],
    linkage: str = "ward",
    baseline_labels: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Fit AgglomerativeClustering for each k and return metrics plus labels."""
    rows: list[dict] = []
    labels_by_k: dict[int, np.ndarray] = {}

    for k in k_values:
        model = AgglomerativeClustering(n_clusters=int(k), linkage=linkage)
        labels = model.fit_predict(X)
        labels_by_k[int(k)] = labels

        metrics = safe_internal_metrics(X, labels)
        row = {
            "method": "Agglomerative",
            "k": int(k),
            "linkage": linkage,
            "clusters": int(len(np.unique(labels))),
            **metrics,
        }

        if baseline_labels is not None:
            row["ari_vs_kmeans"] = float(adjusted_rand_score(baseline_labels, labels))
            row["nmi_vs_kmeans"] = float(normalized_mutual_info_score(baseline_labels, labels))

        rows.append(row)

    return pd.DataFrame(rows), labels_by_k


def evaluate_dbscan_grid(
    X: np.ndarray,
    eps_values: Iterable[float],
    min_samples_values: Iterable[int],
    baseline_labels: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, dict[tuple[float, int], np.ndarray]]:
    """Fit DBSCAN over eps/min_samples grid and return metrics plus labels."""
    rows: list[dict] = []
    labels_by_params: dict[tuple[float, int], np.ndarray] = {}

    for eps in eps_values:
        for min_samples in min_samples_values:
            eps = float(eps)
            min_samples = int(min_samples)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)
            labels_by_params[(eps, min_samples)] = labels

            diag = dbscan_cluster_diagnostics(labels)

            non_noise = labels != -1
            if diag["clusters_ex_noise"] >= 2 and non_noise.sum() >= 2:
                metrics = safe_internal_metrics(X[non_noise], labels[non_noise])
            else:
                metrics = {
                    "silhouette": np.nan,
                    "calinski_harabasz": np.nan,
                    "davies_bouldin": np.nan,
                }

            row = {
                "method": "DBSCAN",
                "eps": eps,
                "min_samples": min_samples,
                **diag,
                **metrics,
            }

            if baseline_labels is not None:
                row["ari_vs_kmeans"] = float(adjusted_rand_score(baseline_labels, labels))
                row["nmi_vs_kmeans"] = float(normalized_mutual_info_score(baseline_labels, labels))

            rows.append(row)

    return pd.DataFrame(rows), labels_by_params


def select_practical_dbscan_candidate(
    dbscan_results: pd.DataFrame,
    max_noise_share: float = 0.50,
    max_largest_cluster_share: float = 0.90,
    min_clusters: int = 2,
) -> pd.Series | None:
    """Select a practical DBSCAN candidate from grid results."""
    if dbscan_results.empty:
        return None

    candidates = dbscan_results.copy()
    candidates = candidates[
        (candidates["clusters_ex_noise"] >= min_clusters)
        & (candidates["noise_share"] <= max_noise_share)
        & (candidates["largest_cluster_share"] <= max_largest_cluster_share)
    ]

    if candidates.empty:
        return None

    sort_cols = ["silhouette", "ari_vs_kmeans", "clusters_ex_noise"]
    available_sort_cols = [col for col in sort_cols if col in candidates.columns]
    return candidates.sort_values(available_sort_cols, ascending=[False] * len(available_sort_cols)).iloc[0]


def profile_by_label(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: Sequence[str],
    extra_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Create median profile by an arbitrary cluster/label column."""
    cols = list(feature_cols) + list(extra_cols or [])
    valid_cols = [col for col in cols if col in df.columns]

    profile = (
        df.groupby(label_col, dropna=False)
        .agg(rows=(label_col, "size"), **{f"median_{col}": (col, "median") for col in valid_cols})
        .reset_index()
    )

    return profile


def make_label_crosstab(
    reference_labels: Sequence[int],
    alternative_labels: Sequence[int],
    reference_name: str = "kmeans_cluster",
    alternative_name: str = "alternative_cluster",
    normalize: str | None = "index",
) -> pd.DataFrame:
    """Create a crosstab comparing baseline KMeans labels to alternative labels."""
    return pd.crosstab(
        pd.Series(reference_labels, name=reference_name),
        pd.Series(alternative_labels, name=alternative_name),
        normalize=normalize,
    )


def make_pca_projection(
    X: np.ndarray,
    df: pd.DataFrame,
    label_cols: Sequence[str],
    n_components: int = 2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, PCA]:
    """Create PCA projection dataframe for visualization only."""
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3 for notebook visualization.")

    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(X)

    out = pd.DataFrame(
        coords,
        columns=[f"PC{i}" for i in range(1, n_components + 1)],
        index=df.index,
    )

    for col in label_cols:
        if col in df.columns:
            out[col] = df[col].values

    out["explained_variance_ratio_sum"] = float(pca.explained_variance_ratio_.sum())
    return out.reset_index(drop=True), pca


def build_method_comparison_table(
    kmeans_metrics: Mapping[str, float],
    best_agglo_row: pd.Series | Mapping | None,
    best_dbscan_row: pd.Series | Mapping | None,
) -> pd.DataFrame:
    """Build compact comparison table for the notebook conclusion."""
    rows: list[dict] = [
        {
            "method": "KMeans selected v3",
            "selected_role": "Production model",
            "clusters": kmeans_metrics.get("clusters", 5),
            "silhouette": kmeans_metrics.get("silhouette", np.nan),
            "calinski_harabasz": kmeans_metrics.get("calinski_harabasz", np.nan),
            "davies_bouldin": kmeans_metrics.get("davies_bouldin", np.nan),
            "ari_vs_kmeans": 1.0,
            "nmi_vs_kmeans": 1.0,
            "comment": "Selected for stable five-level credit-risk ladder and reporting compatibility.",
        }
    ]

    if best_agglo_row is not None:
        row = dict(best_agglo_row)
        rows.append(
            {
                "method": "Agglomerative best candidate",
                "selected_role": "Robustness check",
                "clusters": row.get("clusters", row.get("k", np.nan)),
                "silhouette": row.get("silhouette", np.nan),
                "calinski_harabasz": row.get("calinski_harabasz", np.nan),
                "davies_bouldin": row.get("davies_bouldin", np.nan),
                "ari_vs_kmeans": row.get("ari_vs_kmeans", np.nan),
                "nmi_vs_kmeans": row.get("nmi_vs_kmeans", np.nan),
                "comment": "Useful hierarchy check; not selected because KMeans supports centroid-based scoring.",
            }
        )

    if best_dbscan_row is not None:
        row = dict(best_dbscan_row)
        rows.append(
            {
                "method": "DBSCAN practical candidate",
                "selected_role": "Density/outlier diagnostic",
                "clusters": row.get("clusters_ex_noise", np.nan),
                "silhouette": row.get("silhouette", np.nan),
                "calinski_harabasz": row.get("calinski_harabasz", np.nan),
                "davies_bouldin": row.get("davies_bouldin", np.nan),
                "ari_vs_kmeans": row.get("ari_vs_kmeans", np.nan),
                "nmi_vs_kmeans": row.get("nmi_vs_kmeans", np.nan),
                "comment": "Useful outlier check; not ideal for ordered credit-risk labels.",
            }
        )

    return pd.DataFrame(rows)
