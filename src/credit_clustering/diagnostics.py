import numpy as np
import pandas as pd


def _get_cluster_risk_map(artifact):
    """
    Return {cluster_id: risk_rank}.

    Uses artifact["risk_rank"] first, then falls back to
    artifact["cluster_profile_ranked"].
    """

    if "risk_rank" in artifact and artifact["risk_rank"] is not None:
        risk_rank = artifact["risk_rank"]

        if isinstance(risk_rank, dict):
            return {int(k): int(v) for k, v in risk_rank.items()}

        if isinstance(risk_rank, pd.Series):
            return {int(k): int(v) for k, v in risk_rank.to_dict().items()}

    if "cluster_profile_ranked" in artifact:
        profile = artifact["cluster_profile_ranked"]

        if {"cluster", "risk_rank"}.issubset(profile.columns):
            return {
                int(row["cluster"]): int(row["risk_rank"])
                for _, row in profile[["cluster", "risk_rank"]].dropna().iterrows()
            }

    raise KeyError(
        "Could not infer cluster risk ranks. Expected artifact['risk_rank'] "
        "or artifact['cluster_profile_ranked'] with columns ['cluster', 'risk_rank']."
    )


def _get_cluster_label_map(artifact):
    """
    Return {cluster_id: cluster_label}, if available.
    """

    labels = artifact.get("cluster_labels")

    if labels is None:
        return {}

    if isinstance(labels, dict):
        return {int(k): v for k, v in labels.items()}

    return {}


def _distance_matrix_to_clusters(scored_df, artifact):
    """
    Compute distance from each scored row to each KMeans cluster center.

    Assumes:
    - artifact["pipeline"] is the fitted sklearn Pipeline used for KMeans.
    - artifact["feature_cols"] are the model input features.

    For a sklearn Pipeline ending in KMeans, pipeline.transform(X) returns
    distances to all cluster centers.
    """

    if "pipeline" not in artifact:
        raise KeyError("artifact['pipeline'] not found.")

    if "feature_cols" not in artifact:
        raise KeyError("artifact['feature_cols'] not found.")

    pipeline = artifact["pipeline"]
    feature_cols = artifact["feature_cols"]

    missing_features = [col for col in feature_cols if col not in scored_df.columns]

    if missing_features:
        raise KeyError(
            "The scored dataframe is missing model feature columns required "
            f"for distance calculation: {missing_features}"
        )

    X = scored_df[feature_cols].copy()
    distances = pipeline.transform(X)

    distance_df = pd.DataFrame(
        distances,
        index=scored_df.index,
        columns=[
            f"distance_to_cluster_{i}"
            for i in range(distances.shape[1])
        ],
    )

    return distance_df


def add_adjacent_bucket_distances_and_outlook(
    scored_df,
    artifact,
    neutral_band=0.15,
    upgrade_boundary_multiplier=1.35,
    downgrade_boundary_multiplier=1.35,
):
    """
    Adds adjacent bucket distance diagnostics and a cluster-position outlook.

    Added columns:
    - upper_bucket_cluster
    - upper_bucket_label
    - distance_to_upper_bucket
    - lower_bucket_cluster
    - lower_bucket_label
    - distance_to_lower_bucket
    - upper_distance_ratio_to_assigned
    - lower_distance_ratio_to_assigned
    - outlook_flag
    - outlook_reason

    Interpretation:
    - Positive:
        Company is closer to the stronger adjacent bucket and close enough
        to make the upgrade signal meaningful.

    - Neutral:
        Company remains materially closer to its assigned bucket, or the
        difference between adjacent buckets is not strong enough.

    - Negative:
        Company is closer to the weaker adjacent bucket and close enough
        to make the downgrade signal meaningful.

    Important:
    This is a relative cluster-position flag, not a time-series forecast.
    """

    scored_df = scored_df.copy().reset_index(drop=True)

    required_cols = [
        "assigned_cluster",
        "risk_rank",
        "distance_to_assigned_cluster",
    ]

    missing_required = [
        col for col in required_cols
        if col not in scored_df.columns
    ]

    if missing_required:
        raise KeyError(
            "scored_df is missing required columns: "
            f"{missing_required}"
        )

    risk_map = _get_cluster_risk_map(artifact)
    label_map = _get_cluster_label_map(artifact)
    distance_df = _distance_matrix_to_clusters(scored_df, artifact)

    rank_to_cluster = {
        int(rank): int(cluster)
        for cluster, rank in risk_map.items()
    }

    upper_clusters = []
    lower_clusters = []
    upper_labels = []
    lower_labels = []
    upper_distances = []
    lower_distances = []
    upper_distance_ratios = []
    lower_distance_ratios = []
    outlook_flags = []
    outlook_reasons = []

    for idx, row in scored_df.iterrows():
        assigned_rank = int(row["risk_rank"])

        assigned_distance = pd.to_numeric(
            row["distance_to_assigned_cluster"],
            errors="coerce",
        )

        upper_rank = assigned_rank - 1
        lower_rank = assigned_rank + 1

        upper_cluster = rank_to_cluster.get(upper_rank, np.nan)
        lower_cluster = rank_to_cluster.get(lower_rank, np.nan)

        if pd.notna(upper_cluster):
            upper_cluster = int(upper_cluster)
            upper_distance = distance_df.loc[
                idx,
                f"distance_to_cluster_{upper_cluster}",
            ]
            upper_label = label_map.get(upper_cluster, None)
        else:
            upper_distance = np.nan
            upper_label = None

        if pd.notna(lower_cluster):
            lower_cluster = int(lower_cluster)
            lower_distance = distance_df.loc[
                idx,
                f"distance_to_cluster_{lower_cluster}",
            ]
            lower_label = label_map.get(lower_cluster, None)
        else:
            lower_distance = np.nan
            lower_label = None

        if pd.notna(assigned_distance) and assigned_distance > 0:
            upper_ratio = (
                upper_distance / assigned_distance
                if pd.notna(upper_distance)
                else np.nan
            )

            lower_ratio = (
                lower_distance / assigned_distance
                if pd.notna(lower_distance)
                else np.nan
            )
        else:
            upper_ratio = np.nan
            lower_ratio = np.nan

        if pd.isna(assigned_distance):
            outlook = "Neutral"
            reason = (
                "Assigned-cluster distance is missing; outlook cannot be "
                "reliably assessed."
            )

        elif pd.isna(upper_distance) and pd.isna(lower_distance):
            outlook = "Neutral"
            reason = "No adjacent risk buckets available."

        elif pd.isna(upper_distance):
            near_lower = (
                lower_distance
                <= assigned_distance * downgrade_boundary_multiplier
            )

            if near_lower:
                outlook = "Negative"
                reason = (
                    "Company is near the weaker adjacent bucket and has no "
                    "stronger adjacent bucket available."
                )
            else:
                outlook = "Neutral"
                reason = (
                    "Only the weaker adjacent bucket is available, but the "
                    "company remains materially closer to its assigned bucket."
                )

        elif pd.isna(lower_distance):
            near_upper = (
                upper_distance
                <= assigned_distance * upgrade_boundary_multiplier
            )

            if near_upper:
                outlook = "Positive"
                reason = (
                    "Company is near the stronger adjacent bucket and has no "
                    "weaker adjacent bucket available."
                )
            else:
                outlook = "Neutral"
                reason = (
                    "Only the stronger adjacent bucket is available, but the "
                    "company remains materially closer to its assigned bucket."
                )

        else:
            diff = lower_distance - upper_distance
            threshold = assigned_distance * neutral_band

            near_upper = (
                upper_distance
                <= assigned_distance * upgrade_boundary_multiplier
            )

            near_lower = (
                lower_distance
                <= assigned_distance * downgrade_boundary_multiplier
            )

            if diff > threshold and near_upper:
                outlook = "Positive"
                reason = (
                    "Company is closer to the stronger adjacent bucket and "
                    "close enough to that bucket to indicate positive "
                    "cluster-position outlook."
                )

            elif diff < -threshold and near_lower:
                outlook = "Negative"
                reason = (
                    "Company is closer to the weaker adjacent bucket and "
                    "close enough to that bucket to indicate negative "
                    "cluster-position outlook."
                )

            else:
                outlook = "Neutral"
                reason = (
                    "Company remains materially closer to its assigned bucket "
                    "than to adjacent buckets; no strong upgrade or downgrade "
                    "signal."
                )

        upper_clusters.append(upper_cluster)
        lower_clusters.append(lower_cluster)
        upper_labels.append(upper_label)
        lower_labels.append(lower_label)
        upper_distances.append(upper_distance)
        lower_distances.append(lower_distance)
        upper_distance_ratios.append(upper_ratio)
        lower_distance_ratios.append(lower_ratio)
        outlook_flags.append(outlook)
        outlook_reasons.append(reason)

    scored_df["upper_bucket_cluster"] = upper_clusters
    scored_df["upper_bucket_label"] = upper_labels
    scored_df["distance_to_upper_bucket"] = upper_distances

    scored_df["lower_bucket_cluster"] = lower_clusters
    scored_df["lower_bucket_label"] = lower_labels
    scored_df["distance_to_lower_bucket"] = lower_distances

    scored_df["upper_distance_ratio_to_assigned"] = upper_distance_ratios
    scored_df["lower_distance_ratio_to_assigned"] = lower_distance_ratios

    scored_df["outlook_flag"] = outlook_flags
    scored_df["outlook_reason"] = outlook_reasons

    return scored_df


def compare_to_cluster_profile(scored_row, artifact):
    """
    Compare one scored company row against the median profile
    of its assigned cluster.

    Expected artifact structure:
    - artifact["cluster_profile_ranked"] contains the cluster profile.
    - scored_row["assigned_cluster"] contains the assigned cluster id.

    Handles cluster profiles with either:
    - raw metric names, e.g. "log_assets"
    - median-prefixed names, e.g. "median_log_assets"
    """

    if "cluster_profile_ranked" not in artifact:
        raise KeyError(
            "artifact['cluster_profile_ranked'] not found. "
            f"Available artifact keys: {list(artifact.keys())}"
        )

    profile = artifact["cluster_profile_ranked"]

    if profile is None or len(profile) == 0:
        raise ValueError("artifact['cluster_profile_ranked'] is empty.")

    if "assigned_cluster" not in scored_row.index:
        raise KeyError("scored_row must contain 'assigned_cluster'.")

    cluster = int(scored_row["assigned_cluster"])

    if "cluster" in profile.columns:
        match = profile.loc[
            profile["cluster"].astype(int).eq(cluster)
        ]

        if match.empty:
            available_clusters = sorted(
                profile["cluster"]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )

            raise KeyError(
                f"Cluster {cluster} not found in cluster_profile_ranked. "
                f"Available clusters: {available_clusters}"
            )

        cluster_profile = match.iloc[0]

    elif cluster in profile.index:
        cluster_profile = profile.loc[cluster]

    else:
        raise KeyError(
            f"Cluster {cluster} not found in cluster_profile_ranked index, "
            "and no 'cluster' column exists."
        )

    base_cols = [
        "log_assets",
        "liabilities_to_assets",
        "debt_to_assets",
        "equity_to_assets",
        "current_ratio",
        "quick_ratio",
        "cash_to_assets",
        "net_income_to_assets",
        "cfo_to_assets",
        "revenue_to_assets",
        "interest_coverage",
        "fcf_to_debt",
        "operating_margin",
        "gross_margin",
        "ebitda_margin",
        "debt_to_ebitda",
        "net_debt_to_ebitda",
        "ebitda_interest_coverage",
    ]

    rows = []

    for col in base_cols:
        if col not in scored_row.index:
            continue

        if col in cluster_profile.index:
            cluster_value = cluster_profile[col]
        elif f"median_{col}" in cluster_profile.index:
            cluster_value = cluster_profile[f"median_{col}"]
        else:
            continue

        company_value = pd.to_numeric(scored_row[col], errors="coerce")
        cluster_value = pd.to_numeric(cluster_value, errors="coerce")

        rows.append(
            {
                "metric": col,
                "company_value": company_value,
                "assigned_cluster_median": cluster_value,
                "difference": company_value - cluster_value,
            }
        )

    if not rows:
        raise ValueError(
            "No comparable metrics found. "
            "Check scored_row columns and artifact['cluster_profile_ranked'] columns."
        )

    comparison = pd.DataFrame(rows).set_index("metric")

    comparison["relative_position"] = np.select(
        [
            comparison["difference"] > 0,
            comparison["difference"] < 0,
        ],
        [
            "above_cluster_median",
            "below_cluster_median",
        ],
        default="equal_to_cluster_median",
    )

    return comparison.round(4)


def make_scenarios(base_row):
    base = base_row.copy()
    scenarios = []

    def add_case(name, updates):
        row = base.copy()
        row.update(updates)
        row["scenario"] = name
        scenarios.append(row)

    add_case("base", {})

    add_case(
        "revenue_down_15pct",
        {
            "revenue": base.get("revenue", np.nan) * 0.85,
            "net_income": base.get("net_income", np.nan) * 0.70,
            "cfo": base.get("cfo", np.nan) * 0.75,
            "operating_income": base.get("operating_income", np.nan) * 0.70,
        },
    )

    add_case(
        "debt_up_25pct",
        {
            "long_term_debt": base.get("long_term_debt", 0) * 1.25,
            "liabilities": (
                base.get("liabilities", np.nan)
                + base.get("long_term_debt", 0) * 0.25
            ),
            "cash": max(
                base.get("cash", 0)
                - base.get("long_term_debt", 0) * 0.05,
                0,
            ),
        },
    )

    add_case(
        "cash_burn_case",
        {
            "cash": max(base.get("cash", 0) * 0.50, 0),
            "cfo": -abs(base.get("cfo", 0)),
            "net_income": -abs(base.get("net_income", 0)),
        },
    )

    add_case(
        "near_default_stress",
        {
            "liabilities": base.get("assets", np.nan) * 1.10,
            "equity": -base.get("assets", np.nan) * 0.10,
            "long_term_debt": base.get("assets", np.nan) * 0.75,
            "short_term_debt": base.get("assets", np.nan) * 0.15,
            "interest_expense": max(
                base.get("interest_expense", 0),
                1_000_000,
            ),
            "operating_income": max(
                base.get("interest_expense", 1_000_000) * 0.4,
                0,
            ),
        },
    )

    return pd.DataFrame(scenarios)