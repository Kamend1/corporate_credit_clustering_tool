"""
Public API for the credit_clustering package.

Use explicit imports from this package in notebooks instead of importing from
many submodules or using wildcard imports.
"""

# Configuration
from .config import (
    SCORECARD_CLUSTER_FEATURES,
    SCORECARD_COMPONENT_FEATURES,
    SCORECARD_DOMAIN_WEIGHTS,
    RATIO_COLS,
    INTERPRET_FEATURES,
    SUMMARY_COLS,
    SUMMARY_COLS_WITH_OUTLOOK,
    SCENARIO_SUMMARY_COLS,
    SME_MIN_DENOMINATOR,
    PUBLIC_COMPANY_MIN_ASSETS,
    GUARDRAIL_COLS,

    # Clustering defaults
    DEFAULT_SEGMENT_COL,
    DEFAULT_TARGET_SEGMENTS,
    DEFAULT_N_CLUSTERS,
    DEFAULT_MIN_ROWS_PER_SEGMENT,
    DEFAULT_MIN_FEATURES,
    DEFAULT_ROW_FEATURE_COVERAGE,
    DEFAULT_MIN_FEATURE_COVERAGE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_N_INIT,

    #Scoring and profiling defaults
    DEFAULT_PRIMARY_SEGMENT,
    DEFAULT_SCORING_TEMPERATURE,
    DEFAULT_FX_TO_MODEL_CURRENCY,
    DEFAULT_SCORING_MIN_DENOMINATOR
)

# EDGAR concept / sector mapping
from .edgar_concepts import (
    EDGAR_CONCEPT_MAP,
    CONCEPT_MAP,
    concept_lookup_frame,
    ensure_concept_columns,
    detect_numeric_value_column,
    detect_sort_column,
    create_issuer_year_facts_table,
    build_issuer_year_panel,
    map_sic_major_division,
    map_financial_flag,
)

# Feature engineering
from .features import (
    engineer_private_company_features,
    safe_divide,
)

# Clustering / training
from .clustering import (
    cluster_segments,
    evaluate_segments_k_range,
)

# Profiling / interpretation
from .profiling import (
    build_cluster_profile,
    build_cluster_medians,
    build_feature_extremes,
    build_industry_cluster_mix,
    add_rating_style_labels,
    build_rating_label_maps,
    representatives,
    merge_profile_with_representatives,
)

# Artifacts
from .artifacts import (
    build_credit_clustering_artifact,
    build_single_segment_artifact,
    get_segment_artifact,
    validate_artifact,
    save_artifact,
    load_artifact,
    summarize_artifact,
)

# Scoring / diagnostics
from .scoring import (
    infer_near_default_cluster_from_artifact,
    score_companies,
    make_warning_flags,
)

from .diagnostics import (
    add_adjacent_bucket_distances_and_outlook,
    compare_to_cluster_profile,
    make_scenarios,
)

from .guardrails import (
    apply_credit_guardrails,
)
