"""
Configuration constants for the credit clustering project.

This module is intentionally logic-free. It centralizes feature lists,
scorecard weights, thresholds, and reporting columns so Notebook 02,
Notebook 03, and src modules use the same definitions.
"""

# ---------------------------------------------------------------------
# Core model feature set
# ---------------------------------------------------------------------

SCORECARD_CLUSTER_FEATURES = [
    "structural_distress_risk",
    "earnings_risk",
    "operating_cashflow_risk",
    "liquidity_risk",
    "leverage_risk",
    "debt_service_risk",
]

SCORECARD_DOMAIN_WEIGHTS = {
    "leverage_risk": 0.25,
    "liquidity_risk": 0.20,
    "earnings_risk": 0.15,
    "operating_cashflow_risk": 0.20,
    "debt_service_risk": 0.15,
    "structural_distress_risk": 0.05,
}

# Low-level component features used to construct the six domain-level features.
SCORECARD_COMPONENT_FEATURES = [
    "liabilities_risk",
    "debt_load_risk",
    "equity_buffer_risk",
    "cash_buffer_risk",
    "current_liquidity_risk",
    "quick_liquidity_risk",
    "profitability_risk",
    "cashflow_risk",
    "coverage_risk",
    "fcf_risk",
    "ebitda_margin_risk",
    "debt_to_ebitda_risk",
    "net_debt_to_ebitda_risk",
    "ebitda_coverage_risk",
    "negative_ebitda_flag",
    "negative_equity_flag",
    "liabilities_exceed_assets_flag",
]


# ---------------------------------------------------------------------
# Denominator/materiality thresholds
# ---------------------------------------------------------------------

# Used for SME-compatible ratio construction. This avoids suppressing valid
# private-company ratios while still blocking meaningless tiny denominators.
SME_MIN_DENOMINATOR = 1_000

# Used as a training/materiality reference for public-company datasets.
PUBLIC_COMPANY_MIN_ASSETS = 1_000_000

# Size-band thresholds used for diagnostics only; not clustering features.
SMALL_COMPANY_ASSET_LIMIT = 5_000_000
MID_COMPANY_ASSET_LIMIT = 50_000_000
LARGE_COMPANY_ASSET_LIMIT = 100_000_000_000


# ---------------------------------------------------------------------
# Clustering defaults
# ---------------------------------------------------------------------

DEFAULT_SEGMENT_COL = "financial_flag"
DEFAULT_TARGET_SEGMENTS = ("Non-financial",)

DEFAULT_N_CLUSTERS = 5
DEFAULT_MIN_ROWS_PER_SEGMENT = 500

# Minimum number/share of non-null model features required per row.
DEFAULT_MIN_FEATURES = 4
DEFAULT_ROW_FEATURE_COVERAGE = 0.60

# Minimum non-null coverage required per feature.
# 0.0 means do not drop features based on global coverage.
DEFAULT_MIN_FEATURE_COVERAGE = 0.0

DEFAULT_RANDOM_STATE = 42
DEFAULT_N_INIT = 500


# ---------------------------------------------------------------------
# Profiling defaults
# ---------------------------------------------------------------------

DEFAULT_RATING_STYLE_LABELS = {
    1: "1 - Low risk / investment-grade-like",
    2: "2 - Moderate risk / lower-investment-grade-like",
    3: "3 - Elevated risk / leveraged",
    4: "4 - High risk / speculative",
    5: "5 - Distressed / near-default proxy",
}

DEFAULT_EXTREME_QUANTILES = (
    0.001,
    0.01,
    0.05,
    0.50,
    0.95,
    0.99,
    0.999,
)


# ---------------------------------------------------------------------
# Scoring defaults
# ---------------------------------------------------------------------

DEFAULT_SCORING_SEGMENT = "Non-financial"
DEFAULT_SCORING_TEMPERATURE = 1.0
DEFAULT_FX_TO_MODEL_CURRENCY = 1.0

# Use SME-compatible denominator handling for manual/private-company scoring.
DEFAULT_SCORING_MIN_DENOMINATOR = SME_MIN_DENOMINATOR


# ---------------------------------------------------------------------
# Artifact defaults
# ---------------------------------------------------------------------

DEFAULT_ARTIFACT_VERSION = "v3_scorecard_ebitda"
DEFAULT_PRIMARY_SEGMENT = "Non-financial"


# ---------------------------------------------------------------------
# Input column groups
# ---------------------------------------------------------------------

REQUIRED_OR_OPTIONAL_FINANCIAL_COLUMNS = [
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
    "depreciation_amortization",
    "ebitda",
]

MONETARY_COLUMNS = REQUIRED_OR_OPTIONAL_FINANCIAL_COLUMNS.copy()


# ---------------------------------------------------------------------
# Risk thresholds
# ---------------------------------------------------------------------

RISK_THRESHOLDS = {
    "liabilities_to_assets": {"low": 0.45, "high": 1.00},
    "debt_to_assets": {"low": 0.25, "high": 0.85},
    "equity_to_assets": {"good": 0.40, "bad": 0.00},
    "cash_to_assets": {"good": 0.10, "bad": 0.01},
    "current_ratio": {"good": 2.00, "bad": 0.75},
    "quick_ratio": {"good": 1.00, "bad": 0.25},
    "net_income_to_assets": {"good": 0.05, "bad": -0.05},
    "cfo_to_assets": {"good": 0.08, "bad": -0.03},
    "interest_coverage": {"good": 3.00, "bad": 1.00},
    "fcf_to_debt": {"good": 0.15, "bad": -0.10},
    "ebitda_margin": {"good": 0.20, "bad": 0.00},
    "debt_to_ebitda": {"low": 2.0, "high": 6.0},
    "net_debt_to_ebitda": {"low": 1.5, "high": 5.0},
    "ebitda_interest_coverage": {"good": 4.0, "bad": 1.5},
}

# ---------------------------------------------------------------------
# Diagnostic/reporting columns
# ---------------------------------------------------------------------

RATIO_COLS = [
    "log_assets",
    "liabilities_to_assets",
    "debt_to_assets",
    "debt_to_equity",
    "equity_to_assets",
    "cash_to_assets",
    "net_income_to_assets",
    "cfo_to_assets",
    "revenue_to_assets",
    "current_ratio",
    "quick_ratio",
    "interest_coverage",
    "fcf_to_debt",
    "operating_margin",
    "gross_margin",
    "cfo_to_liabilities",
    "capex_to_revenue",
    "total_debt",
    "net_debt",
    "ebitda",
    "ebitda_margin",
    "debt_to_ebitda",
    "net_debt_to_ebitda",
    "ebitda_interest_coverage",
    "leverage_risk",
    "liquidity_risk",
    "earnings_risk",
    "operating_cashflow_risk",
    "debt_service_risk",
    "debt_service_risk_legacy",
    "structural_distress_risk",
    "scorecard_credit_score",
]

INTERPRET_FEATURES = [
    "log_assets",
    "asset_size_band",
    "liabilities_to_assets",
    "debt_to_assets",
    "debt_to_equity",
    "equity_to_assets",
    "cash_to_assets",
    "net_income_to_assets",
    "cfo_to_assets",
    "revenue_to_assets",
    "current_ratio",
    "quick_ratio",
    "interest_coverage",
    "fcf_to_debt",
    "operating_margin",
    "gross_margin",
    "ebitda_margin",
    "debt_to_ebitda",
    "net_debt_to_ebitda",
    "ebitda_interest_coverage",
    "leverage_risk",
    "liquidity_risk",
    "earnings_risk",
    "operating_cashflow_risk",
    "debt_service_risk",
    "structural_distress_risk",
    "scorecard_credit_score",
]

SUMMARY_COLS = [
    "company_name",
    "fiscal_year",
    "assigned_cluster",
    "cluster_label",
    "risk_rank",
    "cluster_affinity",
    "near_default_affinity",
    "distance_to_assigned_cluster",
    "scorecard_credit_score",
    "feature_coverage_pct",
    "warning_flags",
]

SUMMARY_COLS_WITH_OUTLOOK = [
    "company_name",
    "fiscal_year",
    "assigned_cluster",
    "cluster_label",
    "risk_rank",
    "cluster_affinity",
    "near_default_affinity",
    "distance_to_assigned_cluster",
    "upper_bucket_cluster",
    "upper_bucket_label",
    "distance_to_upper_bucket",
    "lower_bucket_cluster",
    "lower_bucket_label",
    "distance_to_lower_bucket",
    "outlook_flag",
    "outlook_reason",
    "scorecard_credit_score",
    "feature_coverage_pct",
    "warning_flags",
]

SCENARIO_SUMMARY_COLS = [
    "scenario",
    "assigned_cluster",
    "cluster_label",
    "risk_rank",
    "cluster_affinity",
    "near_default_affinity",
    "distance_to_assigned_cluster",
    "upper_bucket_cluster",
    "upper_bucket_label",
    "distance_to_upper_bucket",
    "lower_bucket_cluster",
    "lower_bucket_label",
    "distance_to_lower_bucket",
    "outlook_flag",
    "outlook_reason",
    "scorecard_credit_score",
    "feature_coverage_pct",
    "warning_flags",
]
