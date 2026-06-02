"""
EDGAR concept and sector mapping utilities for the credit clustering project.

This module is intentionally EDGAR-specific. It maps noisy US-GAAP concept
names into the canonical financial columns consumed by features.py.

Notebook 02 should import these maps/helpers instead of defining CONCEPT_MAP,
SIC mapping functions, or issuer-year extraction SQL inline.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# EDGAR / US-GAAP concept map
# ---------------------------------------------------------------------

EDGAR_CONCEPT_MAP: dict[str, list[str]] = {
    # Balance sheet
    "assets": ["us-gaap:Assets"],
    "assets_current": ["us-gaap:AssetsCurrent"],
    "liabilities": ["us-gaap:Liabilities"],
    "liabilities_current": ["us-gaap:LiabilitiesCurrent"],
    "cash": [
        "us-gaap:CashAndCashEquivalentsAtCarryingValue",
        "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "us-gaap:CashAndDueFromBanks",
    ],
    "receivables": ["us-gaap:AccountsReceivableNetCurrent"],
    "inventory": ["us-gaap:InventoryNet"],
    "ppe": ["us-gaap:PropertyPlantAndEquipmentNet"],
    "goodwill": ["us-gaap:Goodwill"],
    "equity": [
        "us-gaap:StockholdersEquity",
        "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "long_term_debt": [
        "us-gaap:LongTermDebt",
        "us-gaap:LongTermDebtNoncurrent",
    ],
    "short_term_debt": [
        "us-gaap:ShortTermBorrowings",
        "us-gaap:LongTermDebtCurrent",
        "us-gaap:CurrentPortionOfLongTermDebt",
    ],

    # Income statement
    "revenue": [
        "us-gaap:Revenues",
        "us-gaap:SalesRevenueNet",
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    ],
    "depreciation_amortization": [
        "us-gaap:DepreciationDepletionAndAmortization",
        "us-gaap:DepreciationDepletionAndAmortizationExpense",
        "us-gaap:DepreciationAndAmortization",
        "us-gaap:Depreciation",
        "us-gaap:AmortizationOfIntangibleAssets",
    ],
    "gross_profit": ["us-gaap:GrossProfit"],
    "operating_income": ["us-gaap:OperatingIncomeLoss"],
    "net_income": ["us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss"],
    "interest_expense": [
        "us-gaap:InterestExpense",
        "us-gaap:InterestExpenseNonOperating",
    ],
    "sga": ["us-gaap:SellingGeneralAndAdministrativeExpense"],
    "rd": ["us-gaap:ResearchAndDevelopmentExpense"],

    # Cash flow
    "cfo": ["us-gaap:NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"],
}


# Backward-compatible alias for older notebook wording.
CONCEPT_MAP = EDGAR_CONCEPT_MAP


# ---------------------------------------------------------------------
# Sector / SIC mapping
# ---------------------------------------------------------------------

def map_sic_major_division(sic) -> str:
    """Map an SEC SIC code into a broad major-sector label."""
    if pd.isna(sic):
        return "Unknown"

    try:
        sic_int = int(float(sic))
    except Exception:
        return "Unknown"

    if 100 <= sic_int < 1000:
        return "Agriculture"
    if 1000 <= sic_int < 1500:
        return "Mining / Energy"
    if 1500 <= sic_int < 1800:
        return "Construction"
    if 2000 <= sic_int < 4000:
        return "Manufacturing / Industrials"
    if 4000 <= sic_int < 5000:
        return "Transportation / Utilities"
    if 5000 <= sic_int < 6000:
        return "Wholesale / Retail"
    if 6000 <= sic_int < 6800:
        return "Finance / Insurance / Real Estate"
    if 7000 <= sic_int < 9000:
        return "Services"
    if 9100 <= sic_int < 9730:
        return "Public Administration"

    return "Other"


def map_financial_flag(sic) -> str:
    """Return Financial / Non-financial / Unknown based on SIC code."""
    if pd.isna(sic):
        return "Unknown"

    try:
        sic_int = int(float(sic))
    except Exception:
        return "Unknown"

    return "Financial" if 6000 <= sic_int < 6800 else "Non-financial"


# ---------------------------------------------------------------------
# Concept lookup and issuer-year panel helpers
# ---------------------------------------------------------------------

def concept_lookup_frame(
    concept_map: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Return a dataframe mapping canonical_feature -> EDGAR concept."""
    concept_map = concept_map or EDGAR_CONCEPT_MAP

    rows = []
    for canonical_feature, concepts in concept_map.items():
        for concept in concepts:
            rows.append(
                {
                    "canonical_feature": canonical_feature,
                    "concept": concept,
                }
            )

    return pd.DataFrame(rows)


def ensure_concept_columns(
    df: pd.DataFrame,
    concept_map: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Ensure all canonical concept columns exist in a panel dataframe."""
    out = df.copy()
    concept_map = concept_map or EDGAR_CONCEPT_MAP

    for col in concept_map.keys():
        if col not in out.columns:
            out[col] = np.nan

    return out


def detect_numeric_value_column(schema: pd.DataFrame) -> str:
    """Detect the numeric fact-value column in the raw EDGAR facts schema."""
    cols = set(schema["column_name"].str.lower())

    if "numeric_value" in cols:
        return "numeric_value"
    if "value" in cols:
        return "value"

    raise ValueError("Could not find numeric_value or value column in raw_facts.")


def detect_sort_column(schema: pd.DataFrame) -> str | None:
    """Detect a useful filing/period sort column if available."""
    schema_cols = schema["column_name"].tolist()
    lower_to_actual = {col.lower(): col for col in schema_cols}
    sort_candidates = ["filing_date", "filed", "period_end", "end", "start"]

    return next(
        (lower_to_actual[col] for col in sort_candidates if col in lower_to_actual),
        None,
    )


def create_issuer_year_facts_table(
    con,
    schema: pd.DataFrame,
    concept_map: Mapping[str, Sequence[str]] | None = None,
    start_year: int = 2020,
    end_year: int = 2025,
    fiscal_period: str = "FY",
    table_name: str = "issuer_year_facts",
    raw_view_name: str = "raw_facts",
) -> tuple[pd.DataFrame, str, str | None]:
    """
    Create an issuer-year-canonical-feature facts table in DuckDB.

    Returns:
    - facts_summary dataframe
    - detected value column name
    - detected sort column name, if any
    """
    concept_lookup = concept_lookup_frame(concept_map)
    con.register("concept_lookup", concept_lookup)

    value_col = detect_numeric_value_column(schema)
    sort_col = detect_sort_column(schema)

    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT
            rf.ticker,
            TRY_CAST(rf.cik AS VARCHAR) AS cik,
            ANY_VALUE(rf.company_name) AS company_name,
            TRY_CAST(rf.sic AS INTEGER) AS sic,
            TRY_CAST(rf.fiscal_year AS INTEGER) AS fiscal_year,
            cl.canonical_feature,
            MEDIAN(TRY_CAST(rf.{value_col} AS DOUBLE)) AS value
        FROM {raw_view_name} rf
        JOIN concept_lookup cl
            ON rf.concept = cl.concept
        WHERE TRY_CAST(rf.fiscal_year AS INTEGER) BETWEEN {int(start_year)} AND {int(end_year)}
          AND rf.fiscal_period = '{fiscal_period}'
          AND TRY_CAST(rf.{value_col} AS DOUBLE) IS NOT NULL
        GROUP BY 1,2,4,5,6
        """
    )

    facts_summary = con.execute(
        f"""
        SELECT
            canonical_feature,
            COUNT(*) AS row_count,
            COUNT(DISTINCT ticker) AS ticker_count
        FROM {table_name}
        GROUP BY canonical_feature
        ORDER BY ticker_count DESC
        """
    ).df()

    return facts_summary, value_col, sort_col


def build_issuer_year_panel(
    con,
    concept_map: Mapping[str, Sequence[str]] | None = None,
    facts_table_name: str = "issuer_year_facts",
) -> pd.DataFrame:
    """
    Pivot issuer-year facts into one row per ticker-year and add sector labels.
    """
    panel = con.execute(
        f"""
        PIVOT {facts_table_name}
        ON canonical_feature
        USING MAX(value)
        GROUP BY ticker, cik, company_name, sic, fiscal_year
        """
    ).df()

    panel = ensure_concept_columns(panel, concept_map)
    panel["major_sector"] = panel["sic"].apply(map_sic_major_division)
    panel["financial_flag"] = panel["sic"].apply(map_financial_flag)

    return panel
