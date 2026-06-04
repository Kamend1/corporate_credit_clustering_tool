"""
Excel/CSV credit report utilities for the private-company credit scoring tool.

This module is presentation/export plumbing only. It does not score companies,
engineer features, train clusters, or modify model artifacts. It consumes the
already prepared Notebook 03 outputs and produces clean report tables plus
CSV/XLSX outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from src.credit_clustering.config import (
    GUARDRAIL_COLS,
    RATIO_COLS,
    SCENARIO_SUMMARY_COLS,
    SUMMARY_COLS_WITH_OUTLOOK,
)


DEFAULT_SCENARIO_INPUT_COLS = [
    "scenario",
    "assets",
    "liabilities",
    "equity",
    "cash",
    "revenue",
    "net_income",
    "cfo",
    "long_term_debt",
    "short_term_debt",
    "interest_expense",
    "operating_income",
    "depreciation_amortization",
    "ebitda",
]


GUARDRAIL_CONTEXT_COLS = [
    "scope",
    "scenario",
    "company_name",
    "fiscal_year",
    "assigned_cluster",
    "cluster_label",
    "risk_rank",
    "scorecard_credit_score",
    "outlook_flag",
    "warning_flags",
]


def _unique_existing_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> list[str]:
    """Return existing columns from df, preserving order and removing duplicates."""
    out: list[str] = []

    for col in columns:
        if col in df.columns and col not in out:
            out.append(col)

    return out


def _reset_comparison_index(comparison_to_cluster: pd.DataFrame) -> pd.DataFrame:
    """Convert cluster comparison table into export-friendly flat format."""
    comparison = comparison_to_cluster.reset_index().copy()

    if "index" in comparison.columns:
        comparison = comparison.rename(columns={"index": "metric"})

    return comparison


def _build_guardrail_table(
    scored_manual_with_outlook: pd.DataFrame,
    scored_scenarios: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Build a dedicated guardrail table for Excel/PDF auditability.

    The output combines the base case and scenario guardrail results into one
    flat table. It is deliberately report-facing and does not change the scored
    dataframes.
    """
    frames: list[pd.DataFrame] = []

    if scored_manual_with_outlook is not None and not scored_manual_with_outlook.empty:
        base = scored_manual_with_outlook.copy()
        base.insert(0, "scope", "base_case")
        if "scenario" not in base.columns:
            base.insert(1, "scenario", "base")
        frames.append(base)

    if scored_scenarios is not None and not scored_scenarios.empty:
        scenarios = scored_scenarios.copy()
        scenarios.insert(0, "scope", "scenario")
        if "company_name" not in scenarios.columns and frames:
            company_name = frames[0].iloc[0].get("company_name")
            scenarios["company_name"] = company_name
        if "fiscal_year" not in scenarios.columns and frames:
            fiscal_year = frames[0].iloc[0].get("fiscal_year")
            scenarios["fiscal_year"] = fiscal_year
        frames.append(scenarios)

    if not frames:
        return pd.DataFrame(columns=GUARDRAIL_CONTEXT_COLS + list(GUARDRAIL_COLS))

    combined = pd.concat(frames, ignore_index=True, sort=False)

    guardrail_cols = _unique_existing_columns(
        combined,
        GUARDRAIL_CONTEXT_COLS + list(GUARDRAIL_COLS),
    )

    return combined[guardrail_cols].copy()


def build_credit_report_tables(
    scored_manual_with_outlook: pd.DataFrame,
    scored_manual: pd.DataFrame,
    comparison_to_cluster: pd.DataFrame,
    scenario_input: pd.DataFrame,
    scored_scenarios: pd.DataFrame,
    artifact: Mapping | None = None,
    summary_cols_with_outlook: Sequence[str] | None = None,
    ratio_cols: Sequence[str] | None = None,
    scenario_summary_cols: Sequence[str] | None = None,
    scenario_input_cols: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Build all report tables used by Notebook 03.

    Returns
    -------
    dict[str, pd.DataFrame]
        Tables:
        - score_summary
        - company_ratios
        - cluster_comparison
        - scored_file
        - scenario_input_snapshot
        - scenario_score_summary
        - scenario_ratios
        - scenario_file
        - guardrails
    """

    artifact = artifact or {}

    summary_cols_with_outlook = list(
        summary_cols_with_outlook
        or artifact.get("summary_cols_with_outlook", SUMMARY_COLS_WITH_OUTLOOK)
    )

    ratio_cols = list(
        ratio_cols
        or artifact.get("ratio_cols", RATIO_COLS)
    )

    scenario_summary_cols = list(
        scenario_summary_cols
        or artifact.get("scenario_summary_cols", SCENARIO_SUMMARY_COLS)
    )

    scenario_input_cols = list(
        scenario_input_cols
        or DEFAULT_SCENARIO_INPUT_COLS
    )

    # ------------------------------------------------------------------
    # Base-case score summary
    # ------------------------------------------------------------------

    existing_summary_cols = _unique_existing_columns(
        scored_manual_with_outlook,
        summary_cols_with_outlook,
    )

    score_summary = scored_manual_with_outlook[
        existing_summary_cols
    ].copy()

    # ------------------------------------------------------------------
    # Base-case ratio/diagnostic snapshot
    # ------------------------------------------------------------------

    company_ratio_cols = _unique_existing_columns(
        scored_manual,
        ["company_name"] + ratio_cols,
    )

    company_ratios = scored_manual[
        company_ratio_cols
    ].copy()

    # ------------------------------------------------------------------
    # Cluster comparison
    # ------------------------------------------------------------------

    cluster_comparison = _reset_comparison_index(
        comparison_to_cluster
    )

    # ------------------------------------------------------------------
    # Flat base-case CSV output
    # ------------------------------------------------------------------

    scored_file = scored_manual_with_outlook.merge(
        company_ratios,
        on="company_name",
        how="left",
    )

    base_output_cols = _unique_existing_columns(
        scored_file,
        list(score_summary.columns) + list(company_ratios.columns),
    )

    scored_file = scored_file[
        base_output_cols
    ].copy()

    # ------------------------------------------------------------------
    # Scenario inputs
    # ------------------------------------------------------------------

    scenario_input_snapshot_cols = _unique_existing_columns(
        scenario_input,
        scenario_input_cols,
    )

    scenario_input_snapshot = scenario_input[
        scenario_input_snapshot_cols
    ].copy()

    # ------------------------------------------------------------------
    # Scenario score summary
    # ------------------------------------------------------------------

    existing_scenario_summary_cols = _unique_existing_columns(
        scored_scenarios,
        scenario_summary_cols,
    )

    scenario_score_summary = scored_scenarios[
        existing_scenario_summary_cols
    ].copy()

    # ------------------------------------------------------------------
    # Scenario ratios and diagnostics
    # ------------------------------------------------------------------

    scenario_ratio_cols = _unique_existing_columns(
        scored_scenarios,
        ["scenario"] + ratio_cols,
    )

    scenario_ratios = scored_scenarios[
        scenario_ratio_cols
    ].copy()

    # ------------------------------------------------------------------
    # Flat scenario CSV output
    # ------------------------------------------------------------------

    scenario_full_cols = _unique_existing_columns(
        scored_scenarios,
        existing_scenario_summary_cols + scenario_ratio_cols,
    )

    scenario_file = scored_scenarios[
        scenario_full_cols
    ].copy()

    # ------------------------------------------------------------------
    # Dedicated guardrail report table
    # ------------------------------------------------------------------

    guardrails = _build_guardrail_table(
        scored_manual_with_outlook=scored_manual_with_outlook,
        scored_scenarios=scored_scenarios,
    )

    return {
        "score_summary": score_summary,
        "company_ratios": company_ratios,
        "cluster_comparison": cluster_comparison,
        "scored_file": scored_file,
        "scenario_input_snapshot": scenario_input_snapshot,
        "scenario_score_summary": scenario_score_summary,
        "scenario_ratios": scenario_ratios,
        "scenario_file": scenario_file,
        "guardrails": guardrails,
    }


def save_credit_report_outputs(
    tables: Mapping[str, pd.DataFrame],
    output_path: str | Path,
    base_filename: str = "manual_2025",
) -> dict[str, Path]:
    """
    Save Notebook 03 report outputs.

    Produces:
    - base score CSV
    - scenario CSV
    - combined Excel report with a dedicated guardrails sheet
    """

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    score_csv = output_path / f"{base_filename}_score_result.csv"
    scenario_csv = output_path / f"{base_filename}_scenario_analysis.csv"
    report_xlsx = output_path / f"{base_filename}_score_report.xlsx"

    tables["scored_file"].to_csv(score_csv, index=False)
    tables["scenario_file"].to_csv(scenario_csv, index=False)

    with pd.ExcelWriter(report_xlsx, engine="openpyxl") as writer:
        tables["score_summary"].to_excel(
            writer,
            sheet_name="score_summary",
            index=False,
        )

        tables["company_ratios"].to_excel(
            writer,
            sheet_name="company_ratios",
            index=False,
        )

        tables["guardrails"].to_excel(
            writer,
            sheet_name="guardrails",
            index=False,
        )

        tables["cluster_comparison"].to_excel(
            writer,
            sheet_name="cluster_comparison",
            index=False,
        )

        tables["scenario_input_snapshot"].to_excel(
            writer,
            sheet_name="scenario_inputs",
            index=False,
        )

        tables["scenario_score_summary"].to_excel(
            writer,
            sheet_name="scenario_scores",
            index=False,
        )

        tables["scenario_ratios"].to_excel(
            writer,
            sheet_name="scenario_ratios",
            index=False,
        )

        tables["scenario_file"].to_excel(
            writer,
            sheet_name="scenario_full_output",
            index=False,
        )

        tables["scored_file"].to_excel(
            writer,
            sheet_name="raw_base_output",
            index=False,
        )

    return {
        "score_csv": score_csv,
        "scenario_csv": scenario_csv,
        "report_xlsx": report_xlsx,
    }


__all__ = [
    "build_credit_report_tables",
    "save_credit_report_outputs",
]
