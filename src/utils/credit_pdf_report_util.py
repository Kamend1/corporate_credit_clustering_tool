"""
Formal PDF credit report utilities for the private-company credit scoring tool.

This module is intentionally presentation-only. It does not score companies,
engineer features, train clusters, or modify model artifacts. It consumes the
already prepared Notebook 03 outputs and produces a formal consulting-style PDF.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

import math
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# -----------------------------------------------------------------------------
# Visual identity
# -----------------------------------------------------------------------------

NAVY = colors.HexColor("#1F4E78")
BLUE = colors.HexColor("#5B9BD5")
LIGHT_BLUE = colors.HexColor("#D9EAF7")
LIGHT_GREY = colors.HexColor("#F2F2F2")
MID_GREY = colors.HexColor("#D9D9D9")
DARK_GREY = colors.HexColor("#333333")
AMBER = colors.HexColor("#FFF2CC")
RED_TINT = colors.HexColor("#FCE4D6")
GREEN_TINT = colors.HexColor("#E2F0D9")
WHITE = colors.white
BLACK = colors.black

PAGE_WIDTH, PAGE_HEIGHT = A4
DEFAULT_MARGIN = 1.45 * cm

# Stable external interpretation scale.
# The raw KMeans cluster ids are intentionally not shown in the executive
# summary because KMeans label numbers are arbitrary and can change between
# model training runs.
CREDIT_RISK_SCALE = [
    "1 - Low risk / investment-grade-like",
    "2 - Moderate risk / lower-investment-grade-like",
    "3 - Elevated risk / leveraged",
    "4 - High risk / weak-credit-like",
    "5 - Near-default / distressed-like",
]

CORE_RISK_DIMENSIONS = [
    ("Leverage risk", "leverage_risk", "Balance-sheet leverage and equity buffer"),
    ("Liquidity risk", "liquidity_risk", "Current, quick, and cash liquidity"),
    ("Earnings risk", "earnings_risk", "Profitability relative to assets"),
    (
        "Operating cash-flow risk",
        "operating_cashflow_risk",
        "Operating cash-flow generation relative to assets",
    ),
    (
        "Debt-service risk",
        "debt_service_risk",
        "Interest, free cash flow, and EBITDA-based debt service",
    ),
    (
        "Structural distress risk",
        "structural_distress_risk",
        "Negative equity or liabilities above assets",
    ),
]

FINANCIAL_SCALE_METRICS = [
    ("Assets", "assets"),
    ("Revenue", "revenue"),
    ("Equity", "equity"),
    ("Total debt", "total_debt"),
    ("Net debt", "net_debt"),
    ("EBITDA", "ebitda"),
    ("Operating income", "operating_income"),
    ("CFO", "cfo"),
    ("Free cash flow", "fcf"),
]

KEY_RATIO_METRICS = [
    ("Liabilities / assets", "liabilities_to_assets", "Capital structure burden"),
    ("Debt / assets", "debt_to_assets", "Debt load"),
    ("Debt / equity", "debt_to_equity", "Leverage relative to book equity"),
    ("Current ratio", "current_ratio", "Short-term liquidity"),
    ("Quick ratio", "quick_ratio", "Liquid asset coverage"),
    ("Interest coverage", "interest_coverage", "EBIT / interest expense"),
    (
        "EBITDA interest coverage",
        "ebitda_interest_coverage",
        "EBITDA / interest expense",
    ),
    ("Debt / EBITDA", "debt_to_ebitda", "Debt payback proxy"),
    ("Net debt / EBITDA", "net_debt_to_ebitda", "Net leverage"),
    ("FCF / debt", "fcf_to_debt", "Cash debt repayment capacity"),
    ("EBITDA margin", "ebitda_margin", "Operating profitability proxy"),
]

CLUSTER_COMPARISON_METRICS = [
    "liabilities_to_assets",
    "debt_to_assets",
    "debt_to_equity",
    "equity_to_assets",
    "current_ratio",
    "quick_ratio",
    "cash_to_assets",
    "net_income_to_assets",
    "cfo_to_assets",
    "interest_coverage",
    "fcf_to_debt",
    "ebitda_margin",
    "debt_to_ebitda",
    "net_debt_to_ebitda",
    "ebitda_interest_coverage",
]

SCENARIO_PDF_COLUMNS = [
    "scenario",
    "cluster_label",
    "scorecard_credit_score",
    "outlook_flag",
    "guardrail_level",
    "debt_to_ebitda",
    "interest_coverage",
    "ebitda_interest_coverage",
    "warning_flags",
]


GUARDRAIL_PDF_FIELDS = [
    ("Guardrail level", "guardrail_level"),
    ("Guardrail flags", "guardrail_flags"),
    ("Guardrail summary", "guardrail_summary"),
    ("Analyst interpretation", "analyst_interpretation"),
    ("Commercial conclusion", "commercial_conclusion"),
]


# -----------------------------------------------------------------------------
# Style helpers
# -----------------------------------------------------------------------------


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "CreditReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=14,
        ),
        "Subtitle": ParagraphStyle(
            "CreditReportSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            textColor=DARK_GREY,
            spaceAfter=8,
        ),
        "Section": ParagraphStyle(
            "CreditReportSection",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=NAVY,
            spaceBefore=10,
            spaceAfter=8,
        ),
        "Subsection": ParagraphStyle(
            "CreditReportSubsection",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            textColor=DARK_GREY,
            spaceBefore=7,
            spaceAfter=5,
        ),
        "Body": ParagraphStyle(
            "CreditReportBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.2,
            textColor=DARK_GREY,
            spaceAfter=6,
        ),
        "Small": ParagraphStyle(
            "CreditReportSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor("#666666"),
        ),
        "SmallBold": ParagraphStyle(
            "CreditReportSmallBold",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=9.5,
            textColor=DARK_GREY,
        ),
        "TableCell": ParagraphStyle(
            "CreditReportTableCell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.4,
            leading=9.0,
            textColor=DARK_GREY,
        ),
        "TableCellBold": ParagraphStyle(
            "CreditReportTableCellBold",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.4,
            leading=9.0,
            textColor=DARK_GREY,
        ),
        "TableHeader": ParagraphStyle(
            "CreditReportTableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.4,
            leading=9.0,
            textColor=WHITE,
        ),
        "Footer": ParagraphStyle(
            "CreditReportFooter",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7,
            leading=9,
            textColor=colors.HexColor("#777777"),
            alignment=TA_CENTER,
        ),
        "CardLabel": ParagraphStyle(
            "CreditReportCardLabel",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7,
            leading=8.5,
            textColor=colors.HexColor("#666666"),
            alignment=TA_CENTER,
        ),
        "CardValue": ParagraphStyle(
            "CreditReportCardValue",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9.2,
            leading=10.8,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
    }


def _p(text: Any, style: ParagraphStyle) -> Paragraph:
    safe = "" if text is None else str(text)
    safe = safe.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(safe, style)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _value(row: pd.Series | Mapping[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(row, pd.Series):
        return row.get(key, default)
    return row.get(key, default)


def _format_number(value: Any, digits: int = 2) -> str:
    if _is_missing(value):
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:,.{digits}f}"


def _format_ratio(value: Any, digits: int = 2) -> str:
    if _is_missing(value):
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}x"


def _format_score(value: Any) -> str:
    if _is_missing(value):
        return "n/a"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def _format_pct(value: Any, digits: int = 1) -> str:
    if _is_missing(value):
        return "n/a"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def _format_metric_value(metric: str, value: Any) -> str:
    if _is_missing(value):
        return "n/a"
    if metric in {"cluster", "assigned_cluster", "risk_rank", "fiscal_year"}:
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return str(value)
    if "affinity" in metric or metric.endswith("_margin") or metric.endswith("_to_assets"):
        return _format_number(value, 4)
    if "coverage" in metric or "debt_to_ebitda" in metric or "ratio" in metric:
        return _format_number(value, 2)
    if "score" in metric or metric.endswith("_risk"):
        return _format_number(value, 2)
    if metric in {"assets", "liabilities", "equity", "cash", "revenue", "net_income", "cfo", "total_debt", "net_debt", "ebitda", "fcf"}:
        return _format_number(value, 0)
    try:
        return _format_number(value, 2)
    except Exception:
        return str(value)


def _first_row(df: pd.DataFrame | None) -> pd.Series:
    if df is None or len(df) == 0:
        return pd.Series(dtype="object")
    return df.iloc[0]


def _table_style(header_rows: int = 1, font_size: float = 7.4) -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, header_rows - 1), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), WHITE),
            ("FONTNAME", (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("LEADING", (0, 0), (-1, -1), font_size + 1.8),
            ("GRID", (0, 0), (-1, -1), 0.35, MID_GREY),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, header_rows), (-1, -1), [WHITE, LIGHT_GREY]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
    )


def _dataframe_table(
    df: pd.DataFrame,
    styles: Mapping[str, ParagraphStyle],
    col_widths: Sequence[float] | None = None,
    max_rows: int | None = None,
) -> Table:
    if max_rows is not None:
        df = df.head(max_rows).copy()
    data = [[_p(col, styles["TableHeader"]) for col in df.columns]]
    for _, row in df.iterrows():
        data.append([_p(_format_metric_value(str(col), row[col]), styles["TableCell"]) for col in df.columns])
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(_table_style())
    return table


# -----------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------


def _save_scorecard_chart(row: pd.Series, output_path: Path) -> Path | None:
    labels = []
    values = []
    for label, col, _ in CORE_RISK_DIMENSIONS:
        value = row.get(col)
        if not _is_missing(value):
            labels.append(label)
            values.append(float(value))
    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    y_positions = range(len(labels))
    ax.barh(list(y_positions), values)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Risk score: 0 = stronger, 1 = weaker", fontsize=8)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_scenario_chart(scored_scenarios: pd.DataFrame, output_path: Path) -> Path | None:
    if scored_scenarios is None or scored_scenarios.empty:
        return None
    required = {"scenario", "scorecard_credit_score"}
    if not required.issubset(scored_scenarios.columns):
        return None
    plot_df = scored_scenarios.dropna(subset=["scorecard_credit_score"]).copy()
    if plot_df.empty:
        return None

    labels = [str(x).replace("_", "\n") for x in plot_df["scenario"]]
    values = plot_df["scorecard_credit_score"].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Scorecard credit score", fontsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


# -----------------------------------------------------------------------------
# Report sections
# -----------------------------------------------------------------------------


def _metadata_table(row: pd.Series, artifact: Mapping[str, Any], styles: Mapping[str, ParagraphStyle]) -> Table:
    metadata = [
        ["Report date", str(date.today())],
        ["Model version", artifact.get("artifact_version", "n/a")],
        ["Segment", artifact.get("primary_segment", row.get("financial_flag", "Non-financial"))],
        ["Model type", "KMeans credit risk clustering"],
        ["Currency", row.get("currency", "n/a")],
        ["Source", "Manual financial statement input"],
    ]
    data = [[_p("Field", styles["TableHeader"]), _p("Value", styles["TableHeader"])]] + [
        [_p(k, styles["TableCell"]), _p(v, styles["TableCell"])] for k, v in metadata
    ]
    table = Table(data, colWidths=[4.2 * cm, 10.0 * cm], hAlign="LEFT")
    table.setStyle(_table_style())
    return table


def _kpi_cards(row: pd.Series, styles: Mapping[str, ParagraphStyle]) -> Table:
    """Build first-page KPI cards without exposing the raw KMeans cluster id."""
    cards = [
        ("Risk label", row.get("cluster_label", "n/a")),
        ("Risk rank", _format_metric_value("risk_rank", row.get("risk_rank"))),
        ("Score", _format_score(row.get("scorecard_credit_score"))),
        ("Outlook", row.get("outlook_flag", "n/a")),
        ("Near-default affinity", _format_pct(row.get("near_default_affinity"))),
    ]

    inner_cards = []
    for label, value in cards:
        inner = Table(
            [[_p(label, styles["CardLabel"])], [_p(value, styles["CardValue"])]],
            colWidths=[5.5 * cm],
        )
        inner.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), WHITE),
                    ("BOX", (0, 0), (-1, -1), 0.8, MID_GREY),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.4, LIGHT_BLUE),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        inner_cards.append(inner)

    empty_cell = ""
    table = Table(
        [inner_cards[:3], inner_cards[3:] + [empty_cell]],
        colWidths=[5.7 * cm, 5.7 * cm, 5.7 * cm],
        rowHeights=[1.85 * cm, 1.85 * cm],
        hAlign="LEFT",
    )
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return table


def _risk_scale_table(styles: Mapping[str, ParagraphStyle]) -> Table:
    """Show the stable 1-to-5 business interpretation scale used in the report."""
    data = [[_p("Credit risk scale used for interpretation", styles["TableHeader"])] ]
    data.extend([[_p(label, styles["TableCell"])] for label in CREDIT_RISK_SCALE])
    table = Table(data, colWidths=[17.2 * cm], hAlign="LEFT")
    table.setStyle(_table_style())
    return table

def _warning_box(row: pd.Series, styles: Mapping[str, ParagraphStyle]) -> Table:
    flags = row.get("warning_flags", "none")
    if _is_missing(flags) or str(flags).strip().lower() in {"", "none", "nan"}:
        text = "No automatic warning flags were triggered."
        fill = GREEN_TINT
    else:
        split_flags = [f.strip() for f in str(flags).split(",") if f.strip()]
        text = "Key warning flags: " + "; ".join(split_flags)
        fill = AMBER
    table = Table([[_p(text, styles["Body"])]], colWidths=[17.2 * cm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), fill),
                ("BOX", (0, 0), (-1, -1), 0.6, MID_GREY),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def _guardrail_assessment_section(row: pd.Series, styles: Mapping[str, ParagraphStyle]) -> list[Any]:
    """Build the PDF guardrail section from the scored dataframe columns."""
    has_guardrails = any(
        field in row.index and not _is_missing(row.get(field)) and str(row.get(field)).strip()
        for _, field in GUARDRAIL_PDF_FIELDS
    )

    if not has_guardrails:
        return []

    level = row.get("guardrail_level", "n/a")
    flags = row.get("guardrail_flags", "No guardrail flags")
    summary = row.get("guardrail_summary", "n/a")
    analyst_interpretation = row.get("analyst_interpretation", "n/a")
    commercial_conclusion = row.get("commercial_conclusion", "n/a")

    if _is_missing(flags) or str(flags).strip().lower() in {"", "none", "nan"}:
        flags = "No guardrail flags"

    level_text = str(level)
    fill = LIGHT_BLUE
    if level_text in {"High caution", "Override required"}:
        fill = RED_TINT
    elif level_text == "Caution":
        fill = AMBER
    elif level_text in {"Clear", "Monitor"}:
        fill = GREEN_TINT if level_text == "Clear" else LIGHT_BLUE

    guardrail_table = Table(
        [
            [_p("Guardrail level", styles["TableCellBold"]), _p(level_text, styles["TableCell"])],
            [_p("Guardrail flags", styles["TableCellBold"]), _p(flags, styles["TableCell"])],
        ],
        colWidths=[4.2 * cm, 13.0 * cm],
        hAlign="LEFT",
    )
    guardrail_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
                ("BACKGROUND", (1, 0), (1, 0), fill),
                ("GRID", (0, 0), (-1, -1), 0.35, MID_GREY),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.4),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    return [
        Spacer(1, 0.12 * cm),
        Paragraph("Guardrail assessment", styles["Subsection"]),
        guardrail_table,
        Spacer(1, 0.12 * cm),
        Paragraph("Guardrail summary", styles["SmallBold"]),
        Paragraph(summary, styles["Body"]),
        Paragraph("Analyst interpretation", styles["SmallBold"]),
        Paragraph(analyst_interpretation, styles["Body"]),
        Paragraph("Commercial conclusion", styles["SmallBold"]),
        Paragraph(commercial_conclusion, styles["Body"]),
    ]


def _executive_narrative(row: pd.Series) -> str:
    company = row.get("company_name", "The company")
    label = row.get("cluster_label", "unlabelled risk group")
    outlook = row.get("outlook_flag", "Neutral")
    score = _format_score(row.get("scorecard_credit_score"))
    return (
        f"{company} is classified as '{label}' under the stable 1-to-5 risk-label scale shown above. "
        f"The scorecard credit score is {score}. The scorecard evaluates relative "
        "credit risk across leverage, liquidity, earnings quality, operating cash-flow "
        "generation, debt-service capacity, and structural balance-sheet indicators. "
        f"The current classification outlook is {outlook}."
    )

def _scorecard_table(row: pd.Series, styles: Mapping[str, ParagraphStyle]) -> Table:
    rows = [["Dimension", "Score", "Interpretation"]]
    for label, col, description in CORE_RISK_DIMENSIONS:
        rows.append([label, _format_metric_value(col, row.get(col)), description])
    df = pd.DataFrame(rows[1:], columns=rows[0])
    return _dataframe_table(df, styles, col_widths=[4.6 * cm, 2.2 * cm, 10.3 * cm])


def _financial_scale_table(row: pd.Series, styles: Mapping[str, ParagraphStyle]) -> Table:
    rows = []
    for label, col in FINANCIAL_SCALE_METRICS:
        if col in row.index and not _is_missing(row.get(col)):
            rows.append({"Metric": label, "Value": _format_metric_value(col, row.get(col))})
    df = pd.DataFrame(rows)
    return _dataframe_table(df, styles, col_widths=[7.0 * cm, 4.2 * cm])


def _ratio_table(row: pd.Series, styles: Mapping[str, ParagraphStyle]) -> Table:
    rows = []
    for label, col, comment in KEY_RATIO_METRICS:
        if col in row.index:
            rows.append(
                {
                    "Ratio": label,
                    "Value": _format_metric_value(col, row.get(col)),
                    "Comment": comment,
                }
            )
    df = pd.DataFrame(rows)
    return _dataframe_table(df, styles, col_widths=[5.3 * cm, 2.8 * cm, 8.9 * cm])


def _cluster_comparison_table(
    comparison_to_cluster: pd.DataFrame,
    styles: Mapping[str, ParagraphStyle],
) -> Table:
    comp = comparison_to_cluster.copy()
    if "metric" not in comp.columns:
        comp = comp.reset_index().rename(columns={"index": "metric"})
    comp = comp[comp["metric"].isin(CLUSTER_COMPARISON_METRICS)].copy()
    keep_cols = [
        col
        for col in ["metric", "company_value", "assigned_cluster_median", "difference", "relative_position"]
        if col in comp.columns
    ]
    comp = comp[keep_cols].copy()
    rename = {
        "metric": "Metric",
        "company_value": "Company",
        "assigned_cluster_median": "Risk group median",
        "difference": "Difference",
        "relative_position": "Position",
    }
    comp = comp.rename(columns=rename)
    return _dataframe_table(comp, styles, col_widths=[4.7 * cm, 3.0 * cm, 3.3 * cm, 3.0 * cm, 3.1 * cm], max_rows=16)


def _scenario_table(scored_scenarios: pd.DataFrame, styles: Mapping[str, ParagraphStyle]) -> Table:
    if scored_scenarios is None or scored_scenarios.empty:
        df = pd.DataFrame([{"Scenario": "No scenario data available"}])
        return _dataframe_table(df, styles)
    cols = [col for col in SCENARIO_PDF_COLUMNS if col in scored_scenarios.columns]
    df = scored_scenarios[cols].copy()
    df = df.rename(
        columns={
            "scenario": "Scenario",
            "cluster_label": "Risk label",
            "scorecard_credit_score": "Score",
            "outlook_flag": "Outlook",
            "guardrail_level": "Guardrail",
            "debt_to_ebitda": "Debt / EBITDA",
            "interest_coverage": "Interest coverage",
            "ebitda_interest_coverage": "EBITDA coverage",
            "warning_flags": "Warnings",
        }
    )
    return _dataframe_table(
        df,
        styles,
        col_widths=[
            2.6 * cm,
            3.3 * cm,
            1.1 * cm,
            1.25 * cm,
            1.6 * cm,
            1.35 * cm,
            1.35 * cm,
            1.35 * cm,
            2.0 * cm,
        ],
    )


def _methodology_section(styles: Mapping[str, ParagraphStyle]) -> list[Any]:
    bullets = [
        "The model is an unsupervised KMeans clustering model, not a supervised default prediction model.",
        "The assigned risk label is a relative risk grouping, not a formal external credit rating.",
        "The output does not represent a probability of default.",
        "The model is calibrated on the reference dataset used in training and should be interpreted accordingly.",
        "SME/private-company use should be treated as diagnostic benchmarking, not as bank-grade credit approval.",
        "Financial institutions are outside the scope of this non-financial model.",
        "Manual input quality materially affects the output.",
    ]
    flow = [
        Paragraph("5. Methodology and Important Limitations", styles["Section"]),
        Paragraph(
            "The analysis uses an unsupervised KMeans clustering model trained on "
            "non-financial company observations. Financial statement data are transformed "
            "into bounded credit-risk indicators across leverage, liquidity, earnings, "
            "operating cash-flow, debt-service capacity, and structural distress. The "
            "company is mapped to the closest trained cluster internally, then presented through a stable business risk label.",
            styles["Body"],
        ),
        Spacer(1, 0.2 * cm),
    ]
    bullet_rows = [[_p("Important limitations", styles["TableHeader"])] ]
    for item in bullets:
        bullet_rows.append([_p("- " + item, styles["TableCell"])] )
    table = Table(bullet_rows, colWidths=[17.2 * cm], hAlign="LEFT")
    table.setStyle(_table_style())
    flow.append(table)
    flow.append(Spacer(1, 0.25 * cm))
    flow.append(
        Paragraph(
            "This report is an analytical credit-risk diagnostic and does not constitute a formal credit rating, lending decision, investment recommendation, or probability-of-default estimate.",
            styles["Small"],
        )
    )
    return flow


# -----------------------------------------------------------------------------
# Page decorations
# -----------------------------------------------------------------------------


def _draw_header_footer(canvas, doc, company_name: str, first_page: bool = False) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 0.65 * cm, width, 0.65 * cm, stroke=0, fill=1)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 7.5)
    if first_page:
        header = "Private Company Credit Risk Assessment Report"
    else:
        header = f"Private Company Credit Risk Assessment Report | {company_name}"
    canvas.drawString(DEFAULT_MARGIN, height - 0.42 * cm, header[:110])
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.setFont("Helvetica", 7)
    footer = f"Page {doc.page} | Analytical credit-risk diagnostic - not a formal rating"
    canvas.drawCentredString(width / 2, 0.65 * cm, footer)
    canvas.restoreState()


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def save_credit_pdf_report(
    report_tables: Mapping[str, pd.DataFrame] | None,
    scored_manual_with_outlook: pd.DataFrame,
    comparison_to_cluster: pd.DataFrame,
    scored_scenarios: pd.DataFrame,
    artifact: Mapping[str, Any] | None,
    output_path: str | Path,
    base_filename: str = "manual_2025",
    prepared_by: str = "KSB Analytica / Credit Clustering Scorecard",
) -> Path:
    """
    Save a formal consulting-style PDF credit report.

    Parameters
    ----------
    report_tables : Mapping[str, pd.DataFrame] | None
        Optional report tables from credit_report_util.py. Used as fallback source.
    scored_manual_with_outlook : pd.DataFrame
        One-row scored company dataframe with outlook columns.
    comparison_to_cluster : pd.DataFrame
        Company-vs-cluster comparison table.
    scored_scenarios : pd.DataFrame
        Scored scenario dataframe.
    artifact : Mapping[str, Any] | None
        Loaded model artifact. Used for metadata only.
    output_path : str | Path
        Output folder.
    base_filename : str
        Base output filename. The PDF will be {base_filename}_credit_report.pdf.
    prepared_by : str
        Header/footer branding line.

    Returns
    -------
    pathlib.Path
        Saved PDF path.
    """

    artifact = artifact or {}
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path / f"{base_filename}_credit_report.pdf"

    styles = _styles()
    row = _first_row(scored_manual_with_outlook)
    company_name = str(row.get("company_name", base_filename))
    fiscal_year = row.get("fiscal_year", "n/a")

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=DEFAULT_MARGIN,
        leftMargin=DEFAULT_MARGIN,
        topMargin=1.35 * cm,
        bottomMargin=1.25 * cm,
        title="Private Company Credit Risk Assessment Report",
        author=prepared_by,
    )

    flow: list[Any] = []

    # Page 1: Cover + executive summary.
    flow.append(Paragraph("Private Company Credit Risk Assessment Report", styles["Title"]))
    flow.append(Paragraph(f"{company_name} | Fiscal Year {fiscal_year}", styles["Subtitle"]))
    flow.append(Paragraph("Credit clustering scorecard analysis based on financial statement inputs", styles["Subtitle"]))
    flow.append(Spacer(1, 0.2 * cm))
    flow.append(_metadata_table(row, artifact, styles))
    flow.append(Spacer(1, 0.35 * cm))
    flow.append(_kpi_cards(row, styles))
    flow.append(Spacer(1, 0.25 * cm))
    flow.append(_risk_scale_table(styles))
    flow.append(Spacer(1, 0.30 * cm))
    flow.append(Paragraph("Executive conclusion", styles["Subsection"]))
    flow.append(Paragraph(_executive_narrative(row), styles["Body"]))
    flow.extend(_guardrail_assessment_section(row, styles))
    flow.append(_warning_box(row, styles))
    flow.append(Spacer(1, 0.25 * cm))
    flow.append(Paragraph("Prepared by: " + prepared_by, styles["Small"]))
    flow.append(PageBreak())

    # Page 2: Scorecard breakdown.
    flow.append(Paragraph("1. Credit Scorecard Breakdown", styles["Section"]))
    flow.append(
        Paragraph(
            "The scorecard components are bounded directional indicators. A value of 0.00 indicates a stronger credit characteristic, while 1.00 indicates a weaker credit characteristic.",
            styles["Body"],
        )
    )
    flow.append(_scorecard_table(row, styles))
    flow.append(Spacer(1, 0.25 * cm))
    with TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        scorecard_chart = _save_scorecard_chart(row, tmp_dir / "scorecard_chart.png")
        scenario_chart = _save_scenario_chart(scored_scenarios, tmp_dir / "scenario_chart.png")
        if scorecard_chart:
            flow.append(Image(str(scorecard_chart), width=16.3 * cm, height=7.0 * cm))
        flow.append(PageBreak())

        # Page 3: Financial diagnostics.
        flow.append(Paragraph("2. Key Financial Diagnostics", styles["Section"]))
        flow.append(Paragraph("Financial scale", styles["Subsection"]))
        flow.append(_financial_scale_table(row, styles))
        flow.append(Spacer(1, 0.25 * cm))
        flow.append(Paragraph("Key ratios", styles["Subsection"]))
        flow.append(_ratio_table(row, styles))
        flow.append(Spacer(1, 0.20 * cm))
        flow.append(
            Paragraph(
                "Note: EBITDA is derived as operating income plus depreciation and amortization where direct EBITDA is not provided.",
                styles["Small"],
            )
        )
        flow.append(PageBreak())

        # Page 4: Cluster comparison.
        flow.append(Paragraph("3. Comparison with Assigned Risk Group Median", styles["Section"]))
        flow.append(
            Paragraph(
                "The comparison table explains the company's position relative to the median financial profile of its assigned labelled risk group. Positive or negative differences should be interpreted according to the economic meaning of each metric; higher liquidity ratios are generally stronger, while higher leverage ratios are generally weaker.",
                styles["Body"],
            )
        )
        flow.append(_cluster_comparison_table(comparison_to_cluster, styles))
        flow.append(PageBreak())

        # Page 5: Scenario analysis.
        flow.append(Paragraph("4. Scenario Analysis", styles["Section"]))
        flow.append(
            Paragraph(
                "Scenario analysis assesses the stability of the company's risk label and scorecard risk profile under simplified operating, leverage, liquidity, and distress assumptions. The scenarios are mechanical sensitivities and do not represent forecasts.",
                styles["Body"],
            )
        )
        flow.append(_scenario_table(scored_scenarios, styles))
        flow.append(Spacer(1, 0.25 * cm))
        if scenario_chart:
            flow.append(Image(str(scenario_chart), width=16.3 * cm, height=6.5 * cm))
        flow.append(PageBreak())

        # Page 6: Methodology and limitations.
        flow.extend(_methodology_section(styles))

        doc.build(
            flow,
            onFirstPage=lambda canvas, document: _draw_header_footer(canvas, document, company_name, first_page=True),
            onLaterPages=lambda canvas, document: _draw_header_footer(canvas, document, company_name, first_page=False),
        )

    return pdf_path


__all__ = ["save_credit_pdf_report"]
