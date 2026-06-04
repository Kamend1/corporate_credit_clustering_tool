"""
Credit guardrail utilities for the credit clustering project.

This module adds a consulting-grade interpretation layer on top of the
mechanical scorecard output.  It does not change the assigned cluster or the
model score.  Instead, it flags situations where the model output should be
qualified before being presented in a client-facing report.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


from .config import (
    CREDIT_GUARDRAILS,
    DEBT_SERVICE_GUARDRAIL_FLAGS,
    GUARDRAIL_COLS,
    GUARDRAIL_SEVERITY_ORDER,
    HIGH_CAUTION_GUARDRAIL_FLAGS,
    LEVERAGE_GUARDRAIL_FLAGS,
    LIQUIDITY_GUARDRAIL_FLAGS,
    STRUCTURAL_GUARDRAIL_FLAGS,
)


BASE_RATIO_GUARDRAILS = CREDIT_GUARDRAILS

LEVERAGE_FLAG_NAMES = LEVERAGE_GUARDRAIL_FLAGS
DEBT_SERVICE_FLAG_NAMES = DEBT_SERVICE_GUARDRAIL_FLAGS
HIGH_CAUTION_FLAG_NAMES = HIGH_CAUTION_GUARDRAIL_FLAGS
LIQUIDITY_FLAG_NAMES = LIQUIDITY_GUARDRAIL_FLAGS
STRUCTURAL_FLAG_NAMES = STRUCTURAL_GUARDRAIL_FLAGS

INVESTMENT_GRADE_MARKERS = (
    "AAA",
    "AA",
    "A",
    "BAA",
    "BBB",
)

SPECULATIVE_GRADE_MARKERS = (
    "BA",
    "BB",
    "B",
    "CAA",
    "CCC",
    "CC",
    "C",
    "D",
)


# ---------------------------------------------------------------------
# Rule engine helpers
# ---------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    """Return True if value is missing / not usable for rule checks."""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False



def _compare(value: Any, operator: str, threshold: float) -> bool:
    """Safely compare a numeric value with a threshold."""
    if _is_missing(value):
        return False

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False

    if not np.isfinite(numeric_value):
        return False

    if operator == ">":
        return numeric_value > threshold
    if operator == ">=":
        return numeric_value >= threshold
    if operator == "<":
        return numeric_value < threshold
    if operator == "<=":
        return numeric_value <= threshold
    if operator == "==":
        return numeric_value == threshold

    raise ValueError(f"Unsupported guardrail operator: {operator!r}")



def _highest_severity(severities: Sequence[str]) -> str:
    """Return the highest severity from a sequence of severity labels."""
    if not severities:
        return "Clear"

    return max(
        severities,
        key=lambda severity: GUARDRAIL_SEVERITY_ORDER.get(severity, -1),
    )


def _looks_speculative_grade(external_rating: str | None) -> bool:
    """
    Return True when external rating text contains speculative-grade markers.

    This is intentionally simple.  It is a consulting guardrail, not a rating parser.
    Examples triggering this rule: 'Ba2 / BB', 'BB stable', 'B+', 'CCC'.
    """
    if not external_rating:
        return False

    text = str(external_rating).upper().replace("-", " ").replace("+", " ")
    tokens = [token.strip() for token in text.replace("/", " ").split()]

    for token in tokens:
        if any(token.startswith(marker) for marker in INVESTMENT_GRADE_MARKERS):
            return False

        if any(token.startswith(marker) for marker in SPECULATIVE_GRADE_MARKERS):
            return True

    return False



def _evaluate_ratio_guardrails(row: pd.Series) -> list[tuple[str, str]]:
    """Evaluate direct ratio-based guardrails for one scored company row."""
    triggered: list[tuple[str, str]] = []

    for flag_name, rule in BASE_RATIO_GUARDRAILS.items():
        column = rule["column"]
        if column not in row.index:
            continue

        if _compare(
            row[column],
            operator=rule["operator"],
            threshold=rule["threshold"],
        ):
            triggered.append((flag_name, rule["severity"]))

    return triggered



def _evaluate_model_contradiction_guardrails(
    row: pd.Series,
    flags: Sequence[str],
) -> list[tuple[str, str]]:
    """Evaluate guardrails that compare the model bucket with financial diagnostics."""
    triggered: list[tuple[str, str]] = []
    flag_set = set(flags)

    risk_rank = row.get("risk_rank")
    try:
        risk_rank_int = int(risk_rank)
    except (TypeError, ValueError):
        risk_rank_int = None

    leverage_or_debt_service_flags_exist = bool(
        flag_set.intersection(LEVERAGE_FLAG_NAMES | DEBT_SERVICE_FLAG_NAMES)
    )
    high_caution_flags_exist = bool(flag_set.intersection(HIGH_CAUTION_FLAG_NAMES))

    if risk_rank_int == 1 and leverage_or_debt_service_flags_exist:
        triggered.append(("top_bucket_with_leverage_caveat", "Caution"))

    if risk_rank_int == 1 and high_caution_flags_exist:
        triggered.append(("top_bucket_with_high_caution_caveat", "High caution"))

    if risk_rank_int is not None and risk_rank_int >= 5:
        triggered.append(("distressed_cluster_requires_manual_review", "Override required"))

    return triggered



def _evaluate_external_rating_guardrails(
    external_rating: str | None,
) -> list[tuple[str, str]]:
    """Evaluate optional external-rating guardrails."""
    if _looks_speculative_grade(external_rating):
        return [("external_rating_speculative_grade", "Caution")]
    return []



def _build_guardrail_summary(level: str, flags: Sequence[str], row: pd.Series) -> str:
    """Build short report-facing guardrail summary."""
    if not flags:
        return (
            "No material guardrail issues were identified based on the configured "
            "ratio, structural, and model-consistency checks."
        )

    risk_rank = row.get("risk_rank")
    cluster_label = row.get("cluster_label", "the assigned model bucket")

    if "distressed_cluster_requires_manual_review" in flags:
        return (
            "The company is assigned to the weakest model-relative bucket. The result "
            "requires manual analyst review before being used in a client-facing conclusion."
        )

    if "top_bucket_with_high_caution_caveat" in flags:
        return (
            "The model assigns the company to the strongest relative credit-risk bucket, "
            "but high-caution leverage or debt-service indicators materially constrain "
            "the interpretation."
        )

    if "top_bucket_with_leverage_caveat" in flags:
        return (
            "The model assigns the company to the strongest relative credit-risk bucket, "
            "but leverage and/or debt-service metrics require explicit qualification."
        )

    if STRUCTURAL_FLAG_NAMES.intersection(flags):
        return (
            "Structural balance-sheet distress indicators were triggered. The model output "
            "should not be used without manual analyst review."
        )

    if level in {"Caution", "High caution"}:
        return (
            f"The company is assigned to {cluster_label} with risk rank {risk_rank}, "
            "but selected financial diagnostics require a qualified interpretation."
        )

    return (
        "Selected monitoring guardrails were triggered, but they do not materially "
        "contradict the assigned model bucket."
    )



def _build_analyst_interpretation(level: str, flags: Sequence[str], row: pd.Series) -> str:
    """Build formal analyst interpretation paragraph."""
    if not flags:
        return (
            "The model output and key financial diagnostics are broadly consistent. "
            "The assigned cluster may be interpreted as a relative credit-risk grouping, "
            "subject to the methodology limitations of the scorecard."
        )

    positives = []
    concerns = []

    if row.get("liquidity_risk") is not None and not _is_missing(row.get("liquidity_risk")):
        if float(row.get("liquidity_risk")) <= 0.25:
            positives.append("liquidity")

    if row.get("operating_cashflow_risk") is not None and not _is_missing(row.get("operating_cashflow_risk")):
        if float(row.get("operating_cashflow_risk")) <= 0.25:
            positives.append("operating cash-flow generation")

    if row.get("structural_distress_risk") is not None and not _is_missing(row.get("structural_distress_risk")):
        if float(row.get("structural_distress_risk")) == 0.0:
            positives.append("absence of structural balance-sheet distress")

    if set(flags).intersection(LEVERAGE_FLAG_NAMES):
        concerns.append("leverage")

    if set(flags).intersection(DEBT_SERVICE_FLAG_NAMES):
        concerns.append("debt-service capacity")

    if set(flags).intersection(LIQUIDITY_FLAG_NAMES):
        concerns.append("liquidity")

    if "external_rating_speculative_grade" in flags:
        concerns.append("external speculative-grade rating indicators")

    positive_text = ", ".join(dict.fromkeys(positives)) or "selected financial strengths"
    concern_text = ", ".join(dict.fromkeys(concerns)) or "selected financial diagnostics"

    return (
        f"The cluster assignment appears primarily supported by {positive_text}. "
        f"However, {concern_text} constrain the interpretation. The result should "
        "therefore be read as a relative model indication rather than an unqualified "
        "credit-strength conclusion."
    )



def _build_commercial_conclusion(level: str, flags: Sequence[str], row: pd.Series) -> str:
    """Build client-safe commercial conclusion paragraph."""
    if level == "Clear":
        return (
            "The model output can be presented as a relative financial-risk diagnostic, "
            "subject to normal review of input quality, business context, and model limitations."
        )

    if level == "Monitor":
        return (
            "The result remains usable as a relative risk diagnostic, but the flagged items "
            "should be monitored and explained in any client-facing discussion."
        )

    if level == "Caution":
        return (
            "The result should be presented with explicit caveats. Further review of debt "
            "maturity, covenants, refinancing risk, working-capital dynamics, and cash-flow "
            "sustainability is recommended before relying on the output for a commercial decision."
        )

    if level == "High caution":
        return (
            "The model output may be directionally useful, but the flagged weaknesses are "
            "material. The conclusion should be manually reviewed and should not be framed "
            "as a clean low-risk or investment-grade-like result."
        )

    return (
        "Manual analyst override is required. The model output should not be used as a "
        "standalone conclusion without detailed review of the financial statements, debt "
        "structure, liquidity, and business context."
    )



def _apply_guardrails_to_row(
    row: pd.Series,
    external_rating: str | None = None,
) -> pd.Series:
    """Apply all configured guardrails to one scored row."""
    triggered = _evaluate_ratio_guardrails(row)
    flags = [flag for flag, _ in triggered]

    triggered.extend(_evaluate_model_contradiction_guardrails(row, flags=flags))
    flags = [flag for flag, _ in triggered]

    triggered.extend(_evaluate_external_rating_guardrails(external_rating))
    flags = [flag for flag, _ in triggered]

    severities = [severity for _, severity in triggered]
    level = _highest_severity(severities)

    # Remove duplicate flags while preserving order.
    unique_flags = list(dict.fromkeys(flags))

    row["guardrail_level"] = level
    row["guardrail_flags"] = ", ".join(unique_flags)
    row["guardrail_summary"] = _build_guardrail_summary(level, unique_flags, row)
    row["analyst_interpretation"] = _build_analyst_interpretation(level, unique_flags, row)
    row["commercial_conclusion"] = _build_commercial_conclusion(level, unique_flags, row)

    return row


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def apply_credit_guardrails(
    scored_df: pd.DataFrame,
    external_rating: str | None = None,
) -> pd.DataFrame:
    """
    Add consulting-grade guardrail columns to a scored dataframe.

    Parameters
    ----------
    scored_df:
        Output from score_companies(), ideally after outlook diagnostics have
        been added with add_adjacent_bucket_distances_and_outlook().
    external_rating:
        Optional external rating text such as 'Ba2 / BB'.  If supplied and it
        appears speculative-grade, an external-rating caution flag is added.

    Returns
    -------
    pd.DataFrame
        Copy of scored_df with the following columns added:
        - guardrail_level
        - guardrail_flags
        - guardrail_summary
        - analyst_interpretation
        - commercial_conclusion
    """
    if scored_df is None:
        raise ValueError("scored_df cannot be None.")

    out = scored_df.copy().reset_index(drop=True)

    if out.empty:
        for col in GUARDRAIL_COLS:
            out[col] = pd.Series(dtype="object")
        return out

    out = out.apply(
        lambda row: _apply_guardrails_to_row(
            row,
            external_rating=external_rating,
        ),
        axis=1,
    )

    return out


__all__ = [
    "GUARDRAIL_COLS",
    "apply_credit_guardrails",
]
