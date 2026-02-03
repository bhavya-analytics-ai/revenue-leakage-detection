from __future__ import annotations

import pandas as pd


RULE_FLAG_COLS = [
    "any_contract_violation",
    "any_discount_violation",
    "any_usage_underbilled",
    "any_price_norm_violation",
]


def build_rule_violation_summary(row: pd.Series) -> str:
    mapping = {
        "any_contract_violation": "contract violation",
        "any_discount_violation": "discount breach",
        "any_usage_underbilled": "usage underbilled",
        "any_price_norm_violation": "price vs norms violation",
    }

    triggered = []
    for col, label in mapping.items():
        if col in row and bool(row[col]) is True:
            triggered.append(label)

    if not triggered:
        return "no explicit rule violations"
    return ", ".join(triggered)


def build_template_explanation(row: pd.Series) -> str:
    """
    Deterministic explanation (default mode).
    Uses:
      - leakage_baseline
      - billed_amount vs expected_revenue_baseline
      - rule flags
      - SHAP top drivers
    """
    invoice_id = row.get("invoice_id", "UNKNOWN")
    billed = row.get("billed_amount", None)
    expected = row.get("expected_revenue_baseline", None)
    leakage = row.get("leakage_baseline", None)

    rule_summary = row.get("rule_violations", "no explicit rule violations")

    top_feats = str(row.get("top_shap_features", "") or "")
    top_imps = str(row.get("top_shap_impacts", "") or "")
    feats = top_feats.split("|") if top_feats else []
    imps = top_imps.split("|") if top_imps else []

    # Format drivers
    drivers = []
    for f, v in zip(feats[:5], imps[:5]):
        try:
            v_float = float(v)
            direction = "increased" if v_float > 0 else "decreased"
            drivers.append(f"{f} ({direction} expected charge)")
        except Exception:
            drivers.append(f"{f}")

    drivers_txt = "; ".join(drivers) if drivers else "SHAP drivers unavailable"

    # Format numeric summary safely
    def _fmt_money(x):
        try:
            return f"${float(x):,.2f}"
        except Exception:
            return "N/A"

    billed_txt = _fmt_money(billed)
    expected_txt = _fmt_money(expected)
    leakage_txt = _fmt_money(leakage)

    explanation = (
        f"Invoice {invoice_id} appears underbilled by ~{leakage_txt} versus the model baseline "
        f"(billed {billed_txt} vs expected {expected_txt}). "
        f"Validation signals: {rule_summary}. "
        f"Primary model drivers: {drivers_txt}. "
        f"Recommended action: verify contract rate/discount application and re-rate usage for this invoice if confirmed."
    )
    return explanation
