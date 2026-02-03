from __future__ import annotations

import argparse
import os

import joblib
import pandas as pd

from .shap_explainer import (
    prepare_feature_matrix,
    compute_shap_values_tree,
    aggregate_invoice_level_shap,
)
from .prompt_builder import build_rule_violation_summary
from .llm_agent import generate_explanations, LLMConfig


DEFAULT_VALIDATED = "data/processed/validated_leakage_cases.csv"
DEFAULT_BASELINE = "data/processed/revenue_baseline_invoice_level.csv"
DEFAULT_FEATURES = "data/processed/billing_features.csv"
DEFAULT_MODEL = "models/revenue_xgb_baseline.joblib"
DEFAULT_OUT = "data/processed/explained_leakage_cases.csv"


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description="Level 6 - Explainability + Explanations")
    parser.add_argument("--validated", default=DEFAULT_VALIDATED)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--features", default=DEFAULT_FEATURES)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--mode", default="template", choices=["template", "openai"])
    args = parser.parse_args()

    # 1) Load core files
    validated = load_csv(args.validated)
    baseline = load_csv(args.baseline)
    billing_features = load_csv(args.features)

    # Normalize invoice_id as string
    for df in (validated, baseline, billing_features):
        if "invoice_id" not in df.columns:
            raise ValueError("All inputs must include 'invoice_id'")
        df["invoice_id"] = df["invoice_id"].astype(str)

    # 2) Load model
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Missing model: {args.model}")
    model = joblib.load(args.model)

    # 3) SHAP at row-level -> aggregate to invoice-level
    X, invoice_ids = prepare_feature_matrix(billing_features, model, invoice_id_col="invoice_id")
    shap_values, _ = compute_shap_values_tree(model, X)
    shap_invoice = aggregate_invoice_level_shap(
        shap_values=shap_values,
        X=X,
        invoice_ids=invoice_ids,
        top_k=args.top_k,
    )

    # 4) Merge invoice-level artifacts
    # Keep only validated leakage invoices if you want — but better to merge and then filter
    merged = baseline.merge(validated, on="invoice_id", how="left").merge(shap_invoice, on="invoice_id", how="left")

    # 5) Rule violation summary string
    merged["rule_violations"] = merged.apply(build_rule_violation_summary, axis=1)

    # 6) Explanation text (template default; openai optional)
    cfg = LLMConfig(mode=args.mode)
    merged["explanation_text"] = generate_explanations(merged, cfg=cfg)

    # 7) Optional filter: keep only validated leakage cases True
    if "validated_leakage" in merged.columns:
        merged = merged[merged["validated_leakage"].fillna(False) == True].copy()

    # 8) Save
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    merged.to_csv(args.out, index=False)
    print(f"[Level 6] Wrote: {args.out}")
    print(f"[Level 6] Rows: {len(merged)}")
    print("[Level 6] Sample:")
    cols_preview = [c for c in [
        "invoice_id",
        "leakage_baseline",
        "rule_violations",
        "top_shap_features",
        "top_shap_impacts",
        "explanation_text",
    ] if c in merged.columns]
    print(merged[cols_preview].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
