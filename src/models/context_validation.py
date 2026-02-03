import pandas as pd
from pathlib import Path

# ---------------- PATHS ----------------
FEATURES_PATH = Path("data/processed/billing_features.csv")
ANOMALY_PATH = Path("data/processed/billing_anomaly_scores.csv")
UNIFIED_PATH = Path("data/processed/billing_unified.csv")

# Outputs
OUTPUT_ALL_INVOICES = Path("data/processed/invoice_validation_all.csv")
OUTPUT_VALIDATED_ONLY = Path("data/processed/validated_leakage_cases.csv")

# ---------------- CONFIG ----------------
# After invoice-level aggregation, keep top X% most anomalous invoices for review
INVOICE_ANOMALY_PERCENTILE = 0.05  # top 5% invoices by anomaly severity


def run_context_validation():
    features = pd.read_csv(FEATURES_PATH)
    anomalies = pd.read_csv(ANOMALY_PATH)
    unified = pd.read_csv(UNIFIED_PATH)

    # Base = Level 2 features + Level 3 anomaly scores
    df = features.merge(anomalies, on="invoice_id", how="inner")

    # Bring only fields not already in features (avoid column collisions)
    df = df.merge(
        unified[["invoice_id", "contract_price", "max_discount_pct", "actual_usage"]],
        on="invoice_id",
        how="left",
    )

    # ---------------- RULE CHECKS (ROW-LEVEL) ----------------
    df["rule_contract_price_violation"] = (
        df["contract_price"].notna() &
        (df["unit_price"] < df["contract_price"])
    )

    df["rule_discount_violation"] = (
        df["max_discount_pct"].notna() &
        (df["discount_pct"] > df["max_discount_pct"])
    )

    df["rule_usage_underbilled"] = (
        df["actual_usage"].notna() &
        (df["actual_usage"] > df["quantity"])
    )

    df["rule_price_vs_customer_norm"] = (
        df["unit_price_vs_cust_avg"] < -0.15 * df["cust_avg_unit_price"]
    )

    rule_cols = [
        "rule_contract_price_violation",
        "rule_discount_violation",
        "rule_usage_underbilled",
        "rule_price_vs_customer_norm",
    ]
    df["num_rules_triggered"] = df[rule_cols].sum(axis=1)

    # ---------------- AGGREGATE FIRST (INVOICE-LEVEL) ----------------
    inv = (
        df.groupby("invoice_id")
        .agg(
            # anomaly severity: smaller = more suspicious
            anomaly_score_min=("anomaly_score", "min"),

            # rule evidence
            max_rules_triggered=("num_rules_triggered", "max"),
            any_contract_violation=("rule_contract_price_violation", "max"),
            any_discount_violation=("rule_discount_violation", "max"),
            any_usage_underbilled=("rule_usage_underbilled", "max"),
            any_price_norm_violation=("rule_price_vs_customer_norm", "max"),
        )
        .reset_index()
    )

    # Final validation: must have >=2 rule types triggered somewhere on invoice
    inv["validated_leakage"] = (inv["max_rules_triggered"] >= 2)

    # Save ALL invoices (this is the “complete Level 4 view”)
    inv_sorted = inv.sort_values("anomaly_score_min", ascending=True)
    inv_sorted.to_csv(OUTPUT_ALL_INVOICES, index=False)

    # ---------------- PRIORITIZED REVIEW SUBSET (OPTIONAL) ----------------
    # Top X% invoices by anomaly severity (invoice-level)
    cutoff = inv_sorted["anomaly_score_min"].quantile(INVOICE_ANOMALY_PERCENTILE)
    prioritized = inv_sorted[inv_sorted["anomaly_score_min"] <= cutoff].copy()

    # Within prioritized, keep only validated leakage (reduced false positives)
    validated = prioritized[prioritized["validated_leakage"]].copy()
    validated.to_csv(OUTPUT_VALIDATED_ONLY, index=False)

    print("Level 4 complete (invoice-level).")
    print(f"All invoices saved → {OUTPUT_ALL_INVOICES} (rows={len(inv_sorted)})")
    print(f"Validated leakage (prioritized subset) → {OUTPUT_VALIDATED_ONLY} (rows={len(validated)})")
    print("Top 10 most suspicious invoices (invoice-level):")
    print(inv_sorted.head(10))


if __name__ == "__main__":
    run_context_validation()
