import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/billing_unified.csv")
OUTPUT_PATH = Path("data/processed/billing_features.csv")


def build_features():
    df = pd.read_csv(INPUT_PATH)

    # ----------------------------
    # BASIC CLEANUP
    # ----------------------------
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["usage_date"] = pd.to_datetime(df["usage_date"], errors="coerce")

    # ----------------------------
    # 1. RAW BILLING FEATURES
    # ----------------------------
    df["quantity"] = df["quantity"].fillna(0)
    df["discount_pct"] = df["discount_pct"].fillna(0)

    # ----------------------------
    # 2. CONTRACT vs BILLED DEVIATION
    # ----------------------------
    df["price_gap_contract"] = df["unit_price"] - df["contract_price"]
    df["price_gap_contract"] = df["price_gap_contract"].fillna(0)

    df["discount_violation"] = (
        (df["discount_pct"] > df["max_discount_pct"]) &
        (~df["off_contract"])
    ).astype(int)

    # ----------------------------
    # 3. USAGE vs BILLING DEVIATION
    # ----------------------------
    df["actual_usage"] = df["actual_usage"].fillna(0)

    df["usage_gap"] = df["actual_usage"] - df["quantity"]

    df["usage_ratio"] = df["quantity"] / df["actual_usage"].replace(0, 1)

    # ----------------------------
    # 4. CUSTOMER HISTORICAL BASELINES
    # ----------------------------
    df["cust_avg_unit_price"] = (
        df.groupby("customer_id")["unit_price"]
        .transform("mean")
    )

    df["cust_avg_quantity"] = (
        df.groupby("customer_id")["quantity"]
        .transform("mean")
    )

    df["cust_avg_discount"] = (
        df.groupby("customer_id")["discount_pct"]
        .transform("mean")
    )

    df["unit_price_vs_cust_avg"] = (
        df["unit_price"] - df["cust_avg_unit_price"]
    )

    # ----------------------------
    # 5. TIME & SEASONALITY FEATURES
    # ----------------------------
    df["invoice_month"] = df["invoice_date"].dt.month
    df["invoice_dayofweek"] = df["invoice_date"].dt.dayofweek

    df["invoice_age_days"] = (
        (pd.Timestamp.today() - df["invoice_date"])
        .dt.days
    )

    # ----------------------------
    # 6. FLAGS AS FEATURES
    # ----------------------------
    df["off_contract"] = df["off_contract"].astype(int)
    df["usage_missing"] = df["usage_missing"].astype(int)
    df["pricing_missing"] = df["pricing_missing"].astype(int)

    # ----------------------------
    # FINAL FEATURE SET
    # ----------------------------
    feature_cols = [
        # Raw
        "unit_price", "quantity", "discount_pct",

        # Contract
        "price_gap_contract", "discount_violation", "off_contract",

        # Usage
        "usage_gap", "usage_ratio", "usage_missing",

        # Customer baselines
        "cust_avg_unit_price", "cust_avg_quantity",
        "cust_avg_discount", "unit_price_vs_cust_avg",

        # Time
        "invoice_month", "invoice_dayofweek", "invoice_age_days",

        # Pricing
        "pricing_missing"
    ]

    features = df[["invoice_id"] + feature_cols]

    features.to_csv(OUTPUT_PATH, index=False)
    print(f"Feature matrix saved → {OUTPUT_PATH}")
    print(f"Rows: {features.shape[0]}, Features: {features.shape[1] - 1}")


if __name__ == "__main__":
    build_features()
