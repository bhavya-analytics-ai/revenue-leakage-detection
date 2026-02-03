import pandas as pd
from pathlib import Path
from load_validate import load_and_validate_all

OUTPUT_PATH = Path("data/processed")
OUTPUT_PATH.mkdir(exist_ok=True)

def merge_all():
    data = load_and_validate_all()

    invoices = data["invoices"]
    contracts = data["contracts"]
    usage = data["usage"]
    pricing = data["pricing"]

    # ---------------- MERGES ----------------

    # Invoice ↔ Contract
    df = invoices.merge(
        contracts,
        on=["customer_id", "product_id"],
        how="left",
        suffixes=("", "_contract")
    )

    df["off_contract"] = df["contract_price"].isna()

    # Invoice ↔ Usage
    df = df.merge(
        usage,
        on=["customer_id", "product_id"],
        how="left",
        suffixes=("", "_usage")
    )

    df["usage_missing"] = df["actual_usage"].isna()

    # Invoice ↔ Pricing
    df = df.merge(
        pricing,
        on="product_id",
        how="left"
    )

    df["pricing_missing"] = df["list_price"].isna()

    # ---------------- FINAL CLEAN ----------------
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["usage_date"] = pd.to_datetime(df["usage_date"], errors="coerce")

    df.to_csv(OUTPUT_PATH / "billing_unified.csv", index=False)
    print("Unified dataset saved → data/processed/billing_unified.csv")


if __name__ == "__main__":
    merge_all()
