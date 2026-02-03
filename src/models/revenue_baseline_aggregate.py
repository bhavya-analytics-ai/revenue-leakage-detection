import pandas as pd
from pathlib import Path

# ---------------- PATHS ----------------
INPUT_PATH = Path("data/processed/revenue_baseline_estimates.csv")
OUTPUT_PATH = Path("data/processed/revenue_baseline_invoice_level.csv")


def main():
    df = pd.read_csv(INPUT_PATH)

    # ---------------- AGGREGATE ----------------
    invoice_df = (
        df.groupby("invoice_id")
        .agg(
            billed_amount=("billed_amount", "first"),
            expected_revenue_baseline=("expected_revenue_baseline", "mean"),
        )
        .reset_index()
    )

    # ---------------- COMPUTE LEAKAGE ----------------
    invoice_df["leakage_baseline"] = (
        invoice_df["expected_revenue_baseline"]
        - invoice_df["billed_amount"]
    )

    # ---------------- SORT (MOST AT RISK FIRST) ----------------
    invoice_df = invoice_df.sort_values(
        "leakage_baseline", ascending=False
    )

    # ---------------- SAVE ----------------
    invoice_df.to_csv(OUTPUT_PATH, index=False)

    print("Invoice-level baseline aggregation complete.")
    print(f"Saved → {OUTPUT_PATH}")
    print("Top 10 invoices by baseline leakage:")
    print(invoice_df.head(10))


if __name__ == "__main__":
    main()
