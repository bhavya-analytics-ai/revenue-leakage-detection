import pandas as pd
import logging
from pathlib import Path

# ---------------- CONFIG ----------------
RAW_DATA_PATH = Path("data/raw")
LOG_PATH = Path("data/processed/ingestion_errors.log")

REQUIRED_SCHEMAS = {
    "invoices": [
        "invoice_id", "customer_id", "product_id",
        "invoice_date", "quantity", "unit_price",
        "discount_pct", "billed_amount"
    ],
    "contracts": [
        "customer_id", "product_id",
        "contract_price", "max_discount_pct",
        "contract_start", "contract_end"
    ],
    "usage": [
        "customer_id", "product_id",
        "usage_date", "actual_usage"
    ],
    "pricing": [
        "product_id", "list_price"
    ]
}

# ---------------- LOGGING ----------------
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------- FUNCTIONS ----------------
def load_csv(name):
    path = RAW_DATA_PATH / f"{name}.csv"
    df = pd.read_csv(path)
    return df


def validate_schema(df, name):
    missing_cols = set(REQUIRED_SCHEMAS[name]) - set(df.columns)
    if missing_cols:
        logging.error(f"{name}: Missing columns {missing_cols}")
        raise ValueError(f"{name} missing columns: {missing_cols}")
    return df


def validate_values(df, name):
    if name == "invoices":
        bad = df[(df["quantity"] < 0) | (df["unit_price"] < 0)]
        if not bad.empty:
            logging.warning(f"invoices: {len(bad)} rows with negative values")

    if name == "usage":
        bad = df[df["actual_usage"] < 0]
        if not bad.empty:
            logging.warning(f"usage: {len(bad)} rows with negative usage")

    return df


def load_and_validate_all():
    data = {}

    for name in ["invoices", "contracts", "usage", "pricing"]:
        df = load_csv(name)
        df = validate_schema(df, name)
        df = validate_values(df, name)
        data[name] = df

    return data
