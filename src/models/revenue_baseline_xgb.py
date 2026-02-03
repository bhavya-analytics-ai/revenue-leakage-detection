import pandas as pd
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# ---------------- PATHS ----------------
FEATURES_PATH = Path("data/processed/billing_features.csv")
UNIFIED_PATH = Path("data/processed/billing_unified.csv")
OUTPUT_PATH = Path("data/processed/revenue_baseline_estimates.csv")
MODEL_PATH = Path("models/revenue_xgb_baseline.joblib")

# ---------------- CONFIG ----------------
SEED = 42
TEST_SIZE = 0.2


def main():
    # ---------------- LOAD ----------------
    features = pd.read_csv(FEATURES_PATH)
    unified = pd.read_csv(UNIFIED_PATH)

    # ---------------- MERGE TARGET ----------------
    df = features.merge(
        unified[["invoice_id", "billed_amount"]],
        on="invoice_id",
        how="left"
    )

    # ---------------- FEATURES & TARGET ----------------
    feature_cols = [
        "quantity",
        "unit_price",
        "discount_pct",
        "price_gap_contract",
        "usage_gap",
        "usage_ratio",
        "cust_avg_unit_price",
        "cust_avg_quantity",
        "cust_avg_discount",
        "unit_price_vs_cust_avg",
        "invoice_month",
        "invoice_dayofweek",
        "invoice_age_days",
    ]

    X = df[feature_cols].fillna(0)
    y = df["billed_amount"]

    # ---------------- SPLIT ----------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    # ---------------- MODEL ----------------
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1,
    )

    # ---------------- TRAIN ----------------
    model.fit(X_train, y_train)

    # ---------------- EVALUATE ----------------
    val_preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_preds)
    print(f"Baseline XGBoost MAE: {mae:.2f}")

    # ---------------- FULL INFERENCE ----------------
    df["expected_revenue_baseline"] = model.predict(X)
    df["leakage_baseline"] = df["expected_revenue_baseline"] - df["billed_amount"]

    # ---------------- SAVE OUTPUT ----------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Baseline revenue estimates saved → {OUTPUT_PATH}")

    # ---------------- SAVE MODEL ----------------
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"XGBoost baseline model saved → {MODEL_PATH}")


if __name__ == "__main__":
    main()
