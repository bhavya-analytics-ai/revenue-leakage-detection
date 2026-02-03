import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ---------------- PATHS ----------------
INPUT_PATH = Path("data/processed/billing_features.csv")
OUTPUT_PATH = Path("data/processed/billing_anomaly_scores.csv")

# ---------------- CONFIG ----------------
RANDOM_STATE = 42
CONTAMINATION = 0.05   # top 5% most abnormal

def run_anomaly_detection():
    # Load features
    df = pd.read_csv(INPUT_PATH)

    invoice_ids = df["invoice_id"]

    # Drop identifier column
    X = df.drop(columns=["invoice_id"])

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest
    iso = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE
    )

    iso.fit(X_scaled)

    # Scores (lower = more anomalous)
    anomaly_score = iso.decision_function(X_scaled)
    anomaly_label = iso.predict(X_scaled)  # -1 = anomaly, 1 = normal

    results = pd.DataFrame({
        "invoice_id": invoice_ids,
        "anomaly_score": anomaly_score,
        "is_anomaly": (anomaly_label == -1).astype(int)
    })

    # Rank: 1 = most suspicious
    results["anomaly_rank"] = (
        results["anomaly_score"]
        .rank(method="first")
        .astype(int)
    )

    results = results.sort_values("anomaly_rank")

    results.to_csv(OUTPUT_PATH, index=False)

    print("Anomaly detection complete.")
    print(f"Saved → {OUTPUT_PATH}")
    print("Top 5 suspicious invoices:")
    print(results.head(5))


if __name__ == "__main__":
    run_anomaly_detection()
