import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from pathlib import Path

# ---------------- PATHS ----------------
FEATURES_PATH = Path("data/processed/billing_features.csv")
UNIFIED_PATH = Path("data/processed/billing_unified.csv")
OUTPUT_PATH = Path("data/processed/revenue_torch_estimates.csv")
MODEL_PATH = Path("models/revenue_model_torch.pt")

# ---------------- CONFIG ----------------
SEED = 42
TEST_SIZE = 0.2
BATCH_SIZE = 128
EPOCHS = 40
LR = 1e-3

torch.manual_seed(SEED)


# ---------------- MODEL ----------------
class RevenueMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def main():
    # ---------------- LOAD ----------------
    features = pd.read_csv(FEATURES_PATH)
    unified = pd.read_csv(UNIFIED_PATH)

    df = features.merge(
        unified[["invoice_id", "billed_amount"]],
        on="invoice_id",
        how="left"
    )

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
    y = df["billed_amount"].values

    # ---------------- SCALE ----------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------- SPLIT ----------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=SEED
    )

    # ---------------- DATASETS ----------------
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # ---------------- INIT ----------------
    model = RevenueMLP(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # ---------------- TRAIN ----------------
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds = []
        val_true = []

        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                val_preds.extend(preds.squeeze().numpy())
                val_true.extend(yb.squeeze().numpy())

        mae = mean_absolute_error(val_true, val_preds)
        print(f"Epoch {epoch+1}/{EPOCHS} | Val MAE: {mae:.2f}")

    # ---------------- SAVE MODEL ----------------
    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    # ---------------- FULL INFERENCE ----------------
    model.eval()
    with torch.no_grad():
        expected = model(
            torch.tensor(X_scaled, dtype=torch.float32)
        ).squeeze().numpy()

    df["expected_revenue_torch"] = expected
    df["leakage_torch"] = df["expected_revenue_torch"] - df["billed_amount"]

    df.to_csv(OUTPUT_PATH, index=False)

    print("PyTorch revenue model complete.")
    print(f"Saved → {OUTPUT_PATH}")
    print(f"Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    main()
