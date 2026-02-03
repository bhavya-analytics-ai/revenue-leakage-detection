import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Paths
EXPLAINED = "data/processed/explained_leakage_cases.csv"
FEATURES = "data/processed/billing_features.csv"
OUT = "data/processed/leakage_patterns.csv"

# 1) Load data
explained = pd.read_csv(EXPLAINED)
features = pd.read_csv(FEATURES)

# 2) Aggregate billing features to invoice level (mean)
agg = (
    features
    .groupby("invoice_id", as_index=False)
    .agg({
        "unit_price": "mean",
        "quantity": "mean",
        "discount_pct": "mean",
        "usage_ratio": "mean",
    })
    .rename(columns={
        "unit_price": "unit_price_mean",
        "quantity": "quantity_mean",
        "discount_pct": "discount_pct_mean",
        "usage_ratio": "usage_ratio_mean",
    })
)

# 3) Merge with explained leakage cases
df = explained.merge(agg, on="invoice_id", how="left")

# 4) Select clustering features
cluster_cols = [
    "leakage_baseline",
    "unit_price_mean",
    "quantity_mean",
    "discount_pct_mean",
    "usage_ratio_mean",
]

X = df[cluster_cols].fillna(0.0)

# 5) Scale + KMeans (K=3)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["leakage_cluster_id"] = kmeans.fit_predict(X_scaled)
pattern_map = {
    0: "Usage Underbilling",
    1: "Pricing / Rate Mismatch",
    2: "Discount-Driven Leakage",
}

df["leakage_pattern"] = df["leakage_cluster_id"].map(pattern_map)


# 6) Save
df.to_csv(OUT, index=False)

print("[Level 7] Wrote:", OUT)
print(df[["invoice_id", "leakage_cluster_id"]].head())
