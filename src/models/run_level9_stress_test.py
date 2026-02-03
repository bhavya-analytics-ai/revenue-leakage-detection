import pandas as pd
import numpy as np

# Paths
INVOICE_BASELINE = "data/processed/revenue_baseline_invoice_level.csv"
OUT = "data/processed/level9_stress_test_results.csv"

np.random.seed(42)

# 1) Load baseline invoice data
df = pd.read_csv(INVOICE_BASELINE)

# Baseline invoices are mostly clean
df = df.copy()
df["synthetic_leakage"] = 0.0
df["is_synthetic"] = False

# 2) Inject synthetic leakage into 10% of invoices
n_inject = int(0.10 * len(df))
inject_idx = np.random.choice(df.index, size=n_inject, replace=False)

# Leakage = 5–15% of expected revenue
leakage_pct = np.random.uniform(0.05, 0.15, size=n_inject)
df.loc[inject_idx, "synthetic_leakage"] = (
    df.loc[inject_idx, "expected_revenue_baseline"] * leakage_pct
)
df.loc[inject_idx, "billed_amount"] = (
    df.loc[inject_idx, "expected_revenue_baseline"]
    - df.loc[inject_idx, "synthetic_leakage"]
)
df.loc[inject_idx, "is_synthetic"] = True

# 3) Detection rule (simple, threshold)
DETECTION_THRESHOLD = 20.0  # dollars

df["detected"] = (
    (df["expected_revenue_baseline"] - df["billed_amount"]) >= DETECTION_THRESHOLD
)

# 4) Metrics
true_positives = ((df["detected"]) & (df["is_synthetic"])).sum()
false_negatives = ((~df["detected"]) & (df["is_synthetic"])).sum()
false_positives = ((df["detected"]) & (~df["is_synthetic"])).sum()

recall = true_positives / (true_positives + false_negatives + 1e-9)
false_positive_rate = false_positives / max((~df["is_synthetic"]).sum(), 1)

print("[Level 9] Recall on injected leakage:", round(recall, 3))
print("[Level 9] False positive rate:", round(false_positive_rate, 3))

# 5) Save results
df.to_csv(OUT, index=False)
print("[Level 9] Wrote:", OUT)
