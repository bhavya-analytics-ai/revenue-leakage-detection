import pandas as pd
import numpy as np
from datetime import timedelta

np.random.seed(42)

N_CUSTOMERS = 50
N_PRODUCTS = 10
N_INVOICES = 3000   # increase size

customers = [f"C{i}" for i in range(N_CUSTOMERS)]
products = [f"P{i}" for i in range(N_PRODUCTS)]

# Pricing
pricing = pd.DataFrame({
    "product_id": products,
    "list_price": np.random.uniform(50, 200, N_PRODUCTS).round(2)
})

# Contracts
contracts = []
for c in customers:
    for p in np.random.choice(products, size=3, replace=False):
        list_price = pricing.loc[pricing.product_id == p, "list_price"].values[0]
        contracts.append({
            "customer_id": c,
            "product_id": p,
            "contract_price": round(list_price * np.random.uniform(0.85, 0.95), 2),
            "max_discount_pct": np.random.choice([5, 10, 15]),
            "contract_start": "2024-01-01",
            "contract_end": "2025-12-31"
        })

contracts = pd.DataFrame(contracts)

# Usage (ONE usage row = ONE invoice)
usage = []
for i in range(N_INVOICES):
    usage.append({
        "customer_id": np.random.choice(customers),
        "product_id": np.random.choice(products),
        "usage_date": pd.Timestamp("2024-06-01") + timedelta(days=np.random.randint(0, 180)),
        "actual_usage": np.random.poisson(20)
    })

usage = pd.DataFrame(usage)

# Invoices (NO DROPPING, EVER)
invoices = []

for i, row in usage.iterrows():

    contract = contracts[
        (contracts.customer_id == row.customer_id) &
        (contracts.product_id == row.product_id)
    ]

    if len(contract) > 0:
        contract = contract.iloc[0]
        unit_price = contract.contract_price
        max_discount = contract.max_discount_pct
    else:
        unit_price = pricing.loc[
            pricing.product_id == row.product_id, "list_price"
        ].values[0]
        max_discount = 0

    quantity = row.actual_usage

    # Inject leakage
    if np.random.rand() < 0.1:
        quantity = max(1, int(quantity * np.random.uniform(0.4, 0.7)))

    discount = np.random.choice([0, 5, 10, 20])
    discount = min(discount, max_discount)

    billed_amount = quantity * unit_price * (1 - discount / 100)

    invoices.append({
        "invoice_id": f"INV{i}",
        "customer_id": row.customer_id,
        "product_id": row.product_id,
        "invoice_date": row.usage_date,
        "quantity": quantity,
        "unit_price": round(unit_price, 2),
        "discount_pct": discount,
        "billed_amount": round(billed_amount, 2)
    })

invoices = pd.DataFrame(invoices)

# Save
pricing.to_csv("data/raw/pricing.csv", index=False)
contracts.to_csv("data/raw/contracts.csv", index=False)
usage.to_csv("data/raw/usage.csv", index=False)
invoices.to_csv("data/raw/invoices.csv", index=False)

print("Invoices:", len(invoices))
print("Synthetic data generated successfully.")
