# 💰 Revenue Leakage Detection System

A production-style machine learning system to **detect, validate, quantify, and explain revenue leakage** in subscription / usage-based billing systems.

This project is intentionally built **level by level**, mirroring how real analytics and ML systems evolve in industry — starting from messy data ingestion to explainable, reviewable financial impact modeling.

---

## 🧭 Project Objective

Revenue leakage occurs when companies bill **less than they should** due to:

* pricing errors
* contract drift
* unauthorized discounts
* usage underbilling
* data integration issues

The goal of this system is to:

1. **Detect suspicious billing records**
2. **Validate them using business context**
3. **Estimate dollar impact**
4. **Explain why each case is flagged**
5. **Categorize recurring leakage patterns**
6. **Validate robustness under stress**

---

## 🔄 End-to-End Data Flow

```
Raw Data (CSV)
│
▼
[Level 1] Data Ingestion & Validation
│ └─ schema checks, joins, missing flags
▼
billing_unified.csv
│
▼
[Level 2] Feature Engineering
│ └─ pricing gaps, usage deviation, customer history
▼
billing_features.csv
│
▼
[Level 3] Anomaly Detection (Isolation Forest)
│ └─ unsupervised risk scoring
▼
billing_anomaly_scores.csv
│
▼
[Level 4] Context-Aware Validation (Rules + Stats)
│ └─ contract rules + behavioral checks
▼
validated_leakage_cases.csv
│
▼
[Level 5] Revenue Impact Modeling (XGBoost)
│ └─ expected revenue vs billed
▼
revenue_baseline_invoice_level.csv
│
▼
[Level 6] Explainability Layer
│ └─ SHAP attributions + rule context + narratives
▼
explained_leakage_cases.csv
│
▼
[Level 7] Pattern Discovery (Clustering)
│ └─ systemic leakage categorization
▼
leakage_patterns.csv
│
▼
[Level 8] Review Interface (Streamlit)
│ └─ filter, inspect, export
▼
Human Review
│
▼
[Level 9] Stress Testing & Reliability Evaluation
```

---

## 🧠 Model Flow

```
Billing Features
│
├─▶ Isolation Forest ──▶ Anomaly Score
│
├─▶ Rule Engine + Stats ─▶ Validated Leakage
│
├─▶ XGBoost Regressor ─▶ Expected Revenue
│ │
│ ▼
│ Leakage = Expected − Billed
│
├─▶ SHAP Explainability ─▶ Feature Attribution
│
└─▶ Clustering (KMeans) ─▶ Leakage Pattern
```

---

## 🔢 Level-by-Level System Design

### ✅ LEVEL 1 — Data Ingestion & Validation

**Purpose:** Accept realistic, messy business data.

**What was done:**

* Loaded invoices, contracts, pricing, usage
* Key-based joins (`customer_id`, `product_id`, `invoice_id`)
* Missing-data flags for usage and pricing
* Error logging during ingestion

**Output:**

* `billing_unified.csv`

---

### ✅ LEVEL 2 — Feature Engineering

**Purpose:** Encode billing behavior for ML.

**Feature groups:**

* Unit price, quantity, discount %
* Contract vs billed price gaps
* Usage vs billing deviation
* Customer historical averages
* Time & seasonality signals

**Output:**

* `billing_features.csv`

---

### ✅ LEVEL 3 — Baseline Anomaly Detection (Core ML)

**Purpose:** Identify suspicious billing records.

**Model:**

* Isolation Forest (unsupervised)

**Output:**

* Row-level anomaly scores
* Ranked suspicious billing rows

**File:**

* `billing_anomaly_scores.csv`

---

### ✅ LEVEL 4 — Context-Aware Validation (Rules + ML)

**Purpose:** Reduce false positives using business logic.

**Validation logic:**

* Contract price violations
* Discount policy breaches
* Off-contract billing
* Deviation from customer norms

**Aggregation:**

* Row-level → invoice-level validation

**Output:**

* `validated_leakage_cases.csv`

---

### ✅ LEVEL 5 — Revenue Impact Modeling

**Purpose:** Quantify financial impact of leakage.

#### Level 5A — XGBoost Baseline (Production Model)

* Trained XGBoost regressor to estimate **expected revenue**
* Evaluation metric: **MAE = 12.82**
* Strong performance on structured tabular data
* Selected as **final production model**

**Outputs:**

* Row-level predictions: `revenue_baseline_estimates.csv`
* Invoice-level aggregation: `revenue_baseline_invoice_level.csv`
* Saved model: `models/revenue_xgb_baseline.joblib`

#### Level 5B — PyTorch Neural Benchmark

* Feedforward MLP implemented in PyTorch
* Used strictly as a benchmark
* Validation MAE ≈ **14.8** (worse than XGBoost)

**Conclusion:**
Tree-based models outperform neural networks for this billing problem.

---

## ✅ Current Status

* Levels **1–5 complete**
* End-to-end pipeline operational
* Models benchmarked and justified
* Business-interpretable outputs available

---

## 🔜 Next (Planned)

### LEVEL 6 — Explainability + LLM Agent

* SHAP-based feature explanations
* Human-readable leakage reasons
* LLM agent to answer:

  > "Why was this invoice flagged?"

---

## 📌 Key Takeaways

* Built like a **real production ML system**, not a toy project
* Explicit baselines and model comparisons
* Clear separation between detection, validation, and impact
* Decisions driven by metrics, not hype

---

*This README reflects the system state up to Level 5.*

### ✅ LEVEL 6 — Explainability & Analyst Narratives

Purpose: Make leakage alerts explainable and reviewable by humans.

Explainability logic:

* Feature-level SHAP attributions computed for the XGBoost revenue model
* SHAP values aggregated from row-level to invoice-level
* Model explanations combined with:
  * Rule violations from Level 4
  * Estimated leakage amount from Level 5

Explanation layer:

* Deterministic analyst-style explanation generated per invoice
* Explains:
  * How much revenue leaked
  * Which features drove the prediction
  * Which validation rules were triggered
  * What action should be taken

Output:

* explained_leakage_cases.csv

---

### ✅ LEVEL 7 — Pattern Discovery (Leakage Categorization)

Purpose: Identify recurring, systemic leakage patterns across invoices.

Pattern discovery logic:

* Applied only to validated and explainable leakage cases
* Billing behavior aggregated to invoice-level using mean values:
  * Unit price
  * Quantity
  * Discount percentage
  * Usage ratio
* Behavioral signals combined with leakage magnitude

Model:

* KMeans clustering (K = 3)

Leakage categories:

* Usage Underbilling
* Pricing / Rate Mismatch
* Discount-Driven Leakage

Output:

* leakage_patterns.csv

---

### ✅ LEVEL 8 — Usable Product Interface

Purpose: Make the system usable for internal teams.

Interface features:

* CSV upload of processed leakage results
* Filter invoices by dollar impact
* View invoice-level explanations
* Export filtered reports

Tool:

* Streamlit

Output:

* Local Streamlit application for internal review

---

### ✅ LEVEL 9 — Evaluation & Stress Testing

Purpose: Validate system robustness and reliability.

Evaluation logic:

* Injected synthetic leakage into a controlled subset of invoices
* Re-applied existing detection logic without retraining
* Created ground truth to assess detection performance

Metrics:

* Recall on injected leakage: 1.0
* False positive rate: ~6.5%

Output:

* level9_stress_test_results.csv

---

## 📁 Repository Structure (Actual)

```
revenue-leakage-detection/
│
├── app/
│ └── streamlit_app.py
│
├── assets/
│ ├── level8_ui_1.png
│ └── level8_ui_2.png
│
├── data/
│ ├── raw/
│ │ ├── contracts.csv
│ │ ├── invoices.csv
│ │ ├── pricing.csv
│ │ └── usage.csv
│ │
│ └── processed/
│ ├── billing_unified.csv
│ ├── billing_features.csv
│ ├── billing_anomaly_scores.csv
│ ├── validated_leakage_cases.csv
│ ├── explained_leakage_cases.csv
│ ├── leakage_patterns.csv
│ ├── revenue_baseline_estimates.csv
│ ├── revenue_baseline_invoice_level.csv
│ └── level9_stress_test_results.csv
│
├── models/
│ ├── revenue_xgb_baseline.joblib
│ └── revenue_model_torch.pt
│
├── src/
│ ├── data/
│ │ ├── generate_synthetic_data.py
│ │ ├── load_validate.py
│ │ └── merge_tables.py
│ │
│ ├── features/
│ │ └── build_features.py
│ │
│ ├── models/
│ │ ├── anomaly_detection.py
│ │ ├── context_validation.py
│ │ ├── revenue_baseline_xgb.py
│ │ ├── revenue_baseline_aggregate.py
│ │ ├── revenue_model_torch.py
│ │ ├── run_level7_pattern_discovery.py
│ │ └── run_level9_stress_test.py
│ │
│ └── explainability/
│ ├── shap_explainer.py
│ ├── prompt_builder.py
│ ├── llm_agent.py
│ └── run_level6_explainability.py
│
├── notebooks/ # EDA only
├── requirements.txt
└── README.md
```
