import streamlit as st
import pandas as pd

st.set_page_config(page_title="Revenue Leakage Dashboard", layout="wide")

st.title("💸 Revenue Leakage Review")

uploaded = st.file_uploader("Upload leakage results CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)

    required_cols = [
        "invoice_id",
        "leakage_baseline",
        "leakage_pattern",
        "explanation_text",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    st.sidebar.header("Filters")
    min_leakage = st.sidebar.number_input(
        "Minimum leakage ($)",
        min_value=0.0,
        value=0.0,
        step=5.0,
    )

    filtered = df[df["leakage_baseline"] >= min_leakage]

    st.subheader("Filtered Leakage Cases")
    st.write(f"Showing {len(filtered)} invoices")

    st.dataframe(
        filtered[
            [
                "invoice_id",
                "leakage_baseline",
                "leakage_pattern",
                "explanation_text",
            ]
        ],
        use_container_width=True,
    )

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered report",
        data=csv,
        file_name="leakage_report.csv",
        mime="text/csv",
    )
else:
    st.info("Upload `leakage_patterns.csv` to begin.")
