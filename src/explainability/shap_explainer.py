from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb


def _get_model_feature_names(model) -> list[str] | None:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "get_booster"):
        try:
            fn = model.get_booster().feature_names
            if fn:
                return list(fn)
        except Exception:
            pass
    return None


def prepare_feature_matrix(
    billing_features_df: pd.DataFrame,
    model,
    invoice_id_col: str = "invoice_id",
) -> tuple[pd.DataFrame, pd.Series]:
    if invoice_id_col not in billing_features_df.columns:
        raise ValueError(f"Missing '{invoice_id_col}'")

    invoice_ids = billing_features_df[invoice_id_col].astype(str)
    X = billing_features_df.drop(columns=[invoice_id_col]).copy()

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.fillna(0.0)

    model_features = _get_model_feature_names(model)
    if model_features is not None:
        missing = [c for c in model_features if c not in X.columns]
        if missing:
            raise ValueError(f"Missing model features: {missing}")
        X = X[model_features]

    return X, invoice_ids


def compute_shap_values_tree(model, X: pd.DataFrame):
    """
    Robust SHAP using XGBoost native pred_contribs with DMatrix.
    This WILL work.
    """
    booster = model.get_booster()

    dmatrix = xgb.DMatrix(X, feature_names=X.columns.tolist())

    shap_contribs = booster.predict(
        dmatrix,
        pred_contribs=True
    )

    # last column = bias
    shap_values = shap_contribs[:, :-1]

    return np.asarray(shap_values), None


def aggregate_invoice_level_shap(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    invoice_ids: pd.Series,
    top_k: int = 5,
) -> pd.DataFrame:
    if shap_values.shape[0] != X.shape[0]:
        raise ValueError("Row mismatch between SHAP and features")

    feature_names = list(X.columns)
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df["invoice_id"] = invoice_ids.values

    grouped = shap_df.groupby("invoice_id")[feature_names].mean()

    rows = []
    for inv_id, row in grouped.iterrows():
        impacts = row.values.astype(float)
        idx = np.argsort(-np.abs(impacts))[:top_k]

        rows.append({
            "invoice_id": inv_id,
            "top_shap_features": "|".join([feature_names[i] for i in idx]),
            "top_shap_impacts": "|".join([f"{impacts[i]:.6f}" for i in idx]),
        })

    return pd.DataFrame(rows)
