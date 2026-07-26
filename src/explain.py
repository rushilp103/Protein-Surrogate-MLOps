"""SHAP explainability utilities for the surrogate model (Step 5 / Step 6)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor


def build_tree_explainer(model: XGBRegressor) -> shap.TreeExplainer:
    """Create a TreeExplainer for a fitted XGBRegressor."""
    return shap.TreeExplainer(model)


def shap_contributions(
    explainer: shap.TreeExplainer,
    feature_row: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, Any]:
    """
    Compute SHAP values for a single feature row.

    Returns base value, per-feature contributions, and the additive prediction
    ``base_value + sum(contributions)``.
    """
    if feature_row.shape[0] != 1:
        raise ValueError("feature_row must contain exactly one sample")
    if list(feature_row.columns) != feature_names:
        raise ValueError("feature_row columns must match feature_names order")

    values = explainer.shap_values(feature_row)
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values)
    if values.ndim == 2:
        values = values[0]

    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.asarray(base).ravel()[0])
    else:
        base = float(base)

    contributions = {
        name: float(values[i]) for i, name in enumerate(feature_names)
    }
    return {
        "base_value": base,
        "shap_values": contributions,
        "prediction_from_shap": base + float(sum(contributions.values())),
    }
