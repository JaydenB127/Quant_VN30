# -*- coding: utf-8 -*-
"""
SHAP explainability for LightGBM models.

Generates SHAP summary and dependence plots to interpret model decisions
globally and by regime.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ..config import REPORT_DIR

logger = logging.getLogger(__name__)


def generate_shap_summary(
    model: object,
    X: pd.DataFrame,
    max_display: int = 20,
    title: str = "SHAP Feature Importance",
    save_path: Optional[Path] = None,
) -> Path:
    """
    Generate a SHAP summary plot (beeswarm).

    Parameters
    ----------
    model : object
        A trained LightGBM model (the underlying lgb.Booster).
    X : pd.DataFrame
        Feature matrix (a representative sample, e.g., validation set).
    max_display : int
        Number of top features to show.
    """
    logger.info("Computing SHAP values for summary plot...")
    
    # Take a sample if X is too large to keep computation fast
    if len(X) > 5000:
        X_sample = X.sample(5000, random_state=42)
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For LightGBM binary classification, shap_values might be a list of length 2
    # or a single array depending on the objective. We want the positive class.
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    fig, ax = plt.subplots(figsize=(10, max_display * 0.4 + 2))
    
    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=max_display,
        show=False,
        plot_size="auto",
    )
    
    plt.title(title, fontweight="bold", fontsize=14, pad=20)
    plt.tight_layout()

    save_path = save_path or REPORT_DIR / "shap_summary.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    
    logger.info("Saved SHAP summary to %s", save_path)
    return save_path


def generate_regime_shap_summary(
    model: object,
    X: pd.DataFrame,
    regime_series: pd.Series,
    max_display: int = 15,
    save_prefix: str = "shap_regime",
):
    """
    Generate SHAP summary plots stratified by regime.
    """
    regime_labels = {0: "Bull", 1: "Sideways", 2: "Bear"}
    
    for regime_id, label in regime_labels.items():
        mask = regime_series == regime_id
        if mask.sum() < 50:
            continue
            
        X_regime = X.loc[mask]
        
        save_path = REPORT_DIR / f"{save_prefix}_{label.lower()}.png"
        generate_shap_summary(
            model=model,
            X=X_regime,
            max_display=max_display,
            title=f"SHAP Summary — {label} Regime",
            save_path=save_path,
        )
