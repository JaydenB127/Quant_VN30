# -*- coding: utf-8 -*-
"""
Performance tables for regime-stratified analysis.
"""
from __future__ import annotations
import pandas as pd
from typing import List
from ..validation.walk_forward import FoldResult, aggregate_fold_results
from ..validation.metrics import classification_metrics


def ablation_study_table(fold_results: List[FoldResult]) -> pd.DataFrame:
    """
    Create ablation study table comparing model variants.

    Returns
    -------
    pd.DataFrame
        Rows = models, columns = metrics (mean ± std across folds).
    """
    df = aggregate_fold_results(fold_results)

    summary = {}
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        row = {}
        for metric in ["auc_roc", "f1_score", "accuracy", "precision", "recall"]:
            if metric in model_df.columns:
                mean = model_df[metric].mean()
                std = model_df[metric].std()
                row[metric] = f"{mean:.4f} ± {std:.4f}"
        summary[model] = row

    return pd.DataFrame(summary).T


def regime_performance_table(
    fold_results: List[FoldResult],
) -> pd.DataFrame:
    """
    Regime-stratified performance table (ensemble model).

    Returns
    -------
    pd.DataFrame
        Rows = regimes, columns = metrics.
    """
    regime_labels = {0: "Bull", 1: "Sideways", 2: "Bear"}
    regime_metrics = {r: {"y_true": [], "y_pred": [], "y_proba": []}
                      for r in [0, 1, 2]}

    for r in fold_results:
        if r.test_regimes is None or r.y_true is None:
            continue
        for regime_id in [0, 1, 2]:
            mask = r.test_regimes == regime_id
            if mask.sum() > 0:
                regime_metrics[regime_id]["y_true"].extend(r.y_true[mask])
                regime_metrics[regime_id]["y_pred"].extend(r.y_pred_ensemble[mask])
                regime_metrics[regime_id]["y_proba"].extend(r.y_proba_ensemble[mask])

    results = {}
    for regime_id, data in regime_metrics.items():
        if len(data["y_true"]) > 0:
            import numpy as np
            y_true = np.array(data["y_true"])
            y_pred = np.array(data["y_pred"])
            y_proba = np.array(data["y_proba"])
            metrics = classification_metrics(y_true, y_pred, y_proba)
            metrics["n_samples"] = len(y_true)
            metrics["pos_rate"] = y_true.mean()
            results[regime_labels.get(regime_id, f"State {regime_id}")] = metrics

    return pd.DataFrame(results).T


def statistical_summary_table(dm_results: dict, bootstrap_results: dict) -> pd.DataFrame:
    """
    Create statistical significance summary table.

    Parameters
    ----------
    dm_results : dict
        {comparison_name: {dm_statistic, p_value}}
    bootstrap_results : dict
        {metric_name: {point_estimate, ci_lower, ci_upper}}

    Returns
    -------
    pd.DataFrame
    """
    rows = []

    for name, dm in dm_results.items():
        rows.append({
            "test": f"DM: {name}",
            "statistic": dm.get("dm_statistic"),
            "p_value": dm.get("p_value"),
            "significant": dm.get("p_value", 1) < 0.05,
        })

    for name, bs in bootstrap_results.items():
        rows.append({
            "test": f"Bootstrap 95% CI: {name}",
            "statistic": bs.get("point_estimate"),
            "ci_lower": bs.get("ci_lower"),
            "ci_upper": bs.get("ci_upper"),
            "significant": bs.get("ci_lower", 0) > 0,
        })

    return pd.DataFrame(rows)


def model_comparison_table(fold_results: List[FoldResult]) -> pd.DataFrame:
    """
    Extended model comparison table for paper — includes model type labels.

    Returns
    -------
    pd.DataFrame
        Rows = models (with type), columns = metrics.
    """
    import numpy as np
    df = aggregate_fold_results(fold_results)

    MODEL_TYPES = {
        "base": "GBDT (Base)",
        "transfer": "GBDT (Transfer)",
        "ensemble": "Ensemble (Ours)",
        "static": "GBDT (Static)",
        "xgb": "GBDT (XGBoost)",
        "lstm": "DL (LSTM)",
        "transformer": "DL (Transformer)",
    }

    summary = {}
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        row = {"type": MODEL_TYPES.get(model, model)}
        for metric in ["auc_roc", "f1_score", "accuracy", "precision", "recall", "log_loss"]:
            if metric in model_df.columns:
                vals = model_df[metric].dropna()
                if len(vals) > 0:
                    row[f"{metric}_mean"] = vals.mean()
                    row[f"{metric}_std"] = vals.std()
                    row[metric] = f"{vals.mean():.4f} ± {vals.std():.4f}"
        summary[model] = row

    result = pd.DataFrame(summary).T
    # Sort by AUC descending
    if "auc_roc_mean" in result.columns:
        result = result.sort_values("auc_roc_mean", ascending=False)

    return result

