# -*- coding: utf-8 -*-
"""
Visualization: equity curves, regime overlays, SHAP plots, performance charts.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from ..config import OUTPUT_DIR, REPORT_DIR

logger = logging.getLogger(__name__)

# Style
plt.rcParams.update({
    "figure.figsize": (14, 7),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
REGIME_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
REGIME_LABELS = {0: "Bull", 1: "Sideways", 2: "Bear"}


def plot_equity_curve_with_regime(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    regime_series: pd.Series,
    title: str = "Strategy vs Benchmark — Equity Curve",
    save_path: Optional[Path] = None,
) -> Path:
    """
    Plot cumulative equity curve with regime overlay background.
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    # Equity curves
    strat_cum = (1 + strategy_returns).cumprod()
    bench_cum = (1 + benchmark_returns).cumprod()

    ax.plot(strat_cum.index, strat_cum.values, label="Strategy (Regime-TL)",
            color="#3498db", linewidth=2)
    ax.plot(bench_cum.index, bench_cum.values, label="Benchmark (Equal-Weight)",
            color="#95a5a6", linewidth=1.5, alpha=0.7)

    # Regime background
    dates = regime_series.index
    for i in range(len(dates) - 1):
        regime = regime_series.iloc[i]
        color = REGIME_COLORS.get(int(regime), "#cccccc")
        ax.axvspan(dates[i], dates[i + 1], alpha=0.08, color=color)

    # Legend for regimes
    from matplotlib.patches import Patch
    regime_patches = [
        Patch(facecolor=c, alpha=0.3, label=REGIME_LABELS[k])
        for k, c in REGIME_COLORS.items()
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + regime_patches, loc="upper left", fontsize=10)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Cumulative Return")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = save_path or REPORT_DIR / "equity_curve_regime.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved equity curve to %s", save_path)
    return save_path


def plot_fold_comparison(
    fold_metrics_df: pd.DataFrame,
    metric: str = "auc_roc",
    save_path: Optional[Path] = None,
) -> Path:
    """Bar chart comparing model performance across folds."""
    fig, ax = plt.subplots(figsize=(14, 6))

    models = fold_metrics_df["model"].unique()
    n_folds = fold_metrics_df["fold"].nunique()
    x = np.arange(n_folds)
    width = 0.8 / len(models)

    colors = {"base": "#3498db", "transfer": "#e74c3c",
              "ensemble": "#2ecc71", "static": "#95a5a6"}

    for i, model in enumerate(models):
        vals = fold_metrics_df[fold_metrics_df["model"] == model][metric].values
        if len(vals) < n_folds:
            vals = np.pad(vals, (0, n_folds - len(vals)), constant_values=np.nan)
        ax.bar(x + i * width, vals, width, label=model.capitalize(),
               color=colors.get(model, "#333333"), alpha=0.85)

    ax.set_xlabel("Fold")
    ax.set_ylabel(metric.upper().replace("_", " "))
    ax.set_title(f"Walk-Forward Performance: {metric.upper()}", fontweight="bold")
    ax.set_xticks(x + width * len(models) / 2)
    ax.set_xticklabels([f"Fold {i}" for i in range(n_folds)])
    ax.legend()
    plt.tight_layout()

    save_path = save_path or REPORT_DIR / f"fold_comparison_{metric}.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved fold comparison to %s", save_path)
    return save_path


def plot_regime_distribution(
    regime_series: pd.Series,
    save_path: Optional[Path] = None,
) -> Path:
    """Plot regime distribution over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8),
                                     gridspec_kw={"height_ratios": [3, 1]})

    # Timeline
    for regime in sorted(regime_series.unique()):
        mask = regime_series == regime
        ax1.fill_between(
            regime_series.index, 0, 1, where=mask,
            color=REGIME_COLORS.get(int(regime), "#ccc"),
            alpha=0.5, label=REGIME_LABELS.get(int(regime), f"State {regime}"),
        )
    ax1.set_ylabel("Regime")
    ax1.set_title("Market Regime Over Time", fontweight="bold")
    ax1.legend(loc="upper right")

    # Distribution pie
    counts = regime_series.value_counts().sort_index()
    colors = [REGIME_COLORS.get(int(k), "#ccc") for k in counts.index]
    labels = [REGIME_LABELS.get(int(k), f"State {k}") for k in counts.index]
    ax2.barh(labels, counts.values, color=colors, alpha=0.8)
    ax2.set_xlabel("Trading Days")
    ax2.set_title("Regime Distribution", fontweight="bold")
    plt.tight_layout()

    save_path = save_path or REPORT_DIR / "regime_distribution.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved regime distribution to %s", save_path)
    return save_path


def plot_feature_importance(
    importance: pd.Series,
    top_n: int = 20,
    save_path: Optional[Path] = None,
) -> Path:
    """Horizontal bar chart of top feature importances."""
    fig, ax = plt.subplots(figsize=(10, 8))

    top = importance.head(top_n).sort_values()
    colors = sns.color_palette("viridis", len(top))
    ax.barh(range(len(top)), top.values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index)
    ax.set_xlabel("Importance (Gain)")
    ax.set_title(f"Top {top_n} Feature Importance", fontweight="bold")
    plt.tight_layout()

    save_path = save_path or REPORT_DIR / "feature_importance.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature importance to %s", save_path)
    return save_path


def plot_drawdown(
    returns: pd.Series,
    save_path: Optional[Path] = None,
) -> Path:
    """Plot drawdown chart."""
    fig, ax = plt.subplots(figsize=(14, 5))

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max

    ax.fill_between(drawdown.index, drawdown.values, 0,
                    color="#e74c3c", alpha=0.4)
    ax.plot(drawdown.index, drawdown.values, color="#e74c3c", linewidth=0.8)
    ax.set_title("Strategy Drawdown", fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = save_path or REPORT_DIR / "drawdown.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved drawdown plot to %s", save_path)
    return save_path


def plot_pr_curve(
    pr_df: pd.DataFrame,
    title: str = "Precision-Recall Curve & F1 Score across Thresholds",
    save_path: Optional[Path] = None,
) -> Path:
    """
    Plot Precision, Recall, and F1 score against decision thresholds.

    Parameters
    ----------
    pr_df : pd.DataFrame
        DataFrame with columns [threshold, precision, recall, f1_score]
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.plot(pr_df["threshold"], pr_df["precision"], label="Precision", color="#3498db", linewidth=2)
    ax1.plot(pr_df["threshold"], pr_df["recall"], label="Recall", color="#e74c3c", linewidth=2)
    ax1.plot(pr_df["threshold"], pr_df["f1_score"], label="F1 Score", color="#2ecc71", linewidth=2, linestyle="--")

    # Find max F1 score
    max_f1_idx = pr_df["f1_score"].idxmax()
    best_thresh = pr_df.loc[max_f1_idx, "threshold"]
    best_f1 = pr_df.loc[max_f1_idx, "f1_score"]
    
    ax1.axvline(best_thresh, color="gray", linestyle=":", alpha=0.8)
    ax1.annotate(f"Optimal Threshold = {best_thresh:.3f}\nMax F1 = {best_f1:.3f}",
                 xy=(best_thresh, best_f1),
                 xytext=(best_thresh + 0.05, best_f1 + 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

    ax1.set_xlabel("Decision Threshold")
    ax1.set_ylabel("Score")
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower center")

    save_path = save_path or REPORT_DIR / "pr_curve.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved PR curve to %s", save_path)
    return save_path
