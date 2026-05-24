# -*- coding: utf-8 -*-
"""
Threshold optimization and Precision-Recall analysis.

Finds the optimal decision threshold for binary classification based on F-beta score
or a minimum required precision.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

logger = logging.getLogger(__name__)


def optimize_threshold_fbeta(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    beta: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Find the threshold that maximizes the F-beta score.
    Beta < 1 heavily weights precision over recall.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_proba : np.ndarray
        Predicted probabilities.
    beta : float
        Beta parameter for F-score.

    Returns
    -------
    Tuple[float, Dict[str, float]]
        (best_threshold, {precision, recall, fbeta})
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Avoid division by zero
    numerator = (1 + beta**2) * (precisions * recalls)
    denominator = (beta**2 * precisions) + recalls
    
    # Handle zeros
    f_scores = np.divide(
        numerator, 
        denominator, 
        out=np.zeros_like(numerator), 
        where=denominator != 0
    )
    
    best_idx = np.argmax(f_scores)
    # precision_recall_curve returns thresholds array length N-1
    # where the last precision is 1.0 and recall is 0.0 with no threshold
    if best_idx < len(thresholds):
        best_threshold = thresholds[best_idx]
    else:
        best_threshold = 1.0
        
    metrics = {
        "precision": precisions[best_idx],
        "recall": recalls[best_idx],
        f"f{beta}_score": f_scores[best_idx],
    }
    
    logger.info("Optimal threshold (F%.1f): %.4f (Prec: %.4f, Rec: %.4f)",
                beta, best_threshold, metrics["precision"], metrics["recall"])
    return best_threshold, metrics


def optimize_threshold_min_precision(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_precision: float = 0.60,
) -> Tuple[float, Dict[str, float]]:
    """
    Find the lowest threshold that achieves a minimum required precision.
    Useful for risk-averse strategies that only trade high-confidence signals.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_proba : np.ndarray
        Predicted probabilities.
    min_precision : float
        Minimum required precision.

    Returns
    -------
    Tuple[float, Dict[str, float]]
        (best_threshold, {precision, recall})
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Find indices where precision meets the requirement
    valid_idx = np.where(precisions >= min_precision)[0]
    
    if len(valid_idx) == 0:
        logger.warning("Could not achieve minimum precision of %.2f", min_precision)
        best_idx = np.argmax(precisions)
        if best_idx < len(thresholds):
            best_threshold = thresholds[best_idx]
        else:
            best_threshold = 1.0
    else:
        # Among valid indices, find the one with the maximum recall (lowest threshold)
        best_valid_idx = valid_idx[np.argmax(recalls[valid_idx])]
        if best_valid_idx < len(thresholds):
            best_threshold = thresholds[best_valid_idx]
        else:
            best_threshold = 1.0
            
        best_idx = best_valid_idx

    metrics = {
        "precision": precisions[best_idx],
        "recall": recalls[best_idx],
    }
    
    logger.info("Threshold for min_precision=%.2f: %.4f (Prec: %.4f, Rec: %.4f)",
                min_precision, best_threshold, metrics["precision"], metrics["recall"])
    return best_threshold, metrics


def generate_pr_curve_data(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> pd.DataFrame:
    """
    Generate DataFrame with Precision-Recall curve points for plotting.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Match lengths (thresholds is 1 shorter than precisions/recalls)
    thresholds_padded = np.append(thresholds, 1.0)
    
    df = pd.DataFrame({
        "threshold": thresholds_padded,
        "precision": precisions,
        "recall": recalls,
    })
    
    # Add F1 score
    df["f1_score"] = 2 * (df["precision"] * df["recall"]) / (df["precision"] + df["recall"]).replace(0, np.nan)
    
    return df
