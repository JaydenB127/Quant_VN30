# -*- coding: utf-8 -*-
"""
Walk-forward validation engine.

Expanding window: train [start → train_end], test [train_end+1 → test_end].
Slide by step_months, no data leakage across time.

Supports: Base LGB, Transfer LGB, Ensemble, Static LGB, XGBoost, LSTM, Transformer.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from ..config import CFG
from ..data.schema import TARGET
from ..features.builder import get_feature_columns
from ..model.base_lgb import BaseLGBModel
from ..model.transfer_lgb import TransferLGBModel
from ..model.ensemble import EnsembleModel
from ..model.baselines import StaticLGBBaseline, XGBoostBaseline, RandomBaseline
from ..regime.hybrid import get_regime_for_dates
from .metrics import classification_metrics

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    # Ground truth
    y_true: np.ndarray = field(repr=False, default=None)
    # --- LGB family predictions ---
    y_proba_base: np.ndarray = field(repr=False, default=None)
    y_proba_transfer: np.ndarray = field(repr=False, default=None)
    y_proba_ensemble: np.ndarray = field(repr=False, default=None)
    y_proba_static: np.ndarray = field(repr=False, default=None)
    y_proba_xgb: np.ndarray = field(repr=False, default=None)
    # --- DL predictions ---
    y_proba_lstm: np.ndarray = field(repr=False, default=None)
    y_proba_transformer: np.ndarray = field(repr=False, default=None)
    # --- Binary versions (derived) ---
    y_pred_base: np.ndarray = field(repr=False, default=None)
    y_pred_transfer: np.ndarray = field(repr=False, default=None)
    y_pred_ensemble: np.ndarray = field(repr=False, default=None)
    y_pred_static: np.ndarray = field(repr=False, default=None)
    y_pred_xgb: np.ndarray = field(repr=False, default=None)
    y_pred_lstm: np.ndarray = field(repr=False, default=None)
    y_pred_transformer: np.ndarray = field(repr=False, default=None)
    # --- Metrics dicts ---
    metrics_base: dict = field(default_factory=dict)
    metrics_transfer: dict = field(default_factory=dict)
    metrics_ensemble: dict = field(default_factory=dict)
    metrics_static: dict = field(default_factory=dict)
    metrics_xgb: dict = field(default_factory=dict)
    metrics_lstm: dict = field(default_factory=dict)
    metrics_transformer: dict = field(default_factory=dict)
    # --- Meta ---
    test_dates: pd.Index = field(repr=False, default=None)
    test_regimes: np.ndarray = field(repr=False, default=None)


# ═══════════════════════════════════════════════════════════════════════
#  Fold date generation
# ═══════════════════════════════════════════════════════════════════════

def generate_fold_dates(
    data_start: str,
    data_end: str,
    initial_train_months: int = 60,
    test_months: int = 6,
    step_months: int = 3,
) -> List[Tuple[str, str, str, str]]:
    """Generate walk-forward fold date ranges (expanding window)."""
    start = pd.Timestamp(data_start)
    end = pd.Timestamp(data_end)

    train_end = start + relativedelta(months=initial_train_months)
    folds = []

    while True:
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + relativedelta(months=test_months) - pd.Timedelta(days=1)

        if test_end > end:
            test_end = end
        if test_start >= end:
            break

        folds.append((
            start.strftime("%Y-%m-%d"),
            train_end.strftime("%Y-%m-%d"),
            test_start.strftime("%Y-%m-%d"),
            test_end.strftime("%Y-%m-%d"),
        ))
        train_end = train_end + relativedelta(months=step_months)

    logger.info("Generated %d walk-forward folds", len(folds))
    return folds


# ═══════════════════════════════════════════════════════════════════════
#  Helper: safe classification metrics
# ═══════════════════════════════════════════════════════════════════════

def _safe_metrics(y_true, y_pred, y_proba):
    """Compute classification metrics; return empty dict on failure."""
    try:
        return classification_metrics(y_true, y_pred, y_proba)
    except Exception as exc:
        logger.warning("Metrics computation failed: %s", exc)
        return {}


# ═══════════════════════════════════════════════════════════════════════
#  Main walk-forward engine
# ═══════════════════════════════════════════════════════════════════════

def run_walk_forward(
    feature_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    initial_train_months: Optional[int] = None,
    test_months: Optional[int] = None,
    step_months: Optional[int] = None,
    run_dl: Optional[bool] = None,
) -> List[FoldResult]:
    """
    Execute full walk-forward validation with ALL model variants.

    Models trained per fold
    -----------------------
    1. Base LightGBM (pre-trained)
    2. Transfer LightGBM (fine-tuned from base)
    3. Ensemble (weighted base + transfer)
    4. Static LightGBM (retrained from scratch — no transfer)
    5. XGBoost (retrained from scratch)
    6. LSTM (optional, if PyTorch available)
    7. Transformer (optional, if PyTorch available)

    Parameters
    ----------
    feature_df : pd.DataFrame
        Complete feature matrix with MultiIndex (date, ticker).
    regime_df : pd.DataFrame
        Regime detection output indexed by date.
    run_dl : bool, optional
        Override whether to run DL baselines. Defaults to config.
    """
    cfg_v = CFG.validation
    cfg_dl = CFG.deep_learning
    initial_train_months = initial_train_months or cfg_v.initial_train_months
    test_months = test_months or cfg_v.test_months
    step_months = step_months or cfg_v.step_months
    run_dl = run_dl if run_dl is not None else cfg_dl.enabled

    # Check PyTorch availability
    has_torch = False
    if run_dl:
        try:
            import torch
            has_torch = True
            logger.info("PyTorch %s found — DL baselines enabled (device: %s)",
                        torch.__version__,
                        "CUDA" if torch.cuda.is_available() else "CPU")
        except ImportError:
            logger.warning("PyTorch not installed — DL baselines disabled")
            has_torch = False

    feature_cols = get_feature_columns(feature_df)
    dates = feature_df.index.get_level_values("date")
    data_start = dates.min().strftime("%Y-%m-%d")
    data_end = dates.max().strftime("%Y-%m-%d")

    folds = generate_fold_dates(
        data_start, data_end,
        initial_train_months, test_months, step_months,
    )

    results: List[FoldResult] = []

    for fold_id, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        logger.info(
            "═══ Fold %d/%d: train [%s → %s], test [%s → %s] ═══",
            fold_id + 1, len(folds), tr_start, tr_end, te_start, te_end,
        )

        # ── Split data ────────────────────────────────────────────────
        train_mask = (dates >= tr_start) & (dates <= tr_end)
        test_mask = (dates >= te_start) & (dates <= te_end)

        train_df = feature_df.loc[train_mask]
        test_df = feature_df.loc[test_mask]

        if len(train_df) < 100 or len(test_df) < 20:
            logger.warning("Fold %d: insufficient data (%d train, %d test) — skipping",
                           fold_id, len(train_df), len(test_df))
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[TARGET.label]
        X_test = test_df[feature_cols]
        y_test = test_df[TARGET.label]

        # Train / valid split (last 20% of training for early stopping)
        n_valid = max(int(len(X_train) * 0.2), 50)
        X_tr, y_tr = X_train.iloc[:-n_valid], y_train.iloc[:-n_valid]
        X_va, y_va = X_train.iloc[-n_valid:], y_train.iloc[-n_valid:]

        # ── Regime info ───────────────────────────────────────────────
        test_dates_unique = test_df.index.get_level_values("date").unique()
        test_regimes = get_regime_for_dates(regime_df, test_dates_unique)
        current_regime = int(test_regimes.mode().iloc[0]) if len(test_regimes) > 0 else 1

        train_dates = train_df.index.get_level_values("date")
        train_regime_labels = get_regime_for_dates(regime_df, train_dates)

        y_true = y_test.values

        # ── 1. Base LightGBM ─────────────────────────────────────────
        base_model = BaseLGBModel()
        base_model.fit(X_tr, y_tr, X_va, y_va)
        y_proba_base = base_model.predict(X_test)

        # ── 2. Transfer LightGBM ─────────────────────────────────────
        transfer_model = TransferLGBModel(base_model)
        transfer_model.finetune(
            X_train, y_train, X_va, y_va,
            regime_mask=train_regime_labels,
            current_regime=current_regime,
        )
        y_proba_transfer = transfer_model.predict(X_test)

        # ── 3. Ensemble ──────────────────────────────────────────────
        ensemble_model = EnsembleModel(base_model, transfer_model)
        regime_conf = regime_df.loc[
            regime_df.index.isin(test_dates_unique), "hmm_prob"
        ].mean()
        regime_conf = regime_conf if not np.isnan(regime_conf) else 0.5
        y_proba_ensemble = ensemble_model.predict(X_test, regime_confidence=regime_conf)

        # ── 4. Static LightGBM ───────────────────────────────────────
        static_model = StaticLGBBaseline()
        static_model.fit(X_tr, y_tr, X_va, y_va)
        y_proba_static = static_model.predict(X_test)

        # ── 5. XGBoost ───────────────────────────────────────────────
        xgb_model = XGBoostBaseline()
        xgb_model.fit(X_tr, y_tr, X_va, y_va)
        y_proba_xgb = xgb_model.predict(X_test)

        # ── 6. LSTM (optional) ───────────────────────────────────────
        y_proba_lstm = np.full(len(y_true), 0.5)
        if has_torch:
            try:
                from ..model.deep_baselines import LSTMBaseline
                lstm = LSTMBaseline(
                    hidden_size=cfg_dl.lstm_hidden,
                    num_layers=cfg_dl.lstm_layers,
                    dropout=cfg_dl.lstm_dropout,
                    seq_len=cfg_dl.seq_len,
                    lr=cfg_dl.lstm_lr,
                    epochs=cfg_dl.lstm_epochs,
                    batch_size=cfg_dl.lstm_batch,
                    patience=cfg_dl.lstm_patience,
                )
                lstm.fit(X_train, y_train)
                y_proba_lstm = lstm.predict(X_test)
                logger.info("Fold %d — LSTM trained successfully", fold_id)
            except Exception as exc:
                logger.warning("Fold %d — LSTM failed: %s", fold_id, exc)

        # ── 7. Transformer (optional) ────────────────────────────────
        y_proba_transformer = np.full(len(y_true), 0.5)
        if has_torch:
            try:
                from ..model.deep_baselines import TransformerBaseline
                tf = TransformerBaseline(
                    d_model=cfg_dl.tf_d_model,
                    n_heads=cfg_dl.tf_n_heads,
                    n_layers=cfg_dl.tf_n_layers,
                    dim_ff=cfg_dl.tf_dim_ff,
                    dropout=cfg_dl.tf_dropout,
                    seq_len=cfg_dl.seq_len,
                    lr=cfg_dl.tf_lr,
                    epochs=cfg_dl.tf_epochs,
                    batch_size=cfg_dl.tf_batch,
                    patience=cfg_dl.tf_patience,
                )
                tf.fit(X_train, y_train)
                y_proba_transformer = tf.predict(X_test)
                logger.info("Fold %d — Transformer trained successfully", fold_id)
            except Exception as exc:
                logger.warning("Fold %d — Transformer failed: %s", fold_id, exc)

        # ── Build FoldResult ─────────────────────────────────────────
        _b = lambda p: (p >= 0.5).astype(int)

        fold_result = FoldResult(
            fold_id=fold_id,
            train_start=tr_start, train_end=tr_end,
            test_start=te_start, test_end=te_end,
            n_train=len(X_train), n_test=len(X_test),
            y_true=y_true,
            y_proba_base=y_proba_base, y_pred_base=_b(y_proba_base),
            y_proba_transfer=y_proba_transfer, y_pred_transfer=_b(y_proba_transfer),
            y_proba_ensemble=y_proba_ensemble, y_pred_ensemble=_b(y_proba_ensemble),
            y_proba_static=y_proba_static, y_pred_static=_b(y_proba_static),
            y_proba_xgb=y_proba_xgb, y_pred_xgb=_b(y_proba_xgb),
            y_proba_lstm=y_proba_lstm, y_pred_lstm=_b(y_proba_lstm),
            y_proba_transformer=y_proba_transformer, y_pred_transformer=_b(y_proba_transformer),
            metrics_base=_safe_metrics(y_true, _b(y_proba_base), y_proba_base),
            metrics_transfer=_safe_metrics(y_true, _b(y_proba_transfer), y_proba_transfer),
            metrics_ensemble=_safe_metrics(y_true, _b(y_proba_ensemble), y_proba_ensemble),
            metrics_static=_safe_metrics(y_true, _b(y_proba_static), y_proba_static),
            metrics_xgb=_safe_metrics(y_true, _b(y_proba_xgb), y_proba_xgb),
            metrics_lstm=_safe_metrics(y_true, _b(y_proba_lstm), y_proba_lstm),
            metrics_transformer=_safe_metrics(y_true, _b(y_proba_transformer), y_proba_transformer),
            test_dates=test_df.index,
            test_regimes=get_regime_for_dates(
                regime_df, test_df.index.get_level_values("date")
            ).values,
        )

        logger.info(
            "Fold %d — AUC: ensemble=%.4f | base=%.4f | static=%.4f | "
            "xgb=%.4f | lstm=%.4f | transformer=%.4f",
            fold_id,
            fold_result.metrics_ensemble.get("auc_roc", np.nan),
            fold_result.metrics_base.get("auc_roc", np.nan),
            fold_result.metrics_static.get("auc_roc", np.nan),
            fold_result.metrics_xgb.get("auc_roc", np.nan),
            fold_result.metrics_lstm.get("auc_roc", np.nan),
            fold_result.metrics_transformer.get("auc_roc", np.nan),
        )

        results.append(fold_result)

    logger.info("Walk-forward complete: %d folds executed", len(results))
    return results


# ═══════════════════════════════════════════════════════════════════════
#  Aggregation
# ═══════════════════════════════════════════════════════════════════════

ALL_MODELS = [
    "base", "transfer", "ensemble", "static",
    "xgb", "lstm", "transformer",
]


def aggregate_fold_results(results: List[FoldResult]) -> pd.DataFrame:
    """Aggregate metrics across all folds for every model."""
    rows = []
    for r in results:
        for model_name in ALL_MODELS:
            metrics = getattr(r, f"metrics_{model_name}", {})
            if not metrics:
                continue
            row = {
                "fold": r.fold_id,
                "model": model_name,
                "test_start": r.test_start,
                "test_end": r.test_end,
                "n_test": r.n_test,
            }
            row.update(metrics)
            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        logger.warning("No metrics to aggregate")
        return df

    # Summary table
    metric_cols = [c for c in df.columns if c not in
                   {"fold", "model", "test_start", "test_end", "n_test"}]
    summary = df.groupby("model")[metric_cols].agg(["mean", "std"])
    logger.info("Aggregated results:\n%s", summary.to_string())
    return df
