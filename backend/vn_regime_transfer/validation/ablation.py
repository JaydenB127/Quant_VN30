# -*- coding: utf-8 -*-
"""
Ablation study framework.

Evaluates the marginal contribution of each component:
- Transfer Learning (finetuning vs from scratch)
- Regime awareness (conditional finetuning)
- Time-decay (weighting recent samples)
- Ensemble (base + adapted)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import CFG
from ..data.schema import TARGET
from ..features.builder import get_feature_columns
from ..model.base_lgb import BaseLGBModel
from ..model.transfer_lgb import TransferLGBModel
from ..model.ensemble import EnsembleModel
from ..regime.hybrid import get_regime_for_dates
from .walk_forward import generate_fold_dates
from .metrics import classification_metrics
from .statistical_tests import diebold_mariano_test

logger = logging.getLogger(__name__)


def run_ablation_study(
    feature_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    initial_train_months: Optional[int] = None,
    test_months: Optional[int] = None,
    step_months: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run ablation study across walk-forward folds.

    Variants tested:
    1. Base Only
    2. + Transfer (no regime, no time-decay)
    3. + Regime (transfer + regime, no time-decay)
    4. + Time-Decay (transfer + decay, no regime)
    5. Full System (transfer + regime + decay + ensemble)

    Returns
    -------
    pd.DataFrame
        Comparative table with metrics and DM test vs Base.
    """
    cfg_v = CFG.validation
    initial_train_months = initial_train_months or cfg_v.initial_train_months
    test_months = test_months or cfg_v.test_months
    step_months = step_months or cfg_v.step_months

    feature_cols = get_feature_columns(feature_df)
    dates = feature_df.index.get_level_values("date")
    data_start = dates.min().strftime("%Y-%m-%d")
    data_end = dates.max().strftime("%Y-%m-%d")

    folds = generate_fold_dates(
        data_start, data_end,
        initial_train_months, test_months, step_months,
    )

    # Accumulators
    all_y_true = []
    variant_preds = {
        "1. Base Only": [],
        "2. + Transfer": [],
        "3. + Regime": [],
        "4. + Time-Decay": [],
        "5. Full System": [],
    }

    for fold_id, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        logger.info(
            "Ablation Fold %d/%d: [%s → %s]",
            fold_id + 1, len(folds), te_start, te_end,
        )

        train_mask = (dates >= tr_start) & (dates <= tr_end)
        test_mask = (dates >= te_start) & (dates <= te_end)

        train_df = feature_df.loc[train_mask]
        test_df = feature_df.loc[test_mask]

        if len(train_df) < 100 or len(test_df) < 20:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[TARGET.label]
        X_test = test_df[feature_cols]
        y_test = test_df[TARGET.label]

        n_valid = max(int(len(X_train) * 0.2), 50)
        X_tr, y_tr = X_train.iloc[:-n_valid], y_train.iloc[:-n_valid]
        X_va, y_va = X_train.iloc[-n_valid:], y_train.iloc[-n_valid:]

        test_dates_unique = test_df.index.get_level_values("date").unique()
        test_regimes = get_regime_for_dates(regime_df, test_dates_unique)
        current_regime = int(test_regimes.mode().iloc[0]) if len(test_regimes) > 0 else 1

        train_dates = train_df.index.get_level_values("date")
        train_regime_labels = get_regime_for_dates(regime_df, train_dates)

        all_y_true.extend(y_test.values)

        # 1. Base
        base_model = BaseLGBModel()
        base_model.fit(X_tr, y_tr, X_va, y_va)
        proba_base = base_model.predict(X_test)
        variant_preds["1. Base Only"].extend(proba_base)

        # 2. + Transfer (no regime, no time-decay)
        m2 = TransferLGBModel(base_model, time_decay_alpha=1.0)
        m2.finetune(X_train, y_train, X_va, y_va, regime_mask=None, current_regime=None)
        variant_preds["2. + Transfer"].extend(m2.predict(X_test))

        # 3. + Regime (transfer + regime, no time-decay)
        m3 = TransferLGBModel(base_model, time_decay_alpha=1.0)
        m3.finetune(X_train, y_train, X_va, y_va, regime_mask=train_regime_labels, current_regime=current_regime)
        variant_preds["3. + Regime"].extend(m3.predict(X_test))

        # 4. + Time-Decay (transfer + decay, no regime)
        m4 = TransferLGBModel(base_model, time_decay_alpha=CFG.model.time_decay_alpha)
        m4.finetune(X_train, y_train, X_va, y_va, regime_mask=None, current_regime=None)
        variant_preds["4. + Time-Decay"].extend(m4.predict(X_test))

        # 5. Full System (transfer + decay + regime + ensemble)
        m5 = TransferLGBModel(base_model, time_decay_alpha=CFG.model.time_decay_alpha)
        m5.finetune(X_train, y_train, X_va, y_va, regime_mask=train_regime_labels, current_regime=current_regime)
        ens = EnsembleModel(base_model, m5)
        
        regime_conf = regime_df.loc[
            regime_df.index.isin(test_dates_unique), "hmm_prob"
        ].mean()
        regime_conf = regime_conf if pd.notna(regime_conf) else 0.5
        variant_preds["5. Full System"].extend(ens.predict(X_test, regime_confidence=regime_conf))

    # Evaluate
    y_true_np = np.array(all_y_true)
    results = []
    
    base_preds = np.array(variant_preds["1. Base Only"])

    for variant_name, preds_list in variant_preds.items():
        preds_np = np.array(preds_list)
        preds_bin = (preds_np >= 0.5).astype(int)
        
        metrics = classification_metrics(y_true_np, preds_bin, preds_np)
        row = {"Variant": variant_name}
        row.update(metrics)
        
        # DM Test vs Base
        if variant_name == "1. Base Only":
            row["DM p-value (vs Base)"] = np.nan
        else:
            dm = diebold_mariano_test(y_true_np, preds_np, base_preds, loss="brier")
            row["DM p-value (vs Base)"] = dm["p_value"]
            
        results.append(row)

    df_results = pd.DataFrame(results)
    logger.info("Ablation Study Results:\n%s", df_results.to_string())
    return df_results
