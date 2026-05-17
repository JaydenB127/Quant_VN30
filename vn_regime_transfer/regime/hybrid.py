# -*- coding: utf-8 -*-
"""
Hybrid regime detector: combines HMM + rules + changepoint.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from ..config import CFG
from .hmm_detector import fit_hmm_expanding
from .changepoint import detect_changepoints
from .rule_based import rule_based_regime

logger = logging.getLogger(__name__)


def detect_regime(index_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full hybrid regime detection pipeline.

    Combines:
      1. HMM states (data-driven)
      2. Rule-based overrides (expert knowledge)
      3. Changepoint detection (structural breaks trigger retrain)

    Priority: Rule-based override > HMM when rule confidence is high.

    Parameters
    ----------
    index_df : pd.DataFrame
        Processed index data with [close, returns].

    Returns
    -------
    pd.DataFrame
        Columns: [hmm_state, hmm_prob, cp_flag, days_since_cp,
                  rule_regime, regime].
        'regime' is the final label: {0: bull, 1: sideways, 2: bear}.
    """
    cfg = CFG.regime

    # 1. HMM
    logger.info("Running HMM regime detection...")
    hmm_df = fit_hmm_expanding(
        index_df,
        n_states=cfg.n_states,
        min_train_days=cfg.hmm_min_train_days,
        covariance_type=cfg.hmm_covariance_type,
        n_iter=cfg.hmm_n_iter,
        seed=CFG.seed,
        refit_interval=cfg.hmm_refit_interval,
    )

    # 2. Changepoint
    logger.info("Running changepoint detection...")
    cp_df = detect_changepoints(
        index_df,
        model=cfg.cp_model,
        min_size=cfg.cp_min_size,
        penalty=cfg.cp_penalty,
    )

    # 3. Rule-based
    logger.info("Running rule-based regime classification...")
    rule_df = rule_based_regime(index_df)

    # 4. Merge all
    regime_df = hmm_df.join(cp_df, how="outer").join(rule_df, how="outer")

    # 5. Hybrid logic: default to HMM, override when rules are confident
    regime_df["regime"] = regime_df["hmm_state"].copy()

    # Override: if rule says bear AND HMM prob < 0.7, trust the rule
    bear_override = (
        (regime_df["rule_regime"] == 2)
        & ((regime_df["hmm_prob"] < 0.7) | regime_df["hmm_state"].isna())
    )
    regime_df.loc[bear_override, "regime"] = 2

    # Override: if rule says bull AND HMM prob < 0.7, trust the rule
    bull_override = (
        (regime_df["rule_regime"] == 0)
        & ((regime_df["hmm_prob"] < 0.7) | regime_df["hmm_state"].isna())
    )
    regime_df.loc[bull_override, "regime"] = 0

    # Fill remaining NaN with sideways
    regime_df["regime"] = regime_df["regime"].fillna(1).astype(int)

    logger.info(
        "Hybrid regime distribution:\n%s",
        regime_df["regime"].value_counts().sort_index().to_string(),
    )

    return regime_df


def get_regime_for_dates(
    regime_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.Series:
    """Map regime labels to arbitrary dates via forward-fill."""
    regime_series = regime_df["regime"].reindex(dates).ffill().fillna(1).astype(int)
    return regime_series


def detect_regime_shift(
    regime_df: pd.DataFrame,
    lookback: int = 5,
) -> pd.Series:
    """
    Detect regime transitions.

    Returns True for dates where regime changed from the previous
    ``lookback`` days' dominant regime.
    """
    regime = regime_df["regime"]
    prev_mode = regime.rolling(lookback, min_periods=1).apply(
        lambda x: pd.Series(x).mode().iloc[0] if len(x) > 0 else np.nan,
        raw=False,
    )
    shift_flag = (regime != prev_mode).astype(int)
    return shift_flag
