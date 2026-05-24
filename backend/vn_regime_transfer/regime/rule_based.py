# -*- coding: utf-8 -*-
"""
Rule-based regime filters using EMA, ATR percentile, and volume.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from ..config import CFG

logger = logging.getLogger(__name__)


def ema_trend_regime(
    close: pd.Series,
    period: int = 200,
) -> pd.Series:
    """
    Rule: close > EMA(period) → bull (0), else bear (2).

    Parameters
    ----------
    close : pd.Series
        Index close prices (date-indexed).
    period : int
        EMA period.

    Returns
    -------
    pd.Series
        0 = bull, 2 = bear.
    """
    ema_val = close.ewm(span=period, adjust=False).mean()
    regime = pd.Series(np.where(close > ema_val, 0, 2), index=close.index)
    return regime


def volatility_regime(
    returns: pd.Series,
    period: int = 60,
    percentile: float = 80.0,
) -> pd.Series:
    """
    Rule: if rolling vol > historical percentile → high vol (1), else low vol (0).

    Parameters
    ----------
    returns : pd.Series
        Index returns (date-indexed).
    period : int
        Rolling window for vol computation.
    percentile : float
        Threshold percentile.

    Returns
    -------
    pd.Series
        1 = high vol, 0 = low vol.
    """
    vol = returns.rolling(period, min_periods=20).std()
    expanding_pct = vol.expanding(min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    high_vol = (expanding_pct > percentile / 100.0).astype(int)
    return high_vol


def rule_based_regime(index_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combined rule-based regime classification.

    Logic:
      - If close < EMA200 AND high vol → bear (2)
      - If close > EMA200 AND low vol → bull (0)
      - Otherwise → sideways (1)

    Parameters
    ----------
    index_df : pd.DataFrame
        Indexed by date, columns [close, returns].

    Returns
    -------
    pd.DataFrame
        Column: [rule_regime] ∈ {0: bull, 1: sideways, 2: bear}.
    """
    trend = ema_trend_regime(index_df["close"], period=CFG.regime.ema_trend_period)
    vol_flag = volatility_regime(
        index_df["returns"],
        period=60,
        percentile=CFG.regime.atr_percentile_threshold,
    )

    # Combine: bear = below EMA200 + high vol
    regime = pd.Series(1, index=index_df.index, dtype=int)  # default sideways
    regime[(trend == 0) & (vol_flag == 0)] = 0  # bull: above EMA200 + low vol
    regime[(trend == 2) & (vol_flag == 1)] = 2  # bear: below EMA200 + high vol

    result = pd.DataFrame({"rule_regime": regime}, index=index_df.index)
    logger.info(
        "Rule-based regime: %s",
        result["rule_regime"].value_counts().to_dict(),
    )
    return result
