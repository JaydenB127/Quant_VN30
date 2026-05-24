# -*- coding: utf-8 -*-
"""
Vietnam market microstructure features.

Captures HOSE/HNX characteristics: circuit-breaker proximity,
limit-hit frequency, settlement window, tick-size effects.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from ..config import CFG

logger = logging.getLogger(__name__)


def circuit_breaker_proximity(close, ref_price, limit_pct=0.07, lag=1):
    """Normalized distance within daily price band, lagged."""
    ceiling = ref_price * (1 + limit_pct)
    floor = ref_price * (1 - limit_pct)
    band = ceiling - floor
    proximity = (close - floor) / band.replace(0, np.nan)
    return proximity.groupby(level="ticker").shift(lag)


def limit_hit_count(is_limit_up, is_limit_down, window=5, lag=1):
    """Rolling count of limit-hit days in last window days, lagged."""
    total_hits = is_limit_up + is_limit_down
    count = total_hits.groupby(level="ticker").transform(
        lambda x: x.rolling(window, min_periods=1).sum()
    )
    return count.groupby(level="ticker").shift(lag)


def limit_hit_direction_ratio(is_limit_up, is_limit_down, window=10, lag=1):
    """Ratio of limit-up to total limit hits in rolling window, lagged."""
    up_count = is_limit_up.groupby(level="ticker").transform(
        lambda x: x.rolling(window, min_periods=1).sum()
    )
    total = (is_limit_up + is_limit_down).groupby(level="ticker").transform(
        lambda x: x.rolling(window, min_periods=1).sum()
    )
    return (up_count / total.replace(0, np.nan)).groupby(level="ticker").shift(lag)


def intraday_range_ratio(high, low, ref_price, lag=1):
    """Intraday range as fraction of allowed band (14% for HOSE), lagged."""
    actual_range = high - low
    max_range = ref_price * 0.14
    return (actual_range / max_range.replace(0, np.nan)).groupby(level="ticker").shift(lag)


def close_position_in_range(open_, high, low, close, lag=1):
    """Where close sits within day's range: (close-low)/(high-low), lagged."""
    denom = (high - low).replace(0, np.nan)
    return ((close - low) / denom).groupby(level="ticker").shift(lag)


def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all Vietnam microstructure features."""
    cfg = CFG.features
    lag = cfg.min_lag
    features = {}
    features["cb_proximity"] = circuit_breaker_proximity(
        df["close"], df["ref_price"], limit_pct=cfg.circuit_breaker_pct, lag=lag)
    features["limit_hit_5d"] = limit_hit_count(
        df["is_limit_up"], df["is_limit_down"], window=cfg.limit_hit_window, lag=lag)
    features["limit_hit_10d"] = limit_hit_count(
        df["is_limit_up"], df["is_limit_down"], window=10, lag=lag)
    features["limit_dir_ratio_10d"] = limit_hit_direction_ratio(
        df["is_limit_up"], df["is_limit_down"], window=10, lag=lag)
    features["intraday_range_ratio"] = intraday_range_ratio(
        df["high"], df["low"], df["ref_price"], lag=lag)
    features["close_in_range"] = close_position_in_range(
        df["open"], df["high"], df["low"], df["close"], lag=lag)

    feat_df = pd.DataFrame(features, index=df.index)
    logger.info("Computed %d microstructure features", len(features))
    return pd.concat([df, feat_df], axis=1)
