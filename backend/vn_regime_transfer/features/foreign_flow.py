# -*- coding: utf-8 -*-
"""
Foreign flow proxy features.

Since direct foreign flow data may not be available via vnstock,
we use proxy indicators derived from volume patterns and index correlation.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def foreign_flow_proxy_volume(
    volume: pd.Series, close: pd.Series, period: int = 20, lag: int = 1
) -> pd.Series:
    """
    Proxy for foreign activity: unusual volume on large-cap moves.
    Large volume + positive return → possible foreign buying.
    """
    ret = close.groupby(level="ticker").pct_change()
    vol_z = volume.groupby(level="ticker").transform(
        lambda x: (x - x.rolling(period, min_periods=period).mean())
        / x.rolling(period, min_periods=period).std().replace(0, np.nan)
    )
    signed_flow = vol_z * np.sign(ret)
    flow_ma = signed_flow.groupby(level="ticker").transform(
        lambda x: x.rolling(period, min_periods=5).mean()
    )
    return flow_ma.groupby(level="ticker").shift(lag)


def index_correlation(
    stock_returns: pd.Series,
    index_returns: pd.Series,
    period: int = 20,
    lag: int = 1,
) -> pd.Series:
    """
    Rolling correlation between stock and index returns.
    Low correlation may indicate stock-specific (possibly foreign-driven) moves.
    """
    # Align index returns to stock's date index
    dates = stock_returns.index.get_level_values("date")
    idx_aligned = index_returns.reindex(dates).values

    corr = stock_returns.groupby(level="ticker").transform(
        lambda x: x.rolling(period, min_periods=10).corr(
            pd.Series(idx_aligned[:len(x)], index=x.index)
        )
    )
    return corr.groupby(level="ticker").shift(lag)


def compute_foreign_flow_features(
    df: pd.DataFrame,
    index_df: pd.DataFrame = None,
    lag: int = 1,
) -> pd.DataFrame:
    """Compute foreign flow proxy features."""
    features = {}

    features["ff_proxy_vol_20"] = foreign_flow_proxy_volume(
        df["volume"], df["close"], period=20, lag=lag)
    features["ff_proxy_vol_60"] = foreign_flow_proxy_volume(
        df["volume"], df["close"], period=60, lag=lag)

    if index_df is not None and "returns" in index_df.columns:
        features["index_corr_20"] = index_correlation(
            df["returns"], index_df["returns"], period=20, lag=lag)
        features["index_corr_60"] = index_correlation(
            df["returns"], index_df["returns"], period=60, lag=lag)

    feat_df = pd.DataFrame(features, index=df.index)
    logger.info("Computed %d foreign flow features", len(features))
    return pd.concat([df, feat_df], axis=1)
