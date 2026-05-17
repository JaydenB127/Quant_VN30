# -*- coding: utf-8 -*-
"""
Data processor: clean, align, flag circuit breakers, handle missing data.

Transforms raw OHLCV into a clean MultiIndex DataFrame ready for
feature engineering.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..config import CFG
from .schema import PROCESSED

logger = logging.getLogger(__name__)


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw OHLCV data.

    Steps
    -----
    1. Remove exact duplicate rows
    2. Drop rows with NaN in OHLCV
    3. Drop rows with non-positive prices or volume
    4. Ensure proper dtypes

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV with columns [date, ticker, open, high, low, close, volume].

    Returns
    -------
    pd.DataFrame
        Cleaned OHLCV.
    """
    n_before = len(df)
    df = df.copy()

    # 1. Remove duplicates
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")

    # 2. Drop NaN in price/volume
    price_cols = ["open", "high", "low", "close", "volume"]
    df = df.dropna(subset=price_cols)

    # 3. Drop non-positive prices
    for col in ["open", "high", "low", "close"]:
        df = df[df[col] > 0]
    df = df[df["volume"] >= 0]

    # 4. Dtypes
    df["date"] = pd.to_datetime(df["date"])
    for col in price_cols:
        df[col] = df[col].astype(np.float64)
    df["ticker"] = df["ticker"].astype(str)

    n_after = len(df)
    if n_before != n_after:
        logger.info(
            "Cleaned OHLCV: %d → %d rows (dropped %d)",
            n_before, n_after, n_before - n_after,
        )

    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple and log returns per ticker (using close-to-close).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned OHLCV.

    Returns
    -------
    pd.DataFrame
        With columns [returns, log_returns] added.
    """
    df = df.copy()
    df = df.sort_values(["ticker", "date"])

    df[PROCESSED.returns] = df.groupby("ticker")["close"].pct_change()
    df[PROCESSED.log_returns] = np.log1p(df[PROCESSED.returns])

    return df


def add_reference_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add reference price = previous day's close (for circuit breaker calc).

    In Vietnam, the reference price is the previous session's closing price.
    """
    df = df.copy()
    df = df.sort_values(["ticker", "date"])
    df[PROCESSED.ref_price] = df.groupby("ticker")["close"].shift(1)
    return df


def flag_circuit_breakers(
    df: pd.DataFrame,
    limit_pct: float = 0.07,
) -> pd.DataFrame:
    """
    Flag days where stock hit circuit breaker limits (±7% in HOSE).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain [close, ref_price].
    limit_pct : float
        Circuit breaker percentage (default 0.07 for HOSE).

    Returns
    -------
    pd.DataFrame
        With columns [is_limit_up, is_limit_down] added.
    """
    df = df.copy()

    if PROCESSED.ref_price not in df.columns:
        df = add_reference_price(df)

    # Tolerance for float comparison
    tol = 0.002  # 0.2% tolerance for rounding in price ticks

    daily_return = (df["close"] - df[PROCESSED.ref_price]) / df[PROCESSED.ref_price]

    df[PROCESSED.is_limit_up] = (daily_return >= limit_pct - tol).astype(np.int8)
    df[PROCESSED.is_limit_down] = (daily_return <= -limit_pct + tol).astype(np.int8)

    n_up = df[PROCESSED.is_limit_up].sum()
    n_down = df[PROCESSED.is_limit_down].sum()
    logger.info(
        "Circuit breaker flags: %d limit-up days, %d limit-down days",
        n_up, n_down,
    )

    return df


def flag_suspended(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag suspended trading days (zero volume or identical OHLC).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned OHLCV.

    Returns
    -------
    pd.DataFrame
        With column [is_suspended] added.
    """
    df = df.copy()

    zero_vol = df["volume"] == 0
    flat_price = (
        (df["open"] == df["high"])
        & (df["high"] == df["low"])
        & (df["low"] == df["close"])
        & (df["volume"] == 0)
    )

    df[PROCESSED.is_suspended] = (zero_vol | flat_price).astype(np.int8)

    n_suspended = df[PROCESSED.is_suspended].sum()
    if n_suspended > 0:
        logger.info("Flagged %d suspended stock-days", n_suspended)

    return df


def build_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert flat DataFrame to MultiIndex (date, ticker).

    Parameters
    ----------
    df : pd.DataFrame
        Processed OHLCV with [date, ticker, ...].

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame sorted by (date, ticker).
    """
    df = df.copy()
    df = df.set_index(["date", "ticker"]).sort_index()
    return df


def process_stock_data(
    raw_df: pd.DataFrame,
    limit_pct: Optional[float] = None,
) -> pd.DataFrame:
    """
    Full processing pipeline: clean → returns → circuit breakers → suspend flags.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw OHLCV data from downloader.
    limit_pct : float, optional
        Circuit breaker limit. Defaults to config value.

    Returns
    -------
    pd.DataFrame
        Processed MultiIndex (date, ticker) DataFrame.
    """
    limit_pct = limit_pct or CFG.features.circuit_breaker_pct

    logger.info("Processing stock data: %d raw rows", len(raw_df))

    df = clean_ohlcv(raw_df)
    df = add_returns(df)
    df = add_reference_price(df)
    df = flag_circuit_breakers(df, limit_pct=limit_pct)
    df = flag_suspended(df)
    df = build_multiindex(df)

    logger.info(
        "Processing complete: %d rows, %d tickers, %d dates",
        len(df),
        df.index.get_level_values("ticker").nunique(),
        df.index.get_level_values("date").nunique(),
    )

    return df


def process_index_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process index (VN-Index) data: clean, add returns.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw index OHLCV.

    Returns
    -------
    pd.DataFrame
        Indexed by date, with returns columns.
    """
    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = df[col].astype(np.float64)

    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log1p(df["returns"])
    df = df.set_index("date").sort_index()

    logger.info("Processed index data: %d rows", len(df))
    return df
