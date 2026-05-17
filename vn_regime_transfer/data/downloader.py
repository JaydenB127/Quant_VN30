# -*- coding: utf-8 -*-
"""
Data downloader using vnstock API.

Downloads OHLCV data for VN30 stocks and VN-Index benchmark,
caches as parquet for fast reload.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ..config import CFG, DATA_DIR, CACHE_DIR
from .universe import get_all_historical_tickers

logger = logging.getLogger(__name__)


def _safe_import_vnstock():
    """Import vnstock with helpful error message."""
    try:
        from vnstock import Vnstock
        return Vnstock
    except ImportError:
        raise ImportError(
            "vnstock is required. Install with: pip install vnstock"
        )


def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    source: str = "VCI",
    retry: int = 3,
    delay: float = 1.0,
) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker (e.g. "FPT", "VNM").
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.
    source : str
        Data source for vnstock (default "VCI").
    retry : int
        Number of retry attempts on failure.
    delay : float
        Seconds to wait between retries.

    Returns
    -------
    pd.DataFrame
        Columns: [date, open, high, low, close, volume, ticker]
    """
    Vnstock = _safe_import_vnstock()

    for attempt in range(1, retry + 1):
        try:
            stock = Vnstock().stock(symbol=ticker, source=source)
            df = stock.quote.history(
                start=start_date,
                end=end_date,
                interval="1D",
            )

            if df is None or df.empty:
                logger.warning("No data returned for %s", ticker)
                return pd.DataFrame()

            # Standardise column names
            col_map = {}
            for col in df.columns:
                cl = col.lower().strip()
                if cl in ("time", "date", "trading_date", "tradingdate"):
                    col_map[col] = "date"
                elif cl in ("open",):
                    col_map[col] = "open"
                elif cl in ("high",):
                    col_map[col] = "high"
                elif cl in ("low",):
                    col_map[col] = "low"
                elif cl in ("close",):
                    col_map[col] = "close"
                elif cl in ("volume",):
                    col_map[col] = "volume"

            df = df.rename(columns=col_map)

            required = {"date", "open", "high", "low", "close", "volume"}
            if not required.issubset(df.columns):
                missing = required - set(df.columns)
                logger.warning(
                    "Missing columns %s for %s. Available: %s",
                    missing, ticker, list(df.columns),
                )
                return pd.DataFrame()

            df["ticker"] = ticker
            df["date"] = pd.to_datetime(df["date"])
            df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
            df = df.sort_values("date").reset_index(drop=True)

            logger.info(
                "Downloaded %s: %d rows [%s → %s]",
                ticker, len(df),
                df["date"].min().strftime("%Y-%m-%d"),
                df["date"].max().strftime("%Y-%m-%d"),
            )
            return df

        except Exception as exc:
            logger.warning(
                "Attempt %d/%d failed for %s: %s",
                attempt, retry, ticker, exc,
            )
            if attempt < retry:
                time.sleep(delay * attempt)

    logger.error("All %d attempts failed for %s", retry, ticker)
    return pd.DataFrame()


def download_index_data(
    symbol: str = "VNINDEX",
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
    source: str = "VCI",
) -> pd.DataFrame:
    """
    Download benchmark index (VN-Index) data.

    Returns
    -------
    pd.DataFrame
        Columns: [date, open, high, low, close, volume]
    """
    Vnstock = _safe_import_vnstock()

    try:
        stock = Vnstock().stock(symbol=symbol, source=source)
        df = stock.quote.history(start=start_date, end=end_date, interval="1D")

        if df is None or df.empty:
            logger.warning("No index data returned for %s", symbol)
            return pd.DataFrame()

        # Standardise columns
        col_map = {}
        for col in df.columns:
            cl = col.lower().strip()
            if cl in ("time", "date", "trading_date", "tradingdate"):
                col_map[col] = "date"
            elif cl in ("open",):
                col_map[col] = "open"
            elif cl in ("high",):
                col_map[col] = "high"
            elif cl in ("low",):
                col_map[col] = "low"
            elif cl in ("close",):
                col_map[col] = "close"
            elif cl in ("volume",):
                col_map[col] = "volume"

        df = df.rename(columns=col_map)
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("date").reset_index(drop=True)

        logger.info(
            "Downloaded %s: %d rows [%s → %s]",
            symbol, len(df),
            df["date"].min().strftime("%Y-%m-%d"),
            df["date"].max().strftime("%Y-%m-%d"),
        )
        return df

    except Exception as exc:
        logger.error("Failed to download index %s: %s", symbol, exc)
        return pd.DataFrame()


def download_universe(
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    source: Optional[str] = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV data for the entire universe and cache as parquet.

    Parameters
    ----------
    tickers : list of str, optional
        Stock tickers. Defaults to VN30 from config.
    start_date : str, optional
        Defaults to config start_date.
    end_date : str, optional
        Defaults to config end_date.
    source : str, optional
        Defaults to config source.
    cache : bool
        If True, save/load from parquet cache.

    Returns
    -------
    pd.DataFrame
        Concatenated OHLCV for all tickers + index data.
    """
    tickers = tickers or get_all_historical_tickers()
    start_date = start_date or CFG.data.start_date
    end_date = end_date or CFG.data.end_date
    source = source or CFG.data.source

    cache_path = CACHE_DIR / f"ohlcv_{CFG.universe.name}_{start_date}_{end_date}.parquet"

    if cache and cache_path.exists():
        logger.info("Loading cached data from %s", cache_path)
        return pd.read_parquet(cache_path)

    frames: List[pd.DataFrame] = []

    for i, ticker in enumerate(tickers, 1):
        logger.info("Downloading %d/%d: %s", i, len(tickers), ticker)
        df = download_stock_data(
            ticker, start_date, end_date, source=source
        )
        if not df.empty:
            frames.append(df)
        # Be nice to the API
        time.sleep(0.5)

    if not frames:
        raise RuntimeError("No data downloaded for any ticker!")

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["date", "ticker"]).reset_index(drop=True)

    if cache:
        result.to_parquet(cache_path, index=False)
        logger.info("Cached %d rows to %s", len(result), cache_path)

    return result


def download_all(cache: bool = True) -> dict:
    """
    Download both universe stocks and index data.

    Returns
    -------
    dict
        {"stocks": DataFrame, "index": DataFrame}
    """
    stocks_df = download_universe(cache=cache)

    index_cache = CACHE_DIR / f"index_{CFG.universe.benchmark_ticker}.parquet"
    if cache and index_cache.exists():
        index_df = pd.read_parquet(index_cache)
    else:
        index_df = download_index_data(
            symbol=CFG.universe.benchmark_ticker,
            start_date=CFG.data.start_date,
            end_date=CFG.data.end_date,
            source=CFG.data.source,
        )
        if cache and not index_df.empty:
            index_df.to_parquet(index_cache, index=False)

    return {"stocks": stocks_df, "index": index_df}
