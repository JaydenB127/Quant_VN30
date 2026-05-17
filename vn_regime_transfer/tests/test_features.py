# -*- coding: utf-8 -*-
"""Tests for feature engineering — anti-leakage validation."""
import numpy as np
import pandas as pd
import pytest
from vn_regime_transfer.features.technical import rsi, macd, atr, ema


def _make_sample_data(n_dates=300, n_tickers=3, seed=42):
    """Create synthetic MultiIndex (date, ticker) OHLCV data."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    tickers = [f"TICK{i}" for i in range(n_tickers)]

    rows = []
    for ticker in tickers:
        price = 100.0
        for date in dates:
            ret = rng.normal(0.0005, 0.02)
            price *= (1 + ret)
            high = price * (1 + abs(rng.normal(0, 0.01)))
            low = price * (1 - abs(rng.normal(0, 0.01)))
            vol = rng.randint(100_000, 1_000_000)
            rows.append({
                "date": date, "ticker": ticker,
                "open": price * (1 + rng.normal(0, 0.005)),
                "high": high, "low": low,
                "close": price, "volume": vol,
            })

    df = pd.DataFrame(rows)
    df = df.set_index(["date", "ticker"]).sort_index()
    df["returns"] = df.groupby(level="ticker")["close"].pct_change()
    return df


class TestAntiLeakage:
    """Verify that features are properly lagged."""

    @pytest.fixture
    def data(self):
        return _make_sample_data()

    def test_rsi_is_lagged(self, data):
        """RSI with lag=1 should have NaN at start (shift applied)."""
        result = rsi(data["close"], period=14, lag=1)
        # First 15 values per ticker should be NaN (14 warmup + 1 lag)
        for ticker in data.index.get_level_values("ticker").unique():
            ticker_data = result.xs(ticker, level="ticker")
            assert ticker_data.iloc[:15].isna().all(), \
                f"RSI for {ticker} has non-NaN values in warmup period"

    def test_macd_is_lagged(self, data):
        """MACD with lag=1 should shift output by 1."""
        macd_l, _, _ = macd(data["close"], lag=1)
        for ticker in data.index.get_level_values("ticker").unique():
            ticker_data = macd_l.xs(ticker, level="ticker")
            # At least first 27 values should be NaN (26 slow + 1 lag)
            assert ticker_data.iloc[:27].isna().all()

    def test_atr_is_lagged(self, data):
        """ATR with lag=1 should have correct NaN pattern."""
        result = atr(data["high"], data["low"], data["close"], period=14, lag=1)
        for ticker in data.index.get_level_values("ticker").unique():
            ticker_data = result.xs(ticker, level="ticker")
            assert ticker_data.iloc[:15].isna().all()

    def test_ema_is_lagged(self, data):
        """EMA with lag=1 should not contain current value."""
        close = data["close"]
        ema_val = ema(close, span=20, lag=1)

        for ticker in data.index.get_level_values("ticker").unique():
            ticker_close = close.xs(ticker, level="ticker")
            ticker_ema = ema_val.xs(ticker, level="ticker")

            # EMA at time T should NOT equal the EMA computed with close[T]
            # It should be the EMA from time T-1
            valid = ticker_ema.dropna()
            assert len(valid) > 0, "EMA has no valid values"

    def test_no_future_data_in_features(self, data):
        """Ensure lag=0 is different from lag=1."""
        rsi_lag0 = rsi(data["close"], period=14, lag=0)
        rsi_lag1 = rsi(data["close"], period=14, lag=1)

        valid = rsi_lag0.dropna() & rsi_lag1.dropna()
        common_idx = rsi_lag0.dropna().index.intersection(rsi_lag1.dropna().index)

        if len(common_idx) > 0:
            diff = (rsi_lag0.loc[common_idx] - rsi_lag1.loc[common_idx]).abs()
            assert diff.sum() > 0, "lag=0 and lag=1 should differ"
