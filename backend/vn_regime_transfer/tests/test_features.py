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
        """RSI with lag=1 should shift RSI with lag=0 by 1."""
        result_lag1 = rsi(data["close"], period=14, lag=1)
        result_lag0 = rsi(data["close"], period=14, lag=0)
        
        for ticker in data.index.get_level_values("ticker").unique():
            r1 = result_lag1.xs(ticker, level="ticker")
            r0 = result_lag0.xs(ticker, level="ticker")
            assert np.isnan(r1.iloc[0])
            np.testing.assert_array_almost_equal(r1.iloc[1:].values, r0.iloc[:-1].values)

    def test_macd_is_lagged(self, data):
        """MACD with lag=1 should shift output by 1."""
        macd_l1, _, _ = macd(data["close"], lag=1)
        macd_l0, _, _ = macd(data["close"], lag=0)
        
        for ticker in data.index.get_level_values("ticker").unique():
            m1 = macd_l1.xs(ticker, level="ticker")
            m0 = macd_l0.xs(ticker, level="ticker")
            assert np.isnan(m1.iloc[0])
            np.testing.assert_array_almost_equal(m1.iloc[1:].values, m0.iloc[:-1].values)

    def test_atr_is_lagged(self, data):
        """ATR with lag=1 should shift ATR with lag=0 by 1."""
        result_lag1 = atr(data["high"], data["low"], data["close"], period=14, lag=1)
        result_lag0 = atr(data["high"], data["low"], data["close"], period=14, lag=0)
        
        for ticker in data.index.get_level_values("ticker").unique():
            r1 = result_lag1.xs(ticker, level="ticker")
            r0 = result_lag0.xs(ticker, level="ticker")
            assert np.isnan(r1.iloc[0])
            np.testing.assert_array_almost_equal(r1.iloc[1:].values, r0.iloc[:-1].values)

    def test_ema_is_lagged(self, data):
        """EMA with lag=1 should shift EMA with lag=0 by 1."""
        close = data["close"]
        ema_val_lag1 = ema(close, span=20, lag=1)
        ema_val_lag0 = ema(close, span=20, lag=0)

        for ticker in data.index.get_level_values("ticker").unique():
            e1 = ema_val_lag1.xs(ticker, level="ticker")
            e0 = ema_val_lag0.xs(ticker, level="ticker")
            assert np.isnan(e1.iloc[0])
            np.testing.assert_array_almost_equal(e1.iloc[1:].values, e0.iloc[:-1].values)

    def test_no_future_data_in_features(self, data):
        """Ensure lag=0 is different from lag=1."""
        rsi_lag0 = rsi(data["close"], period=14, lag=0)
        rsi_lag1 = rsi(data["close"], period=14, lag=1)

        common_idx = rsi_lag0.dropna().index.intersection(rsi_lag1.dropna().index)

        if len(common_idx) > 0:
            diff = (rsi_lag0.loc[common_idx] - rsi_lag1.loc[common_idx]).abs()
            assert diff.sum() > 0, "lag=0 and lag=1 should differ"
