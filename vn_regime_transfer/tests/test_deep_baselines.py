# -*- coding: utf-8 -*-
"""Tests for LSTM and Transformer baselines."""
import numpy as np
import pandas as pd
import pytest


def _can_import_torch():
    try:
        import torch
        return True
    except ImportError:
        return False


def _make_sequential_data(n_dates=200, n_tickers=3, n_features=10, seed=42):
    """Create synthetic MultiIndex (date, ticker) data for DL testing."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    tickers = [f"T{i}" for i in range(n_tickers)]

    rows = []
    for ticker in tickers:
        for date in dates:
            row = {"date": date, "ticker": ticker}
            for j in range(n_features):
                row[f"feat_{j}"] = rng.randn()
            row["label"] = rng.binomial(1, 0.3)
            rows.append(row)

    df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    return df, [f"feat_{j}" for j in range(n_features)]


class TestBuildSequences:
    """Test the sequence builder for DL models."""

    def test_basic_shape(self):
        from vn_regime_transfer.model.deep_baselines import build_sequences

        df, feat_cols = _make_sequential_data(n_dates=100, n_tickers=2)
        X = df[feat_cols]
        y = df["label"]

        X_seq, y_seq = build_sequences(X, y, seq_len=10)

        # Each ticker has 100 dates, so 90 sequences per ticker × 2 tickers
        assert X_seq.shape[0] == 90 * 2
        assert X_seq.shape[1] == 10  # seq_len
        assert X_seq.shape[2] == len(feat_cols)  # n_features
        assert len(y_seq) == X_seq.shape[0]

    def test_empty_with_short_data(self):
        from vn_regime_transfer.model.deep_baselines import build_sequences

        df, feat_cols = _make_sequential_data(n_dates=5, n_tickers=1)
        X = df[feat_cols]
        y = df["label"]

        X_seq, y_seq = build_sequences(X, y, seq_len=20)
        assert X_seq.shape[0] == 0

    def test_build_sequences_produces_correct_values(self):
        from vn_regime_transfer.model.deep_baselines import build_sequences

        df, feat_cols = _make_sequential_data(n_dates=30, n_tickers=1)
        X = df[feat_cols]
        y = df["label"]
        seq_len = 5

        X_seq, y_seq = build_sequences(X, y, seq_len=seq_len)

        # First sequence should be features from days 0-4
        x_vals = X.values
        np.testing.assert_array_almost_equal(X_seq[0], x_vals[:seq_len])
        # Label should be from day 5
        assert y_seq[0] == y.values[seq_len]


@pytest.mark.skipif(not _can_import_torch(), reason="PyTorch not installed")
class TestLSTM:
    """Test LSTM baseline."""

    def test_fit_predict(self):
        from vn_regime_transfer.model.deep_baselines import LSTMBaseline

        df, feat_cols = _make_sequential_data(n_dates=100, n_tickers=2)
        X = df[feat_cols]
        y = df["label"]

        lstm = LSTMBaseline(
            hidden_size=16, num_layers=1, dropout=0.1,
            seq_len=10, lr=1e-3, epochs=5, batch_size=32, patience=3,
        )
        lstm.fit(X, y)

        preds = lstm.predict(X)
        assert len(preds) == len(X)
        assert (preds >= 0).all() and (preds <= 1).all()

    def test_binary_prediction(self):
        from vn_regime_transfer.model.deep_baselines import LSTMBaseline

        df, feat_cols = _make_sequential_data(n_dates=80, n_tickers=1)
        lstm = LSTMBaseline(
            hidden_size=8, num_layers=1, seq_len=5,
            epochs=3, batch_size=16,
        )
        lstm.fit(df[feat_cols], df["label"])
        binary = lstm.predict_binary(df[feat_cols])
        assert set(np.unique(binary)).issubset({0, 1})


@pytest.mark.skipif(not _can_import_torch(), reason="PyTorch not installed")
class TestTransformer:
    """Test Transformer baseline."""

    def test_fit_predict(self):
        from vn_regime_transfer.model.deep_baselines import TransformerBaseline

        df, feat_cols = _make_sequential_data(n_dates=100, n_tickers=2)
        X = df[feat_cols]
        y = df["label"]

        tf = TransformerBaseline(
            d_model=16, n_heads=2, n_layers=1, dim_ff=32,
            seq_len=10, lr=1e-3, epochs=5, batch_size=32, patience=3,
        )
        tf.fit(X, y)

        preds = tf.predict(X)
        assert len(preds) == len(X)

    def test_untrained_returns_default(self):
        from vn_regime_transfer.model.deep_baselines import TransformerBaseline

        df, feat_cols = _make_sequential_data(n_dates=50, n_tickers=1)
        tf = TransformerBaseline()
        # Not trained — should return 0.5
        preds = tf.predict(df[feat_cols])
        assert (preds == 0.5).all()
