# -*- coding: utf-8 -*-
"""
Deep Learning baselines: LSTM and Transformer for time-series classification.

Both models consume a fixed-length lookback window of features and output
a probability of the stock dropping.  They follow the same
``fit(X, y) / predict(X)`` interface as the LightGBM baselines so the
walk-forward engine treats them identically.

Design notes
~~~~~~~~~~~~
* Input is *tabular* features (not raw OHLCV) — this keeps the comparison
  fair, because every model sees the *same* engineered features.
* The lookback window is formed by reshaping the flat feature matrix into
  ``(batch, seq_len, n_features)`` using the per-ticker history.
* Training uses early stopping on a validation split (last 20 % of train).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import CFG, RANDOM_SEED

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy PyTorch imports — only fail when actually called
# ---------------------------------------------------------------------------

def _torch_imports():
    """Return (torch, nn, DataLoader, TensorDataset, optim)."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        import torch.optim as optim
        return torch, nn, DataLoader, TensorDataset, optim
    except ImportError:
        raise ImportError(
            "PyTorch is required for deep-learning baselines.  "
            "Install with:  pip install torch"
        )


# ═══════════════════════════════════════════════════════════════════════
#  Sequence builder — tabular → (batch, seq_len, features)
# ═══════════════════════════════════════════════════════════════════════

def build_sequences(
    X: pd.DataFrame,
    y: pd.Series,
    seq_len: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a MultiIndex (date, ticker) feature matrix into 3-D sequences.

    For every (date, ticker) row we look back ``seq_len`` trading days
    *for the same ticker* and stack the features into a 2-D window.

    Parameters
    ----------
    X : pd.DataFrame   — MultiIndex (date, ticker), columns = features
    y : pd.Series       — aligned labels
    seq_len : int       — look-back window length

    Returns
    -------
    X_seq : np.ndarray, shape (N, seq_len, n_features)
    y_seq : np.ndarray, shape (N,)
    """
    tickers = X.index.get_level_values("ticker").unique()
    X_seqs: List[np.ndarray] = []
    y_seqs: List[np.ndarray] = []

    for ticker in tickers:
        # Slice for this ticker, keep temporal order
        mask = X.index.get_level_values("ticker") == ticker
        xf = X.loc[mask].values.astype(np.float32)
        yf = y.loc[mask].values.astype(np.float32)

        for i in range(seq_len, len(xf)):
            X_seqs.append(xf[i - seq_len : i])
            y_seqs.append(yf[i])

    if not X_seqs:
        return np.empty((0, seq_len, X.shape[1]), dtype=np.float32), np.empty(0)

    return np.stack(X_seqs), np.array(y_seqs, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════
#  LSTM Model
# ═══════════════════════════════════════════════════════════════════════

class LSTMBaseline:
    """
    Two-layer LSTM with dropout → dense head → sigmoid.

    Parameters
    ----------
    hidden_size : int
        LSTM hidden dimension.
    num_layers : int
        Stacked LSTM layers.
    dropout : float
        Dropout between LSTM layers & before the head.
    seq_len : int
        Look-back window (in trading days).
    lr : float
        Learning rate.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    patience : int
        Early-stopping patience (epochs).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        seq_len: int = 20,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 256,
        patience: int = 10,
        seed: int = RANDOM_SEED,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_len = seq_len
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.model = None
        self.device = None
        self.n_features: int = 0

    # ── internal PyTorch module ──────────────────────────────────────
    @staticmethod
    def _build_net(n_features, hidden, n_layers, drop):
        torch, nn, *_ = _torch_imports()

        class _LSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features,
                    hidden_size=hidden,
                    num_layers=n_layers,
                    batch_first=True,
                    dropout=drop if n_layers > 1 else 0.0,
                )
                self.dropout = nn.Dropout(drop)
                self.head = nn.Sequential(
                    nn.Linear(hidden, 32),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(32, 1),
                )

            def forward(self, x):               # (B, T, F)
                out, _ = self.lstm(x)            # (B, T, H)
                last = out[:, -1, :]             # (B, H)  — last time-step
                return self.head(self.dropout(last)).squeeze(-1)

        return _LSTM()

    # ── public interface ─────────────────────────────────────────────
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame = None,
        y_valid: pd.Series = None,
    ) -> "LSTMBaseline":
        torch, nn, DataLoader, TensorDataset, optim = _torch_imports()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_features = X_train.shape[1]

        # Build sequences
        Xs, ys = build_sequences(X_train, y_train, self.seq_len)
        if len(Xs) == 0:
            logger.warning("LSTM: no sequences built — skipping training")
            return self

        # Replace NaN with 0 in sequences
        Xs = np.nan_to_num(Xs, nan=0.0)

        # Validation sequences
        if X_valid is not None and y_valid is not None:
            Xv, yv = build_sequences(X_valid, y_valid, self.seq_len)
            Xv = np.nan_to_num(Xv, nan=0.0)
        else:
            # Split last 20 %
            split = int(len(Xs) * 0.8)
            Xv, yv = Xs[split:], ys[split:]
            Xs, ys = Xs[:split], ys[:split]

        train_ds = TensorDataset(
            torch.from_numpy(Xs), torch.from_numpy(ys)
        )
        val_ds = TensorDataset(
            torch.from_numpy(Xv), torch.from_numpy(yv)
        )
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size * 2)

        net = self._build_net(
            self.n_features, self.hidden_size, self.num_layers, self.dropout
        ).to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        opt = optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=3
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            # --- train ---
            net.train()
            train_loss = 0.0
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                logits = net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(train_ds)

            # --- validate ---
            net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    val_loss += criterion(net(xb), yb).item() * len(xb)
            val_loss /= len(val_ds)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 20 == 0 or epoch == 1:
                logger.info(
                    "LSTM epoch %3d | train %.4f | val %.4f | best %.4f",
                    epoch, train_loss, val_loss, best_val_loss,
                )

            if patience_counter >= self.patience:
                logger.info("LSTM early stop at epoch %d", epoch)
                break

        if best_state is not None:
            net.load_state_dict(best_state)
        self.model = net.eval().to(self.device)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return drop probabilities for each row."""
        if self.model is None:
            logger.warning("LSTM not trained — returning 0.5")
            return np.full(len(X), 0.5)

        torch, *_ = _torch_imports()
        # Need sequences; for rows without enough history → default 0.5
        Xs, _ = build_sequences(X, pd.Series(0, index=X.index), self.seq_len)
        Xs = np.nan_to_num(Xs, nan=0.0)

        if len(Xs) == 0:
            return np.full(len(X), 0.5)

        preds = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(Xs), self.batch_size * 2):
                batch = torch.from_numpy(Xs[start : start + self.batch_size * 2]).to(self.device)
                logits = self.model(batch)
                preds.append(torch.sigmoid(logits).cpu().numpy())

        pred_arr = np.concatenate(preds)

        # Align back — sequences drop the first seq_len rows per ticker
        n_tickers = X.index.get_level_values("ticker").nunique()
        n_dropped = n_tickers * self.seq_len
        full = np.full(len(X), 0.5)
        if len(pred_arr) <= len(full):
            full[n_dropped : n_dropped + len(pred_arr)] = pred_arr
        return full

    def predict_binary(self, X, threshold=0.5):
        return (self.predict(X) >= threshold).astype(int)


# ═══════════════════════════════════════════════════════════════════════
#  Transformer Model
# ═══════════════════════════════════════════════════════════════════════

class TransformerBaseline:
    """
    Transformer encoder (no decoder) for tabular time-series classification.

    Architecture
    ~~~~~~~~~~~~
    Input projection → Positional encoding → N × TransformerEncoderLayer
    → mean-pool over time → dense head → sigmoid.

    Parameters
    ----------
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of encoder layers.
    dim_ff : int
        Feed-forward hidden dim inside each layer.
    dropout : float
        Dropout rate.
    seq_len : int
        Look-back window.
    lr : float
        Learning rate.
    epochs : int
        Max training epochs.
    batch_size : int
        Mini-batch size.
    patience : int
        Early-stopping patience.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.2,
        seq_len: int = 20,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 256,
        patience: int = 10,
        seed: int = RANDOM_SEED,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.seq_len = seq_len
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.model = None
        self.device = None
        self.n_features: int = 0

    @staticmethod
    def _build_net(n_features, d_model, n_heads, n_layers, dim_ff, drop, seq_len):
        torch, nn, *_ = _torch_imports()
        import math

        class _PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=500):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                pos = torch.arange(0, max_len).unsqueeze(1).float()
                div = torch.exp(
                    torch.arange(0, d_model, 2).float()
                    * (-math.log(10000.0) / d_model)
                )
                pe[:, 0::2] = torch.sin(pos * div)
                pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
                self.register_buffer("pe", pe.unsqueeze(0))   # (1, T, D)

            def forward(self, x):
                return x + self.pe[:, : x.size(1), :]

        class _Transformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Linear(n_features, d_model)
                self.pos_enc = _PositionalEncoding(d_model, max_len=seq_len + 10)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=dim_ff,
                    dropout=drop,
                    batch_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer, num_layers=n_layers
                )
                self.dropout = nn.Dropout(drop)
                self.head = nn.Sequential(
                    nn.Linear(d_model, 32),
                    nn.GELU(),
                    nn.Dropout(drop),
                    nn.Linear(32, 1),
                )

            def forward(self, x):                 # (B, T, F)
                x = self.input_proj(x)            # (B, T, D)
                x = self.pos_enc(x)
                x = self.encoder(x)               # (B, T, D)
                x = x.mean(dim=1)                 # (B, D)  mean-pool
                return self.head(self.dropout(x)).squeeze(-1)

        return _Transformer()

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame = None,
        y_valid: pd.Series = None,
    ) -> "TransformerBaseline":
        torch, nn, DataLoader, TensorDataset, optim = _torch_imports()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_features = X_train.shape[1]

        Xs, ys = build_sequences(X_train, y_train, self.seq_len)
        if len(Xs) == 0:
            logger.warning("Transformer: no sequences — skipping")
            return self
        Xs = np.nan_to_num(Xs, nan=0.0)

        if X_valid is not None and y_valid is not None:
            Xv, yv = build_sequences(X_valid, y_valid, self.seq_len)
            Xv = np.nan_to_num(Xv, nan=0.0)
        else:
            split = int(len(Xs) * 0.8)
            Xv, yv = Xs[split:], ys[split:]
            Xs, ys = Xs[:split], ys[:split]

        train_dl = DataLoader(
            TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys)),
            batch_size=self.batch_size, shuffle=True,
        )
        val_dl = DataLoader(
            TensorDataset(torch.from_numpy(Xv), torch.from_numpy(yv)),
            batch_size=self.batch_size * 2,
        )

        net = self._build_net(
            self.n_features, self.d_model, self.n_heads,
            self.n_layers, self.dim_ff, self.dropout, self.seq_len,
        ).to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        opt = optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        best_val = float("inf")
        best_state = None
        wait = 0

        for epoch in range(1, self.epochs + 1):
            net.train()
            t_loss = 0.0
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                t_loss += loss.item() * len(xb)
            t_loss /= len(Xs)
            scheduler.step()

            net.eval()
            v_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    v_loss += criterion(net(xb), yb).item() * len(xb)
            v_loss /= len(Xv)

            if v_loss < best_val:
                best_val = v_loss
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if epoch % 20 == 0 or epoch == 1:
                logger.info(
                    "TF  epoch %3d | train %.4f | val %.4f | best %.4f",
                    epoch, t_loss, v_loss, best_val,
                )
            if wait >= self.patience:
                logger.info("Transformer early stop at epoch %d", epoch)
                break

        if best_state is not None:
            net.load_state_dict(best_state)
        self.model = net.eval().to(self.device)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), 0.5)

        torch, *_ = _torch_imports()
        Xs, _ = build_sequences(X, pd.Series(0, index=X.index), self.seq_len)
        Xs = np.nan_to_num(Xs, nan=0.0)
        if len(Xs) == 0:
            return np.full(len(X), 0.5)

        preds = []
        self.model.eval()
        with torch.no_grad():
            for s in range(0, len(Xs), self.batch_size * 2):
                batch = torch.from_numpy(Xs[s : s + self.batch_size * 2]).to(self.device)
                preds.append(torch.sigmoid(self.model(batch)).cpu().numpy())

        pred_arr = np.concatenate(preds)
        n_tickers = X.index.get_level_values("ticker").nunique()
        n_dropped = n_tickers * self.seq_len
        full = np.full(len(X), 0.5)
        if len(pred_arr) <= len(full):
            full[n_dropped : n_dropped + len(pred_arr)] = pred_arr
        return full

    def predict_binary(self, X, threshold=0.5):
        return (self.predict(X) >= threshold).astype(int)
