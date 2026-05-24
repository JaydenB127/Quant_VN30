# -*- coding: utf-8 -*-
"""
Deep Learning baselines: LSTM and Transformer for time-series classification.

Both models consume a fixed-length lookback window of features and output
a probability of the stock dropping.  They follow the same
``fit(X, y) / predict(X)`` interface as the LightGBM baselines so the
walk-forward engine treats them identically.

Design notes
~~~~~~~~────
* Input is *tabular* features (not raw OHLCV) — this keeps the comparison
  fair, because every model sees the *same* engineered features.
* The lookback window is formed by reshaping the flat feature matrix into
  ``(batch, seq_len, n_features)`` using the per-ticker history.
* Training uses early stopping on a validation split (last 20 % of train).
"""
from __future__ import annotations

import logging
import math
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
#  Sequence builders — tabular → (batch, seq_len, features)
# ═══════════════════════════════════════════════════════════════════════

def build_sequences_with_indices(
    X: pd.DataFrame,
    y: pd.Series,
    seq_len: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a MultiIndex (date, ticker) feature matrix into 3-D sequences
    with original row indices aligned for correct prediction mapping.
    """
    tickers = X.index.get_level_values("ticker").unique()
    X_seqs: List[np.ndarray] = []
    y_seqs: List[np.ndarray] = []
    indices: List[int] = []

    for ticker in tickers:
        mask = X.index.get_level_values("ticker") == ticker
        pos_indices = np.where(mask)[0]
        xf = X.loc[mask].values.astype(np.float32)
        yf = y.loc[mask].values.astype(np.float32)

        for i in range(seq_len, len(xf)):
            X_seqs.append(xf[i - seq_len : i])
            y_seqs.append(yf[i])
            indices.append(pos_indices[i])

    if not X_seqs:
        return (
            np.empty((0, seq_len, X.shape[1]), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64)
        )

    return np.stack(X_seqs), np.array(y_seqs, dtype=np.float32), np.array(indices, dtype=np.int64)


def build_sequences(
    X: pd.DataFrame,
    y: pd.Series,
    seq_len: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible sequence builder returning only X_seq and y_seq.
    """
    X_seq, y_seq, _ = build_sequences_with_indices(X, y, seq_len)
    return X_seq, y_seq


# ═══════════════════════════════════════════════════════════════════════
#  Picklable PyTorch Networks (Module-level scope)
# ═══════════════════════════════════════════════════════════════════════

def get_lstm_net_class():
    torch, nn, *_ = _torch_imports()

    class LSTMNet(nn.Module):
        def __init__(self, n_features: int, hidden: int, n_layers: int, drop: float):
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
            
    return LSTMNet


def get_transformer_net_class():
    torch, nn, *_ = _torch_imports()

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 500):
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

    class TransformerNet(nn.Module):
        def __init__(self, n_features: int, d_model: int, n_heads: int, n_layers: int, dim_ff: int, drop: float, seq_len: int):
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 10)
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

    return TransformerNet


# ═══════════════════════════════════════════════════════════════════════
#  Base Deep Model ABC / Parent class
# ═══════════════════════════════════════════════════════════════════════

class BaseDeepModel:
    """
    Common base class for Deep Learning Baselines (LSTM & Transformer).
    Deduplicates training loops, predictions, and sequence alignment.
    """

    def __init__(
        self,
        seq_len: int = 20,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 256,
        patience: int = 10,
        seed: int = RANDOM_SEED,
    ):
        self.seq_len = seq_len
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.model = None
        self.device = None
        self.n_features = 0

    def _build_net(self, n_features: int):
        raise NotImplementedError("Subclasses must implement _build_net")

    def _get_optimizer_and_scheduler(self, net, torch, optim):
        raise NotImplementedError("Subclasses must implement _get_optimizer_and_scheduler")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame = None,
        y_valid: pd.Series = None,
    ) -> BaseDeepModel:
        torch, nn, DataLoader, TensorDataset, optim = _torch_imports()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_features = X_train.shape[1]

        # Build sequences
        Xs, ys, _ = build_sequences_with_indices(X_train, y_train, self.seq_len)
        if len(Xs) == 0:
            logger.warning("%s: no sequences built — skipping training", self.__class__.__name__)
            return self

        # Replace NaN with 0 in sequences
        Xs = np.nan_to_num(Xs, nan=0.0)

        # Validation sequences
        if X_valid is not None and y_valid is not None:
            Xv, yv, _ = build_sequences_with_indices(X_valid, y_valid, self.seq_len)
            Xv = np.nan_to_num(Xv, nan=0.0)
        else:
            # Split last 20 %
            split = int(len(Xs) * 0.8)
            Xv, yv = Xs[split:], ys[split:]
            Xs, ys = Xs[:split], ys[:split]

        train_ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys))
        val_ds = TensorDataset(torch.from_numpy(Xv), torch.from_numpy(yv))
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size * 2)

        net = self._build_net(self.n_features).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        opt, scheduler = self._get_optimizer_and_scheduler(net, torch, optim)

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
            
            # Step scheduler if it needs metrics (ReduceLROnPlateau)
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 20 == 0 or epoch == 1:
                logger.info(
                    "%s epoch %3d | train %.4f | val %.4f | best %.4f",
                    self.__class__.__name__[:4], epoch, train_loss, val_loss, best_val_loss,
                )

            if patience_counter >= self.patience:
                logger.info("%s early stop at epoch %d", self.__class__.__name__, epoch)
                break

        if best_state is not None:
            net.load_state_dict(best_state)
        self.model = net.eval().to(self.device)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return drop probabilities for each row aligned correctly by MultiIndex positions."""
        if self.model is None:
            logger.warning("%s not trained — returning 0.5", self.__class__.__name__)
            return np.full(len(X), 0.5)

        torch, *_ = _torch_imports()
        
        # Build sequences and get original integer index mappings
        Xs, _, indices = build_sequences_with_indices(X, pd.Series(0, index=X.index), self.seq_len)
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

        # Correct alignment: map predictions back to their precise original integer row positions
        full = np.full(len(X), 0.5)
        full[indices] = pred_arr
        return full

    def predict_binary(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict(X) >= threshold).astype(int)


# ═══════════════════════════════════════════════════════════════════════
#  LSTM Model (Refactored Subclass)
# ═══════════════════════════════════════════════════════════════════════

class LSTMBaseline(BaseDeepModel):
    """
    Two-layer LSTM with dropout → dense head → sigmoid.
    Inherits training, prediction, and correct sequence alignment from BaseDeepModel.
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
        super().__init__(
            seq_len=seq_len,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            seed=seed
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

    def _build_net(self, n_features: int):
        LSTMNet = get_lstm_net_class()
        return LSTMNet(n_features, self.hidden_size, self.num_layers, self.dropout)

    def _get_optimizer_and_scheduler(self, net, torch, optim):
        opt = optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=3
        )
        return opt, scheduler


# ═══════════════════════════════════════════════════════════════════════
#  Transformer Model (Refactored Subclass)
# ═══════════════════════════════════════════════════════════════════════

class TransformerBaseline(BaseDeepModel):
    """
    Transformer encoder (no decoder) for tabular time-series classification.
    Inherits training, prediction, and correct sequence alignment from BaseDeepModel.
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
        super().__init__(
            seq_len=seq_len,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            seed=seed
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_ff = dim_ff
        self.dropout = dropout

    def _build_net(self, n_features: int):
        TransformerNet = get_transformer_net_class()
        return TransformerNet(
            n_features, self.d_model, self.n_heads,
            self.n_layers, self.dim_ff, self.dropout, self.seq_len
        )

    def _get_optimizer_and_scheduler(self, net, torch, optim):
        opt = optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        return opt, scheduler
