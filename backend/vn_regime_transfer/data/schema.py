# -*- coding: utf-8 -*-
"""
Typed dataclass schemas for market data flowing through the pipeline.

These serve as documentation *and* runtime contracts — every DataFrame
that crosses a module boundary must match the column set defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class OHLCVColumns:
    """Canonical column names for raw price data."""
    date: str = "date"
    ticker: str = "ticker"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str = "volume"


@dataclass(frozen=True)
class ProcessedColumns(OHLCVColumns):
    """Additional columns after processing."""
    adj_close: str = "adj_close"
    returns: str = "returns"
    log_returns: str = "log_returns"
    is_limit_up: str = "is_limit_up"
    is_limit_down: str = "is_limit_down"
    ref_price: str = "ref_price"             # reference price for circuit breaker
    is_suspended: str = "is_suspended"


@dataclass(frozen=True)
class FeatureColumns:
    """Feature group names — each maps to a list of actual column names."""
    momentum: str = "momentum"
    volatility: str = "volatility"
    volume_feat: str = "volume_feat"
    trend: str = "trend"
    mean_reversion: str = "mean_reversion"
    microstructure: str = "microstructure"
    foreign_flow: str = "foreign_flow"


@dataclass(frozen=True)
class TargetColumns:
    """Target / label columns."""
    forward_return: str = "fwd_return"       # Close(T+h)/Close(T) - 1
    label: str = "label"                     # 1 = drop, 0 = not drop


@dataclass(frozen=True)
class RegimeColumns:
    """Regime detection output columns."""
    hmm_state: str = "hmm_state"
    hmm_prob: str = "hmm_prob"               # posterior prob of current state
    changepoint_flag: str = "cp_flag"
    days_since_cp: str = "days_since_cp"
    rule_regime: str = "rule_regime"
    final_regime: str = "regime"             # hybrid final label


# ── Singleton column definitions ──────────────────────────────────────
OHLCV = OHLCVColumns()
PROCESSED = ProcessedColumns()
FEATURES = FeatureColumns()
TARGET = TargetColumns()
REGIME = RegimeColumns()
