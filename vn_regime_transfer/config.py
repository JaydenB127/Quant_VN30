# -*- coding: utf-8 -*-
"""
Central configuration for the entire pipeline.

Every tuneable hyper-parameter, path, and constant lives here so that
the whole experiment is reproducible from a single file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Reproducibility ────────────────────────────────────────────────────
RANDOM_SEED: int = 42

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
CACHE_DIR: Path = PROJECT_ROOT / "data" / "cache"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
REPORT_DIR: Path = OUTPUT_DIR / "reports"
MODEL_DIR: Path = OUTPUT_DIR / "models"
MLFLOW_DIR: Path = OUTPUT_DIR / "mlruns"

for _d in (DATA_DIR, CACHE_DIR, OUTPUT_DIR, REPORT_DIR, MODEL_DIR, MLFLOW_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── Universe ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class UniverseConfig:
    """Stock universe definition."""
    name: str = "VN30"
    exchange: str = "HOSE"
    # VN30 tickers as of 2024 — will be refreshed by downloader
    tickers: Tuple[str, ...] = (
        "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR",
        "HDB", "HPG", "MBB", "MSN", "MWG", "PLX", "POW", "SAB",
        "SHB", "SSB", "SSI", "STB", "TCB", "TPB", "VCB", "VHM",
        "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
    )
    benchmark_ticker: str = "VNINDEX"


# ── Data ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DataConfig:
    """Date ranges and data parameters."""
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"
    train_end: str = "2024-06-30"   # inclusive
    test_start: str = "2024-07-01"
    freq: str = "1D"
    source: str = "VCI"             # vnstock data source


# ── Feature Engineering ────────────────────────────────────────────────
@dataclass(frozen=True)
class FeatureConfig:
    """Feature construction parameters — ALL use shift(1) minimum."""
    min_lag: int = 1                # anti-leakage: shift(1)
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    ema_periods: Tuple[int, ...] = (20, 50, 200)
    roc_periods: Tuple[int, ...] = (5, 10, 20)
    volume_ma_period: int = 20
    adx_period: int = 14
    # Vietnam-specific
    circuit_breaker_pct: float = 0.07   # ±7 % daily limit
    limit_hit_window: int = 5           # rolling window for limit-hit count


# ── Target ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TargetConfig:
    """Forward-looking target definition."""
    horizon: int = 3                     # T+3 days
    drop_threshold: float = -0.015       # < -1.5 % → label = 1 (drop)
    # y = 1 if Close(T+horizon)/Close(T) - 1 < drop_threshold


# ── Regime Detection ──────────────────────────────────────────────────
@dataclass(frozen=True)
class RegimeConfig:
    """HMM + rule-based regime detection."""
    n_states: int = 3                    # bull / sideways / bear
    hmm_covariance_type: str = "full"
    hmm_n_iter: int = 100
    hmm_min_train_days: int = 252        # ≥ 1 year before first prediction
    hmm_refit_interval: int = 60         # refit HMM every N days
    # Changepoint detection (ruptures)
    cp_model: str = "rbf"
    cp_min_size: int = 20
    cp_penalty: float = 10.0
    # Rule-based overrides
    ema_trend_period: int = 200
    atr_percentile_threshold: float = 80.0
    foreign_flow_ma: int = 20


# ── Model ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelConfig:
    """LightGBM base + transfer learning parameters."""
    # Base model
    base_params: Dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda_l1": 10.0,
        "lambda_l2": 50.0,
        "min_child_samples": 50,
        "random_state": RANDOM_SEED,
        "verbosity": -1,
        "n_jobs": -1,
    })
    base_num_boost_round: int = 1000
    base_early_stopping: int = 50
    # Transfer / fine-tune
    ft_num_boost_round: int = 80
    ft_learning_rate: float = 0.02     # lower LR for fine-tuning
    time_decay_alpha: float = 0.997    # w_t = α^(T-t)
    # Ensemble
    ensemble_w_base: float = 0.6
    ensemble_w_adapted: float = 0.4


# ── Walk-Forward Validation ───────────────────────────────────────────
@dataclass(frozen=True)
class ValidationConfig:
    """Walk-forward cross-validation parameters."""
    initial_train_months: int = 60       # 5 years
    test_months: int = 6
    step_months: int = 3                 # slide 3 months
    min_folds: int = 5
    # Bootstrap
    n_bootstrap: int = 10_000
    bootstrap_ci: float = 0.95
    # PSI
    psi_threshold: float = 0.25         # > 0.25 → significant drift


# ── Backtest / Execution ──────────────────────────────────────────────
@dataclass(frozen=True)
class BacktestConfig:
    """Vietnam-specific execution simulation."""
    entry_delay_days: int = 1            # signal T → trade T+1
    hold_period_days: int = 3            # minimum hold
    commission_rate: float = 0.0015      # 0.15 % one-way
    slippage_rate: float = 0.001         # 0.1 %
    circuit_breaker_pct: float = 0.065   # skip if |ret| ≥ 6.5 %
    max_position_risk: float = 0.02      # 2 % portfolio per position
    initial_capital: float = 1_000_000_000.0  # 1 tỷ VND
    trading_days_per_year: int = 252


# ── Deep Learning Baselines ───────────────────────────────────────────
@dataclass(frozen=True)
class DeepLearningConfig:
    """LSTM and Transformer baseline parameters."""
    seq_len: int = 20                    # look-back window (trading days)
    # LSTM
    lstm_hidden: int = 64
    lstm_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_lr: float = 1e-3
    lstm_epochs: int = 100
    lstm_batch: int = 256
    lstm_patience: int = 10
    # Transformer
    tf_d_model: int = 64
    tf_n_heads: int = 4
    tf_n_layers: int = 2
    tf_dim_ff: int = 128
    tf_dropout: float = 0.2
    tf_lr: float = 1e-3
    tf_epochs: int = 100
    tf_batch: int = 256
    tf_patience: int = 10
    # Whether to include DL baselines (can be slow)
    enabled: bool = True


# ── Performance / Optimisation ────────────────────────────────────────
@dataclass(frozen=True)
class PerformanceConfig:
    """Settings for scaling to full universe."""
    n_jobs: int = -1                     # parallel jobs (-1 = all cores)
    feature_cache: bool = True           # cache feature matrix to parquet
    use_float32: bool = True             # downcast features to float32
    chunk_size: int = 10                 # tickers per parallel chunk
    hmm_refit_interval: int = 60         # refit HMM every N days
    verbose: int = 1                     # 0 = silent, 1 = info, 2 = debug


# ── Convenience aggregate ─────────────────────────────────────────────
@dataclass
class PipelineConfig:
    """Top-level config aggregating all sub-configs."""
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    deep_learning: DeepLearningConfig = field(default_factory=DeepLearningConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    seed: int = RANDOM_SEED


# Singleton default
CFG = PipelineConfig()
