# Regime-Aware Transfer Learning for Short-Side Alpha Generation in Vietnam's Equity Market

## Overview

A quantitative research framework that combines **Hidden Markov Model regime detection** with **LightGBM transfer learning** to predict underperforming stocks in Vietnam's equity market (HOSE). The model pre-trains on multi-cycle data and adaptively fine-tunes upon regime shifts.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Data Ingestion  │────▶│ Feature Engineer  │────▶│  Regime Detection  │
│  (vnstock API)   │     │  (~50 features)   │     │  (HMM + Rules)     │
└─────────────────┘     └──────────────────┘     └────────┬───────────┘
                                                          │
                         ┌────────────────────────────────┘
                         ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────┐
│   Base LightGBM   │──▶│ Transfer Learning │──▶│  Ensemble Scoring    │
│   (Pre-trained)   │   │ (Fine-tune w/     │   │ (0.6*base+0.4*adapt) │
│                   │   │  time-decay)      │   │                      │
└──────────────────┘   └──────────────────┘   └──────────┬───────────┘
                                                          │
                         ┌────────────────────────────────┘
                         ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────┐
│  Walk-Forward     │──▶│  VN Backtest      │──▶│  Statistical Tests   │
│  Validation       │   │  (T+1, ±7% CB)   │   │  (DM, Bootstrap CI)  │
└──────────────────┘   └──────────────────┘   └──────────────────────┘
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r vn_regime_transfer/requirements.txt

# 2. Run full pipeline
python -m vn_regime_transfer.run_pipeline

# 3. Quick mode (fewer folds, faster)
python -m vn_regime_transfer.run_pipeline --quick

# 4. Run tests
pytest vn_regime_transfer/tests/ -v
```

## Vietnam Market Constraints

| Constraint | Implementation |
|---|---|
| Circuit Breakers ±7% | Skip entry/exit on limit-hit days |
| Settlement T+2.5 | Entry delay T+1 from signal |
| Commission | 0.15% × 2 (buy + sell) |
| Slippage | 0.1% per trade |
| Position Sizing | 2% max risk per position |

## Project Structure

```
vn_regime_transfer/
├── config.py              # All hyperparameters & constants
├── data/                  # Data ingestion & processing
│   ├── downloader.py      # vnstock API wrapper
│   ├── processor.py       # Clean, flag circuit breakers
│   └── schema.py          # Typed column definitions
├── features/              # Feature engineering
│   ├── technical.py       # ~40 TA indicators (RSI, MACD, ATR...)
│   ├── microstructure.py  # VN-specific (CB proximity, limit hits)
│   ├── foreign_flow.py    # Foreign flow proxy features
│   └── builder.py         # Feature matrix + anti-leakage validation
├── regime/                # Regime detection
│   ├── hmm_detector.py    # GaussianHMM (3 states, expanding window)
│   ├── changepoint.py     # PELT structural break detection
│   ├── rule_based.py      # EMA200 + ATR percentile filters
│   └── hybrid.py          # Combined regime classification
├── model/                 # ML models
│   ├── base_lgb.py        # Pre-trained LightGBM
│   ├── transfer_lgb.py    # Fine-tune with init_model + time-decay
│   ├── ensemble.py        # Weighted base + adapted ensemble
│   └── baselines.py       # Static LGB, Random, Buy-and-Hold
├── validation/            # Validation & testing
│   ├── walk_forward.py    # Walk-forward engine
│   ├── statistical_tests.py  # DM test, Bootstrap CI, PSI
│   └── metrics.py         # Sharpe, Calmar, CVaR, hit rate
├── backtest/              # Execution simulation
│   └── executor_vn.py     # VN-specific backtest engine
├── reporting/             # Output generation
│   ├── plots.py           # Equity curves, regime overlays
│   ├── tables.py          # Ablation study, regime performance
│   └── paper_report.py    # LaTeX report generator
├── tests/                 # Unit tests
│   ├── test_features.py   # Anti-leakage validation
│   ├── test_regime.py     # Regime detection tests
│   ├── test_model.py      # Model train/predict tests
│   └── test_backtest.py   # Metrics & statistical test checks
├── run_pipeline.py        # Main entry point
└── requirements.txt       # Dependencies
```

## Key Design Decisions

1. **Anti-leakage**: ALL features use `shift(1)` minimum. Target uses forward-looking `Close(T+3)`.
2. **Reproducibility**: Single `config.py` with `RANDOM_SEED=42` applied everywhere.
3. **Transfer Learning**: Uses LightGBM's `init_model` for warm-start + time-decay weighting `w_t = α^(T-t)`.
4. **Regime-aware**: Fine-tuning is conditioned on current regime (only uses same-regime samples).
5. **Walk-forward**: Expanding window (no random splits), 60m train / 6m test / 3m step.

## License

MIT
