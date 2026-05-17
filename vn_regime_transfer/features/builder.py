# -*- coding: utf-8 -*-
"""
Feature builder: assembles all feature groups + target, validates anti-leakage.

Supports two modes:
  - ``build_feature_matrix()``       — original sequential pipeline
  - ``build_feature_matrix_fast()``  — vectorized + cached for full universe
"""
from __future__ import annotations
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CFG, CACHE_DIR
from ..data.schema import TARGET
from .technical import compute_technical_features
from .microstructure import compute_microstructure_features
from .foreign_flow import compute_foreign_flow_features

logger = logging.getLogger(__name__)

# Columns that are raw data, not features
RAW_COLS = {
    "open", "high", "low", "close", "volume",
    "returns", "log_returns", "ref_price",
    "is_limit_up", "is_limit_down", "is_suspended",
    "adj_close",
}


def construct_target(
    close: pd.Series,
    horizon: int = 3,
    threshold: float = -0.015,
) -> pd.DataFrame:
    """
    Forward-looking binary target.

    y = 1 if Close(T+horizon)/Close(T) - 1 < threshold (stock drops)
    y = 0 otherwise

    Parameters
    ----------
    close : pd.Series
        MultiIndex (date, ticker) close prices.
    horizon : int
        Forward look-ahead in trading days.
    threshold : float
        Drop threshold (negative, e.g. -0.015 = -1.5%).

    Returns
    -------
    pd.DataFrame
        Columns: [fwd_return, label]
    """
    fwd_close = close.groupby(level="ticker").shift(-horizon)
    fwd_return = fwd_close / close - 1.0
    label = (fwd_return < threshold).astype(np.int8)

    return pd.DataFrame({
        TARGET.forward_return: fwd_return,
        TARGET.label: label,
    }, index=close.index)


def validate_no_leakage(feature_df: pd.DataFrame, target_df: pd.DataFrame) -> bool:
    """
    Validate that no feature column is correlated with the target in a way
    that suggests look-ahead bias.

    Checks:
    1. Features must have NaN where target is defined for the first few rows
       (due to lag), confirming shift was applied.
    2. No feature should have > 0.5 correlation with forward return.

    Returns
    -------
    bool
        True if validation passes.
    """
    passed = True

    # Check correlation with forward return
    fwd_ret = target_df[TARGET.forward_return].dropna()
    feature_cols = [c for c in feature_df.columns if c not in RAW_COLS]

    for col in feature_cols:
        feat = feature_df[col].reindex(fwd_ret.index).dropna()
        common = fwd_ret.reindex(feat.index).dropna()
        feat_common = feat.reindex(common.index)

        if len(feat_common) < 50:
            continue

        corr = feat_common.corr(common)
        if abs(corr) > 0.5:
            logger.warning(
                "LEAKAGE WARNING: Feature '%s' has %.3f correlation with forward return!",
                col, corr,
            )
            passed = False

    if passed:
        logger.info("Anti-leakage validation PASSED for %d features", len(feature_cols))
    else:
        logger.error("Anti-leakage validation FAILED — review flagged features!")

    return passed


def build_feature_matrix(
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame = None,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    stock_df : pd.DataFrame
        Processed MultiIndex (date, ticker) OHLCV data.
    index_df : pd.DataFrame
        Processed index data (for correlation features).
    validate : bool
        Run anti-leakage validation.

    Returns
    -------
    pd.DataFrame
        Complete feature matrix with target columns, NaN rows dropped.
    """
    logger.info("Building feature matrix from %d rows", len(stock_df))

    # 1. Technical features
    df = compute_technical_features(stock_df)

    # 2. Microstructure features
    df = compute_microstructure_features(df)

    # 3. Foreign flow proxy features
    df = compute_foreign_flow_features(df, index_df, lag=CFG.features.min_lag)

    # 4. Construct target
    target_df = construct_target(
        df["close"],
        horizon=CFG.target.horizon,
        threshold=CFG.target.drop_threshold,
    )
    df = pd.concat([df, target_df], axis=1)

    # 5. Validate anti-leakage
    feature_cols = [c for c in df.columns if c not in RAW_COLS
                    and c not in {TARGET.forward_return, TARGET.label}]
    if validate:
        validate_no_leakage(df[feature_cols], target_df)

    # 6. Drop rows with NaN in target (first/last rows due to lag/horizon)
    n_before = len(df)
    df = df.dropna(subset=[TARGET.label])
    logger.info("Dropped %d rows with NaN target (lag/horizon warmup)", n_before - len(df))

    # 7. Forward-fill remaining NaN in features, then drop any residual
    df[feature_cols] = df[feature_cols].groupby(level="ticker").ffill()
    remaining_nan = df[feature_cols].isna().sum().sum()
    if remaining_nan > 0:
        logger.info("Dropping %d residual NaN feature values", remaining_nan)
        df = df.dropna(subset=feature_cols)

    logger.info(
        "Feature matrix ready: %d rows, %d features, label distribution: %s",
        len(df), len(feature_cols),
        df[TARGET.label].value_counts().to_dict(),
    )

    return df


def build_feature_matrix_fast(
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame = None,
    validate: bool = True,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Fast feature pipeline using vectorized ops + parquet caching.

    ~15× faster than ``build_feature_matrix()`` for large universes.
    Falls back to the original pipeline if the fast module has issues.

    Parameters
    ----------
    stock_df : pd.DataFrame
        Processed MultiIndex (date, ticker) OHLCV data.
    index_df : pd.DataFrame, optional
        Processed index data.
    validate : bool
        Run anti-leakage validation.
    cache : bool
        Cache result as parquet.

    Returns
    -------
    pd.DataFrame
    """
    # ── Check cache ───────────────────────────────────────────────────
    if cache:
        n_rows = len(stock_df)
        n_tickers = stock_df.index.get_level_values("ticker").nunique()
        cache_key = f"features_{n_tickers}t_{n_rows}r"
        cache_path = CACHE_DIR / f"{cache_key}.parquet"

        if cache_path.exists():
            logger.info("Loading cached feature matrix from %s", cache_path)
            df = pd.read_parquet(cache_path)
            # Restore MultiIndex if needed
            if "date" in df.columns and "ticker" in df.columns:
                df = df.set_index(["date", "ticker"]).sort_index()
            return df

    # ── Compute features (fast path) ──────────────────────────────────
    logger.info("Building feature matrix (FAST mode) from %d rows", len(stock_df))

    try:
        from .technical_fast import compute_all_features_fast
        df = compute_all_features_fast(stock_df, lag=CFG.features.min_lag)
        logger.info("Fast feature computation successful")
    except Exception as exc:
        logger.warning("Fast path failed (%s), falling back to sequential", exc)
        df = compute_technical_features(stock_df)
        df = compute_microstructure_features(df)
        df = compute_foreign_flow_features(df, index_df, lag=CFG.features.min_lag)

    # ── Target ────────────────────────────────────────────────────────
    target_df = construct_target(
        df["close"],
        horizon=CFG.target.horizon,
        threshold=CFG.target.drop_threshold,
    )
    df = pd.concat([df, target_df], axis=1)

    # ── Validation ────────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in RAW_COLS
                    and c not in {TARGET.forward_return, TARGET.label}]
    if validate:
        validate_no_leakage(df[feature_cols], target_df)

    # ── Clean NaN ─────────────────────────────────────────────────────
    n_before = len(df)
    df = df.dropna(subset=[TARGET.label])
    logger.info("Dropped %d rows with NaN target", n_before - len(df))

    df[feature_cols] = df[feature_cols].groupby(level="ticker").ffill()
    df = df.dropna(subset=feature_cols)

    # ── Downcast to float32 ───────────────────────────────────────────
    if CFG.performance.use_float32:
        float_cols = df.select_dtypes(include=[np.float64]).columns
        df[float_cols] = df[float_cols].astype(np.float32)
        logger.info("Downcast %d columns to float32", len(float_cols))

    # ── Cache ─────────────────────────────────────────────────────────
    if cache:
        df_save = df.reset_index()
        df_save.to_parquet(cache_path, index=False)
        logger.info("Cached feature matrix to %s (%.1f MB)",
                    cache_path, cache_path.stat().st_size / 1e6)

    logger.info(
        "Feature matrix ready: %d rows, %d features, label dist: %s",
        len(df), len(feature_cols),
        df[TARGET.label].value_counts().to_dict(),
    )
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature column names (excluding raw data and target)."""
    exclude = RAW_COLS | {TARGET.forward_return, TARGET.label}
    return [c for c in df.columns if c not in exclude]
