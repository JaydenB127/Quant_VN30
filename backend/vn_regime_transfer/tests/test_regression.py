# -*- coding: utf-8 -*-
"""Regression test for the full pipeline execution."""
import os
import pytest
import pandas as pd
from vn_regime_transfer.run_pipeline import main


def test_pipeline_execution_regression():
    """
    Run the full pipeline in fast, quick, and cached mode.
    Verify that it executes without errors and returns structured results.
    """
    # Execute the pipeline with parameters to run quickly and offline
    results = main(args=[
        "--quick",
        "--skip-download",
        "--fast",
        "--no-dl",
        "--seed", "42"
    ])

    # ── Verify return structure ──
    assert isinstance(results, dict)
    assert "fold_metrics" in results
    assert "portfolio_metrics" in results
    assert "dm_results" in results
    assert "bootstrap_results" in results
    assert "integrity_passed" in results

    # ── Verify dataframes ──
    fold_metrics = results["fold_metrics"]
    assert isinstance(fold_metrics, pd.DataFrame)
    assert not fold_metrics.empty
    assert "fold" in fold_metrics.columns
    assert "auc_roc" in fold_metrics.columns

    # ── Verify integrity check ──
    assert results["integrity_passed"] is True

    # ── Verify statistical test outputs ──
    dm_results = results["dm_results"]
    assert "ensemble_vs_static" in dm_results
    assert "ensemble_vs_xgb" in dm_results

    # ── Verify portfolio metrics ──
    p_metrics = results["portfolio_metrics"]
    if p_metrics:  # Only if backtest produced results
        assert "sharpe" in p_metrics
        assert "max_drawdown" in p_metrics
        assert p_metrics["max_drawdown"] <= 0


def test_async_pipeline_execution_regression():
    """
    Run the FinanceForecastingPipeline using the new async execution engine
    and BufferedTrackingService. Verify that all steps execute successfully.
    """
    import asyncio
    import uuid
    from plugins.finance.pipeline import FinanceForecastingPipeline
    from ets.core.tracking.buffer import BufferedTrackingService

    async def _run():
        pipeline = FinanceForecastingPipeline()
        tracker = BufferedTrackingService(buffer_size=10, flush_interval_seconds=1.0)
        run_id = uuid.uuid4()
        try:
            results = await pipeline.run(
                run_id=run_id,
                config={
                    "skip_download": True,
                    "fast": True,
                    "quick": True,
                    "run_dl": False,
                },
                tracker=tracker,
            )
            return results
        finally:
            await tracker.close()

    results = asyncio.run(_run())

    # ── Verify results structure ──
    assert isinstance(results, dict)
    assert "fold_metrics_df" in results
    assert "portfolio_m" in results
    assert "dm_results" in results
    assert "bootstrap_results" in results

    # Verify fold metrics
    fold_metrics = results["fold_metrics_df"]
    assert isinstance(fold_metrics, pd.DataFrame)
    assert not fold_metrics.empty

