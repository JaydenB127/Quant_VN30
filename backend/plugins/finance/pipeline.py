# -*- coding: utf-8 -*-
"""
Finance forecasting pipeline plugin wrapping validated quantitative research logic.
Extends the generic BasePipeline with step-based walk-forward execution.
"""
from __future__ import annotations

import logging
from typing import List
import numpy as np
import pandas as pd

from ets.core.pipeline.base import BasePipeline, PipelineStep, PipelineContext
from vn_regime_transfer.config import CFG, REPORT_DIR
from vn_regime_transfer.data.downloader import download_all
from vn_regime_transfer.data.processor import process_stock_data, process_index_data
from vn_regime_transfer.features.builder import build_feature_matrix, build_feature_matrix_fast, get_feature_columns
from vn_regime_transfer.data.schema import TARGET
from vn_regime_transfer.regime.hybrid import detect_regime, get_regime_for_dates
from vn_regime_transfer.validation.walk_forward import run_walk_forward, aggregate_fold_results
from vn_regime_transfer.validation.metrics import portfolio_metrics, regime_stratified_metrics
from vn_regime_transfer.validation.threshold import optimize_threshold_fbeta, generate_pr_curve_data
from vn_regime_transfer.validation.statistical_tests import diebold_mariano_test, diebold_mariano_returns_test
from vn_regime_transfer.validation.integrity import validate_metrics_integrity
from vn_regime_transfer.backtest.executor_vn import simulate_long_avoidance_strategy
from vn_regime_transfer.reporting.tables import (
    ablation_study_table, regime_performance_table, statistical_summary_table,
    model_comparison_table
)
from vn_regime_transfer.reporting.plots import (
    plot_equity_curve_with_regime, plot_fold_comparison,
    plot_regime_distribution, plot_drawdown, plot_pr_curve
)
from vn_regime_transfer.reporting.explainability import generate_shap_summary, generate_regime_shap_summary
from vn_regime_transfer.reporting.paper_report import generate_latex_report

logger = logging.getLogger(__name__)


class FinanceDataLoadStep(PipelineStep):
    """Phase 1: Downloads VN30 universe and index data."""
    def __init__(self):
        super().__init__("data_ingestion")

    async def execute(self, context: PipelineContext) -> PipelineContext:
        skip_download = context.config.get("skip_download", False)
        dataset_name = context.config.get("dataset_name", "VN30 (default)")
        logger.info("Finance Ingestion | dataset=%s | skip_download=%s", dataset_name, skip_download)

        # Log the dataset being used as a parameter for traceability
        tracker = context.tracker
        if tracker and hasattr(tracker, "log_parameter") and context.run_id:
            await tracker.log_parameter(context.run_id, "dataset_used", dataset_name)

        import asyncio
        import os

        dataset_csv_path = context.config.get("dataset_csv_path")
        if dataset_csv_path and os.path.exists(dataset_csv_path):
            logger.info("Finance Ingestion | Loading custom dataset from %s", dataset_csv_path)
            try:
                import pandas as pd
                df = pd.read_csv(dataset_csv_path)
                
                # Standardize column mapping case-insensitively
                col_map = {}
                ticker_col = None
                for col in df.columns:
                    cl = col.lower().strip()
                    if cl in ("time", "date", "trading_date", "tradingdate"):
                        col_map[col] = "date"
                    elif cl in ("symbol", "ticker", "asset"):
                        ticker_col = col
                    elif cl in ("open",):
                        col_map[col] = "open"
                    elif cl in ("high",):
                        col_map[col] = "high"
                    elif cl in ("low",):
                        col_map[col] = "low"
                    elif cl in ("close",):
                        col_map[col] = "close"
                    elif cl in ("volume",):
                        col_map[col] = "volume"
                
                df = df.rename(columns=col_map)
                df["date"] = pd.to_datetime(df["date"])
                
                # If ticker column exists, rename it to ticker. Otherwise assign a default.
                if ticker_col:
                    df = df.rename(columns={ticker_col: "ticker"})
                else:
                    # Deduce ticker from filename or default to "BTC" / "ASSET"
                    default_ticker = "BTC" if "btc" in dataset_name.lower() else "ASSET"
                    df["ticker"] = default_ticker
                
                # Keep only required columns
                required_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
                # If some columns are missing (e.g. volume is missing or named differently), fill them
                for col in ["open", "high", "low", "close"]:
                    if col not in df.columns:
                        raise ValueError(f"Custom dataset is missing required column: {col}")
                if "volume" not in df.columns:
                    df["volume"] = 0.0
                
                stocks_df = df[required_cols].copy()
                stocks_df = stocks_df.sort_values(["date", "ticker"]).reset_index(drop=True)
                
                # Create index_raw as equally-weighted average of all stocks on each date
                # Group by date and take the mean of prices, and sum of volumes
                index_df = stocks_df.groupby("date").agg({
                    "open": "mean",
                    "high": "mean",
                    "low": "mean",
                    "close": "mean",
                    "volume": "sum"
                }).reset_index()
                
                context.results["stocks_raw"] = stocks_df
                context.results["index_raw"] = index_df
                
                logger.info("Successfully loaded custom dataset: %d stock rows, %d index rows, %d unique tickers", 
                            len(stocks_df), len(index_df), stocks_df["ticker"].nunique())
                return context
            except Exception as e:
                logger.exception("Failed to load custom dataset from %s, falling back to default VN30: %s", dataset_csv_path, e)

        data = await asyncio.to_thread(download_all, cache=True, skip_download=skip_download)
        context.results["stocks_raw"] = data["stocks"]
        context.results["index_raw"] = data["index"]
        return context


class FinanceDataProcessStep(PipelineStep):
    """Phase 2: Cleans raw data and computes reference prices / returns."""
    def __init__(self):
        super().__init__("data_processing")

    async def execute(self, context: PipelineContext) -> PipelineContext:
        stocks_raw = context.results["stocks_raw"]
        index_raw = context.results["index_raw"]
        
        import asyncio
        context.results["stock_df"] = await asyncio.to_thread(process_stock_data, stocks_raw)
        context.results["index_df"] = await asyncio.to_thread(process_index_data, index_raw)
        return context


class FinanceFeatureStep(PipelineStep):
    """Phase 3: Computes technical indicators, microstructure and volume features."""
    def __init__(self):
        super().__init__("feature_engineering")

    async def execute(self, context: PipelineContext) -> PipelineContext:
        stock_df = context.results["stock_df"]
        index_df = context.results["index_df"]
        fast = context.config.get("fast", True)
        
        import asyncio
        if fast:
            feature_df = await asyncio.to_thread(build_feature_matrix_fast, stock_df, index_df, validate=True)
        else:
            feature_df = await asyncio.to_thread(build_feature_matrix, stock_df, index_df, validate=True)
            
        context.results["feature_df"] = feature_df
        context.results["feature_cols"] = get_feature_columns(feature_df)
        return context


class FinanceRegimeDetectionStep(PipelineStep):
    """Phase 4: Performs walk-forward HMM, PELT and rule-based regime modeling."""
    def __init__(self):
        super().__init__("regime_detection")

    async def execute(self, context: PipelineContext) -> PipelineContext:
        index_df = context.results["index_df"]
        
        import asyncio
        regime_df = await asyncio.to_thread(detect_regime, index_df)
        
        # Save plot
        await asyncio.to_thread(plot_regime_distribution, regime_df["regime"])
        
        context.results["regime_df"] = regime_df
        return context


class FinanceWalkForwardStep(PipelineStep):
    """Phase 5: Expanding-window walk-forward 7-model cross validation."""
    def __init__(self):
        super().__init__("walk_forward_validation")

    async def execute(self, context: PipelineContext) -> PipelineContext:
        feature_df = context.results["feature_df"]
        regime_df = context.results["regime_df"]
        quick = context.config.get("quick", True)
        run_dl = context.config.get("run_dl", False)

        import asyncio
        from vn_regime_transfer.config import CFG

        # Apply hyperparameter overrides if provided
        if "learning_rate" in context.config:
            CFG.model.base_params["learning_rate"] = float(context.config["learning_rate"])
        if "max_depth" in context.config:
            CFG.model.base_params["max_depth"] = int(context.config["max_depth"])

        if quick:
            fold_results = await asyncio.to_thread(
                run_walk_forward,
                feature_df, regime_df,
                initial_train_months=36,
                test_months=6, step_months=6,
                run_dl=run_dl,
            )
        else:
            fold_results = await asyncio.to_thread(run_walk_forward, feature_df, regime_df, run_dl=run_dl)

        if not fold_results:
            raise ValueError("No walk-forward folds produced!")

        context.results["fold_results"] = fold_results
        context.results["fold_metrics_df"] = await asyncio.to_thread(aggregate_fold_results, fold_results)

        # Log metrics fold-by-fold to tracker
        tracker = context.tracker
        if tracker and hasattr(tracker, "log_metric") and context.run_id:
            for r in fold_results:
                for model in ["ensemble", "transfer", "base", "static", "xgb"]:
                    metrics_dict = getattr(r, f"metrics_{model}", {})
                    if metrics_dict:
                        for k, v in metrics_dict.items():
                            if v is not None and not np.isnan(v):
                                await tracker.log_metric(context.run_id, f"{model}_{k}", float(v), step=r.fold_id)

        return context


class FinanceBacktestStep(PipelineStep):
    """Phase 5.5: Simulated T+1 trading execution with VN limits and cost model."""
    def __init__(self):
        super().__init__("backtesting")

    async def execute(self, context: PipelineContext) -> PipelineContext:
        fold_results = context.results["fold_results"]
        feature_df = context.results["feature_df"]
        regime_df = context.results["regime_df"]

        pred_frames = []
        for r in fold_results:
            if r.test_dates is not None and r.y_proba_ensemble is not None:
                pred_frames.append(pd.DataFrame(
                    {"pred_proba": r.y_proba_ensemble},
                    index=r.test_dates,
                ))

        if not pred_frames:
            logger.warning("No predictions available for backtest")
            context.results["backtest_result"] = pd.DataFrame()
            context.results["portfolio_m"] = {}
            return context

        all_preds = pd.concat(pred_frames)
        all_preds = all_preds[~all_preds.index.duplicated(keep="last")]

        cfg_bt = CFG.backtest
        import asyncio
        backtest_result = await asyncio.to_thread(
            simulate_long_avoidance_strategy,
            predictions=all_preds,
            price_df=feature_df,
            regime_df=regime_df,
            entry_delay=cfg_bt.entry_delay_days,
            hold_period=cfg_bt.hold_period_days,
            commission=cfg_bt.commission_rate,
            slippage=cfg_bt.slippage_rate,
            cb_threshold=cfg_bt.circuit_breaker_pct,
            initial_capital=cfg_bt.initial_capital,
        )

        portfolio_m = {}
        if not backtest_result.empty:
            strategy_returns = backtest_result["strategy_return"]
            portfolio_m = await asyncio.to_thread(portfolio_metrics, strategy_returns)
            validate_metrics_integrity(portfolio_m, context="backtest")

            # Stratified
            regime_aligned = get_regime_for_dates(regime_df, strategy_returns.index)
            regime_portfolio_m = await asyncio.to_thread(regime_stratified_metrics, strategy_returns, regime_aligned)
            regime_portfolio_m.to_csv(REPORT_DIR / "regime_portfolio_metrics.csv")

            # Comparative static
            static_pred_frames = []
            for r in fold_results:
                if r.test_dates is not None and r.y_proba_static is not None:
                    static_pred_frames.append(pd.DataFrame(
                        {"pred_proba": r.y_proba_static}, index=r.test_dates
                    ))
            if static_pred_frames:
                static_preds = pd.concat(static_pred_frames)
                static_preds = static_preds[~static_preds.index.duplicated(keep="last")]
                static_bt = await asyncio.to_thread(
                    simulate_long_avoidance_strategy,
                    predictions=static_preds, price_df=feature_df, regime_df=regime_df,
                    entry_delay=cfg_bt.entry_delay_days, hold_period=cfg_bt.hold_period_days,
                    commission=cfg_bt.commission_rate, slippage=cfg_bt.slippage_rate
                )
                if not static_bt.empty:
                    context.results["strategy_returns_static"] = static_bt["strategy_return"]

        context.results["backtest_result"] = backtest_result
        context.results["portfolio_m"] = portfolio_m

        # Log portfolio metrics to tracker
        tracker = context.tracker
        if tracker and hasattr(tracker, "log_metric") and context.run_id and portfolio_m:
            for k, v in portfolio_m.items():
                if v is not None and not np.isnan(v):
                    await tracker.log_metric(context.run_id, f"portfolio_{k}", float(v), step=0)

        return context


class FinanceStatisticalTestStep(PipelineStep):
    """Phase 6: Performs DM significance checks and metrics integrity auditing."""
    def __init__(self):
        super().__init__("statistical_testing")

    async def execute(self, context: PipelineContext) -> PipelineContext:
        fold_results = context.results["fold_results"]
        
        all_y_true = np.concatenate([r.y_true for r in fold_results])
        all_y_proba_ens = np.concatenate([r.y_proba_ensemble for r in fold_results])
        all_y_proba_static = np.concatenate([r.y_proba_static for r in fold_results])
        all_y_proba_xgb = np.concatenate([r.y_proba_xgb for r in fold_results])

        import asyncio
        dm_results = {
            "ensemble_vs_static": await asyncio.to_thread(diebold_mariano_test, all_y_true, all_y_proba_ens, all_y_proba_static),
            "ensemble_vs_xgb": await asyncio.to_thread(diebold_mariano_test, all_y_true, all_y_proba_ens, all_y_proba_xgb),
        }

        bootstrap_results = {
            "ensemble_auc": {
                "point_estimate": float(np.mean(all_y_proba_ens)) if len(all_y_proba_ens) > 0 else 0.5
            }
        }

        # Integrity Validation
        flat_metrics = {f"dm_pvalue_{k}": v["p_value"] for k, v in dm_results.items()}
        validate_metrics_integrity(flat_metrics, context="pipeline")

        # Returns DM
        backtest_result = context.results["backtest_result"]
        strategy_returns_static = context.results.get("strategy_returns_static")
        if not backtest_result.empty and strategy_returns_static is not None:
            strategy_returns_ens = backtest_result["strategy_return"]
            dm_results["returns_ens_vs_static"] = await asyncio.to_thread(
                diebold_mariano_returns_test,
                strategy_returns_ens.values, strategy_returns_static.values
            )

        context.results["dm_results"] = dm_results
        context.results["bootstrap_results"] = bootstrap_results

        # Log DM test p-values to tracker
        tracker = context.tracker
        if tracker and hasattr(tracker, "log_metric") and context.run_id and dm_results:
            for k, v in dm_results.items():
                if "p_value" in v and v["p_value"] is not None and not np.isnan(v["p_value"]):
                    await tracker.log_metric(context.run_id, f"dm_pvalue_{k}", float(v["p_value"]), step=0)

        return context


class FinanceReportStep(PipelineStep):
    """Phase 7: Generates tables, Plots, explainability SHAPs, and LaTeX papers."""
    def __init__(self):
        super().__init__("reporting")

    async def execute(self, context: PipelineContext) -> PipelineContext:
        fold_results = context.results["fold_results"]
        fold_metrics_df = context.results["fold_metrics_df"]
        dm_results = context.results["dm_results"]
        bootstrap_results = context.results["bootstrap_results"]
        portfolio_m = context.results["portfolio_m"]
        backtest_result = context.results["backtest_result"]
        regime_df = context.results["regime_df"]

        # Tables
        import asyncio
        import os
        import shutil
        models_tbl = await asyncio.to_thread(ablation_study_table, fold_results)
        regime_tbl = await asyncio.to_thread(regime_performance_table, fold_results)
        stat_tbl = await asyncio.to_thread(statistical_summary_table, dm_results, bootstrap_results)

        models_tbl.to_csv(REPORT_DIR / "models_summary.csv")
        regime_tbl.to_csv(REPORT_DIR / "regime_performance.csv")
        stat_tbl.to_csv(REPORT_DIR / "statistical_tests.csv")
        fold_metrics_df.to_csv(REPORT_DIR / "fold_metrics.csv", index=False)

        # Plotting
        await asyncio.to_thread(plot_fold_comparison, fold_metrics_df, metric="auc_roc")
        await asyncio.to_thread(plot_fold_comparison, fold_metrics_df, metric="f1_score")

        if not backtest_result.empty:
            strategy_returns = backtest_result["strategy_return"]
            await asyncio.to_thread(plot_drawdown, strategy_returns)

            regime_aligned = get_regime_for_dates(regime_df, strategy_returns.index)
            await asyncio.to_thread(
                plot_equity_curve_with_regime,
                strategy_returns, backtest_result["benchmark_return"], regime_aligned
            )

        # LaTeX Academic Paper
        await asyncio.to_thread(generate_latex_report, models_tbl, regime_tbl, stat_tbl, portfolio_m, fold_metrics_df)

        # Save summary keys
        context.results["reporting"] = {
            "summary_table": models_tbl.to_dict(),
            "regime_table": regime_tbl.to_dict(),
            "stat_table": stat_tbl.to_dict(),
        }

        # ── Copy all generated files into a per-run directory ──────────────────
        # This ensures each run has its OWN artifact set that won't be overwritten
        # by subsequent runs. The shared REPORT_DIR is the generation target;
        # we snapshot it to REPORT_DIR/runs/<run_id_short>/ per run.
        run_id_short = (context.run_id or "unknown")[:8]
        run_report_dir = REPORT_DIR / "runs" / run_id_short
        run_report_dir.mkdir(parents=True, exist_ok=True)

        tracker = context.tracker
        if tracker and hasattr(tracker, "log_artifact") and context.run_id:
            try:
                for filename in sorted(os.listdir(REPORT_DIR)):
                    src = os.path.join(REPORT_DIR, filename)
                    if not os.path.isfile(src):
                        continue
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in (".png", ".csv", ".pdf", ".txt", ".tex"):
                        continue

                    # Copy to per-run snapshot directory
                    dst = run_report_dir / filename
                    shutil.copy2(src, dst)

                    content_type = (
                        "image/png" if ext == ".png"
                        else "text/csv" if ext == ".csv"
                        else "application/pdf" if ext == ".pdf"
                        else "text/plain"
                    )
                    artifact_type = (
                        "plot" if ext == ".png"
                        else "report" if ext in (".pdf", ".tex")
                        else "dataset" if ext == ".csv"
                        else "other"
                    )
                    await tracker.log_artifact(
                        run_id=context.run_id,
                        name=filename,
                        storage_backend="local",
                        storage_key=str(dst),
                        size_bytes=os.path.getsize(str(dst)),
                        content_type=content_type,
                        artifact_type=artifact_type,
                        metadata={"path": str(dst), "run_dir": str(run_report_dir)},
                    )
                logger.info("Registered %d artifacts for run %s in %s", len(list(run_report_dir.iterdir())), context.run_id[:8], run_report_dir)
            except Exception as exc:
                logger.error("Failed to register artifacts in reporting step: %s", exc)

        return context


class FinanceForecastingPipeline(BasePipeline):
    """
    Modular 8-step quantitative prediction pipeline representing
    the original regime-aware quant system inside the new ETS architecture.
    """

    def __init__(self):
        super().__init__("finance_forecasting")

    def get_steps(self) -> List[PipelineStep]:
        return [
            FinanceDataLoadStep(),
            FinanceDataProcessStep(),
            FinanceFeatureStep(),
            FinanceRegimeDetectionStep(),
            FinanceWalkForwardStep(),
            FinanceBacktestStep(),
            FinanceStatisticalTestStep(),
            FinanceReportStep(),
        ]
