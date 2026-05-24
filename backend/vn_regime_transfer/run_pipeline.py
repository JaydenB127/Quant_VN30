# -*- coding: utf-8 -*-
"""
Main pipeline entry point.

Usage:
    python -m vn_regime_transfer.run_pipeline
    python -m vn_regime_transfer.run_pipeline --quick
    python -m vn_regime_transfer.run_pipeline --skip-download
"""
from __future__ import annotations

import sys
import os
# Add backend directory to sys.path so that absolute imports work from both root and backend directory execution contexts
backend_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


def set_global_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def main(args=None):
    """Run the full pipeline."""
    parser = argparse.ArgumentParser(
        description="Regime-Aware Transfer Learning Pipeline for VN Equity"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer folds, subset features")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download, use cached data")
    parser.add_argument("--fast", action="store_true",
                        help="Use vectorized feature computation (15x faster)")
    parser.add_argument("--no-dl", action="store_true",
                        help="Skip LSTM/Transformer baselines (faster)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run full ablation study on model components")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    opts = parser.parse_args(args)

    set_global_seed(opts.seed)

    from .config import CFG, OUTPUT_DIR, REPORT_DIR
    from .data.downloader import download_all
    from .data.processor import process_stock_data, process_index_data
    from .features.builder import build_feature_matrix, build_feature_matrix_fast, get_feature_columns
    from .data.schema import TARGET
    from .regime.hybrid import detect_regime
    from .validation.walk_forward import (
        run_walk_forward, aggregate_fold_results,
    )
    from .validation.ablation import run_ablation_study
    from .validation.metrics import portfolio_metrics, regime_stratified_metrics
    from .validation.threshold import optimize_threshold_fbeta, generate_pr_curve_data
    from .validation.statistical_tests import (
        diebold_mariano_test, diebold_mariano_returns_test,
    )
    from .validation.integrity import validate_metrics_integrity
    from .backtest.executor_vn import simulate_long_avoidance_strategy
    from .reporting.tables import (
        ablation_study_table, regime_performance_table, statistical_summary_table,
        model_comparison_table,
    )
    from .reporting.plots import (
        plot_equity_curve_with_regime, plot_fold_comparison,
        plot_regime_distribution, plot_drawdown,
        plot_pr_curve,
    )
    from .reporting.explainability import generate_shap_summary, generate_regime_shap_summary
    from .reporting.paper_report import generate_latex_report

    start_time = time.time()

    # ════════════════════════════════════════════════════════════════════
    # PHASE 1: DATA INGESTION
    # ════════════════════════════════════════════════════════════════════
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  PHASE 1: DATA INGESTION                           ║")
    logger.info("╚══════════════════════════════════════════════════════╝")

    if opts.skip_download:
        logger.info("Loading cached data...")
    data = download_all(cache=True, skip_download=opts.skip_download)
    stocks_raw = data["stocks"]
    index_raw = data["index"]

    logger.info("Stocks: %d rows, Index: %d rows", len(stocks_raw), len(index_raw))

    # ════════════════════════════════════════════════════════════════════
    # PHASE 2: DATA PROCESSING
    # ════════════════════════════════════════════════════════════════════
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  PHASE 2: DATA PROCESSING                          ║")
    logger.info("╚══════════════════════════════════════════════════════╝")

    stock_df = process_stock_data(stocks_raw)
    index_df = process_index_data(index_raw)

    logger.info("Processed stocks: %d rows", len(stock_df))
    logger.info("Processed index: %d rows", len(index_df))

    # ════════════════════════════════════════════════════════════════════
    # PHASE 3: FEATURE ENGINEERING
    # ════════════════════════════════════════════════════════════════════
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  PHASE 3: FEATURE ENGINEERING                      ║")
    logger.info("╚══════════════════════════════════════════════════════╝")

    if opts.fast:
        feature_df = build_feature_matrix_fast(stock_df, index_df, validate=True)
    else:
        feature_df = build_feature_matrix(stock_df, index_df, validate=True)
    feature_cols = get_feature_columns(feature_df)
    logger.info("Feature matrix: %d rows, %d features", len(feature_df), len(feature_cols))

    # ════════════════════════════════════════════════════════════════════
    # PHASE 4: REGIME DETECTION
    # ════════════════════════════════════════════════════════════════════
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  PHASE 4: REGIME DETECTION                         ║")
    logger.info("╚══════════════════════════════════════════════════════╝")

    regime_df = detect_regime(index_df)

    plot_regime_distribution(regime_df["regime"])

    # ════════════════════════════════════════════════════════════════════
    # PHASE 5: WALK-FORWARD VALIDATION
    # ════════════════════════════════════════════════════════════════════
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  PHASE 5: WALK-FORWARD VALIDATION                  ║")
    logger.info("╚══════════════════════════════════════════════════════╝")

    run_dl = not opts.no_dl
    if opts.quick:
        fold_results = run_walk_forward(
            feature_df, regime_df,
            initial_train_months=36,
            test_months=6, step_months=6,
            run_dl=run_dl,
        )
    else:
        fold_results = run_walk_forward(feature_df, regime_df, run_dl=run_dl)

    if not fold_results:
        logger.error("No fold results! Check data coverage.")
        raise ValueError("No fold results! Check data coverage.")

    fold_metrics_df = aggregate_fold_results(fold_results)

    # ════════════════════════════════════════════════════════════════════
    # PHASE 5.1: ABLATION STUDY (Optional)
    # ════════════════════════════════════════════════════════════════════
    ablation_tbl = None
    if opts.ablation:
        logger.info("╔══════════════════════════════════════════════════════╗")
        logger.info("║  PHASE 5.1: ABLATION STUDY                         ║")
        logger.info("╚══════════════════════════════════════════════════════╝")
        if opts.quick:
            ablation_tbl = run_ablation_study(
                feature_df, regime_df,
                initial_train_months=36,
                test_months=6, step_months=6,
            )
        else:
            ablation_tbl = run_ablation_study(feature_df, regime_df)
            
        ablation_tbl.to_csv(REPORT_DIR / "ablation_components.csv", index=False)
        logger.info("Ablation results saved to %s", REPORT_DIR / "ablation_components.csv")

    # ════════════════════════════════════════════════════════════════════
    # PHASE 5.5: BACKTESTING — connect predictions to execution simulator
    # ════════════════════════════════════════════════════════════════════
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  PHASE 5.5: BACKTESTING (VN Execution Simulation)   ║")
    logger.info("╚══════════════════════════════════════════════════════╝")

    # Build prediction DataFrame from fold results
    # Each fold's test predictions are concatenated chronologically
    pred_frames = []
    for r in fold_results:
        if r.test_dates is not None and r.y_proba_ensemble is not None:
            fold_pred = pd.DataFrame(
                {"pred_proba": r.y_proba_ensemble},
                index=r.test_dates,
            )
            pred_frames.append(fold_pred)

    backtest_result = None
    portfolio_m = {}
    strategy_returns_ens = None
    strategy_returns_static = None

    if pred_frames:
        all_preds = pd.concat(pred_frames)
        # Remove duplicate indices (overlapping folds) — keep last fold's prediction
        all_preds = all_preds[~all_preds.index.duplicated(keep="last")]

        # Run backtest on ensemble predictions
        cfg_bt = CFG.backtest
        backtest_result = simulate_long_avoidance_strategy(
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

        if not backtest_result.empty:
            strategy_returns_ens = backtest_result["strategy_return"]
            portfolio_m = portfolio_metrics(strategy_returns_ens)

            # Validate portfolio metrics integrity
            validate_metrics_integrity(portfolio_m, context="backtest_portfolio")

            logger.info("Backtest portfolio metrics:")
            for k, v in portfolio_m.items():
                logger.info("  %s: %.4f", k, v)

            if regime_df is not None:
                from .regime.hybrid import get_regime_for_dates
                regime_aligned = get_regime_for_dates(regime_df, strategy_returns_ens.index)
                regime_portfolio_m = regime_stratified_metrics(strategy_returns_ens, regime_aligned)
                regime_portfolio_m.to_csv(REPORT_DIR / "regime_portfolio_metrics.csv")
                logger.info("Regime-stratified portfolio metrics saved to %s", REPORT_DIR / "regime_portfolio_metrics.csv")

            # Also run backtest on static model for comparison
            static_pred_frames = []
            for r in fold_results:
                if r.test_dates is not None and r.y_proba_static is not None:
                    static_pred_frames.append(pd.DataFrame(
                        {"pred_proba": r.y_proba_static},
                        index=r.test_dates,
                    ))
            if static_pred_frames:
                static_preds = pd.concat(static_pred_frames)
                static_preds = static_preds[~static_preds.index.duplicated(keep="last")]
                static_bt = simulate_long_avoidance_strategy(
                    predictions=static_preds,
                    price_df=feature_df,
                    regime_df=regime_df,
                    entry_delay=cfg_bt.entry_delay_days,
                    hold_period=cfg_bt.hold_period_days,
                    commission=cfg_bt.commission_rate,
                    slippage=cfg_bt.slippage_rate,
                )
                if not static_bt.empty:
                    strategy_returns_static = static_bt["strategy_return"]
        else:
            logger.warning("Backtest produced no results")
    else:
        logger.warning("No fold predictions available for backtesting")

    # ════════════════════════════════════════════════════════════════════
    # PHASE 6: STATISTICAL TESTING
    # ════════════════════════════════════════════════════════════════════
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  PHASE 6: STATISTICAL TESTING                      ║")
    logger.info("╚══════════════════════════════════════════════════════╝")

    # Concatenate predictions across folds
    all_y_true = np.concatenate([r.y_true for r in fold_results])
    all_y_proba_ens = np.concatenate([r.y_proba_ensemble for r in fold_results])
    all_y_proba_static = np.concatenate([r.y_proba_static for r in fold_results])

    # Diebold-Mariano: ensemble vs all baselines
    dm_results = {
        "ensemble_vs_static": diebold_mariano_test(
            all_y_true, all_y_proba_ens, all_y_proba_static,
            loss="brier",
        ),
    }

    # Ensemble vs XGBoost
    all_y_proba_xgb = np.concatenate([r.y_proba_xgb for r in fold_results])
    dm_results["ensemble_vs_xgb"] = diebold_mariano_test(
        all_y_true, all_y_proba_ens, all_y_proba_xgb, loss="brier",
    )

    # Ensemble vs LSTM (if available)
    all_y_proba_lstm = np.concatenate([r.y_proba_lstm for r in fold_results])
    if not (all_y_proba_lstm == 0.5).all():
        dm_results["ensemble_vs_lstm"] = diebold_mariano_test(
            all_y_true, all_y_proba_ens, all_y_proba_lstm, loss="brier",
        )

    # Ensemble vs Transformer (if available)
    all_y_proba_tf = np.concatenate([r.y_proba_transformer for r in fold_results])
    if not (all_y_proba_tf == 0.5).all():
        dm_results["ensemble_vs_transformer"] = diebold_mariano_test(
            all_y_true, all_y_proba_ens, all_y_proba_tf, loss="brier",
        )

    for name, dm in dm_results.items():
        logger.info("DM test [%s] (%s): stat=%.4f, p=%.4f",
                    dm.get("loss_used", "brier"), name,
                    dm["dm_statistic"], dm["p_value"])

    # Bootstrap CI for ensemble Sharpe
    from .validation.statistical_tests import bootstrap_confidence_interval
    bootstrap_results = {}
    if len(all_y_proba_ens) > 0:
        bootstrap_results["ensemble_auc"] = bootstrap_confidence_interval(
            all_y_proba_ens, statistic_fn=np.mean,
            n_bootstrap=CFG.validation.n_bootstrap,
        )

    # ── Metrics integrity validation ──────────────────────────────
    # Detect any hardcoded or fabricated metrics before saving
    flat_metrics = {}
    for name, dm in dm_results.items():
        flat_metrics[f"dm_pvalue_{name}"] = dm.get("p_value")
    for name, bs in bootstrap_results.items():
        flat_metrics[f"bootstrap_{name}"] = bs.get("point_estimate")
    integrity_report = validate_metrics_integrity(flat_metrics, context="pipeline_output")
    if not integrity_report.passed:
        logger.error("INTEGRITY CHECK FAILED — review output before publishing!\n%s",
                     integrity_report.summary())

    # DM test on strategy returns (if backtest ran)
    if strategy_returns_ens is not None and strategy_returns_static is not None:
        dm_returns = diebold_mariano_returns_test(
            strategy_returns_ens.values,
            strategy_returns_static.values,
        )
        dm_results["strategy_returns_ens_vs_static"] = dm_returns
        logger.info(
            "DM test [strategy returns] ensemble vs static: "
            "stat=%.4f, p=%.4f, sharpe_diff=%.4f",
            dm_returns["dm_statistic"],
            dm_returns["p_value"],
            dm_returns.get("sharpe_diff", 0),
        )

    # ════════════════════════════════════════════════════════════════════
    # PHASE 7: REPORTING
    # ════════════════════════════════════════════════════════════════════
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  PHASE 7: REPORTING                                ║")
    logger.info("╚══════════════════════════════════════════════════════╝")

    # Tables
    # This existing table compares models from walk-forward
    models_tbl = ablation_study_table(fold_results) 
    regime_tbl = regime_performance_table(fold_results)
    stat_tbl = statistical_summary_table(dm_results, bootstrap_results)
    comparison_tbl = model_comparison_table(fold_results)

    # Save tables
    models_tbl.to_csv(REPORT_DIR / "models_summary.csv")
    if ablation_tbl is not None:
        ablation_tbl.to_csv(REPORT_DIR / "ablation_components_full.csv")
    regime_tbl.to_csv(REPORT_DIR / "regime_performance.csv")
    stat_tbl.to_csv(REPORT_DIR / "statistical_tests.csv")
    comparison_tbl.to_csv(REPORT_DIR / "model_comparison.csv")
    fold_metrics_df.to_csv(REPORT_DIR / "fold_metrics.csv", index=False)

    # Plots
    plot_fold_comparison(fold_metrics_df, metric="auc_roc")
    plot_fold_comparison(fold_metrics_df, metric="f1_score")

    # PR Curve and Threshold Optimization
    if len(all_y_proba_ens) > 0 and len(all_y_true) > 0:
        pr_df = generate_pr_curve_data(all_y_true, all_y_proba_ens)
        pr_df.to_csv(REPORT_DIR / "pr_curve_data.csv", index=False)
        plot_pr_curve(pr_df)
        
        opt_thresh, opt_metrics = optimize_threshold_fbeta(all_y_true, all_y_proba_ens, beta=0.5)
        logger.info("Optimized Decision Threshold (F0.5): %.4f (Precision: %.4f, Recall: %.4f)",
                    opt_thresh, opt_metrics["precision"], opt_metrics["recall"])

    # SHAP Explainability
    logger.info("Generating SHAP explainability plots...")
    try:
        from .model.base_lgb import BaseLGBModel
        from .features.builder import get_feature_columns
        from .data.schema import TARGET
        
        # Train a quick model on the last 2 years for SHAP analysis
        recent_dates = feature_df.index.get_level_values("date")
        cutoff = recent_dates.max() - pd.DateOffset(years=2)
        shap_df = feature_df[recent_dates >= cutoff].copy()
        
        fcols = get_feature_columns(shap_df)
        X_shap = shap_df[fcols]
        y_shap = shap_df[TARGET.label]
        
        shap_model = BaseLGBModel()
        shap_model.fit(X_shap, y_shap)
        
        if shap_model.model is not None:
            generate_shap_summary(shap_model.model, X_shap)
            
            if regime_df is not None:
                from .regime.hybrid import get_regime_for_dates
                shap_regimes = get_regime_for_dates(regime_df, X_shap.index.get_level_values("date"))
                generate_regime_shap_summary(shap_model.model, X_shap, shap_regimes)
    except Exception as e:
        logger.warning("Failed to generate SHAP plots: %s", e)

    # LaTeX report
    generate_latex_report(
        models_tbl, regime_tbl, stat_tbl,
        portfolio_m, fold_metrics_df,
    )

    # Save backtest results if available
    if backtest_result is not None and not backtest_result.empty:
        backtest_result.to_csv(REPORT_DIR / "backtest_daily_returns.csv")
        logger.info("Backtest daily returns saved to %s",
                    REPORT_DIR / "backtest_daily_returns.csv")

        # Equity curve + drawdown plots
        plot_drawdown(strategy_returns_ens)
        if regime_df is not None:
            regime_aligned = get_regime_for_dates(
                regime_df, strategy_returns_ens.index,
            )
            bench_rets = backtest_result["benchmark_return"]
            plot_equity_curve_with_regime(
                strategy_returns_ens, bench_rets, regime_aligned,
            )

    # ════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  PIPELINE COMPLETE                                  ║")
    logger.info("╚══════════════════════════════════════════════════════╝")
    logger.info("Total time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)
    logger.info("Results saved to: %s", REPORT_DIR)
    logger.info("")
    if ablation_tbl is not None:
        logger.info("═══ ABLATION COMPONENTS ═══")
        logger.info("\n%s", ablation_tbl.to_string())
        logger.info("")
    logger.info("═══ MODEL COMPARISON ═══")
    logger.info("\n%s", models_tbl.to_string())
    logger.info("")
    logger.info("═══ REGIME PERFORMANCE ═══")
    logger.info("\n%s", regime_tbl.to_string())
    logger.info("")
    logger.info("═══ STATISTICAL TESTS ═══")
    logger.info("\n%s", stat_tbl.to_string())

    print("\n[SUCCESS] Pipeline completed successfully!")
    print(f"   Reports: {REPORT_DIR}")
    print(f"   Time: {elapsed:.1f}s")

    return {
        "fold_metrics": fold_metrics_df,
        "portfolio_metrics": portfolio_m,
        "ablation_table": ablation_tbl,
        "dm_results": dm_results,
        "bootstrap_results": bootstrap_results,
        "integrity_passed": integrity_report.passed if 'integrity_report' in locals() else True,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    main()
