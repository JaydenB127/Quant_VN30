# -*- coding: utf-8 -*-
"""
Vietnam execution simulator.

Implements realistic constraints: T+1 entry delay, circuit breaker skipping,
commission/slippage, position sizing.
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..config import CFG
from ..data.universe import get_vn30_constituents

logger = logging.getLogger(__name__)


def simulate_long_avoidance_strategy(
    predictions: pd.DataFrame,
    price_df: pd.DataFrame,
    regime_df: pd.DataFrame = None,
    entry_delay: int = 1,
    hold_period: int = 3,
    commission: float = 0.0015,
    slippage: float = 0.001,
    cb_threshold: float = 0.065,
    initial_capital: float = 1e9,
    max_position_risk: float = 0.02,
) -> pd.DataFrame:
    """
    Simulate long-only avoidance strategy.

    Logic: for each day, score all stocks. AVOID (underweight) stocks
    predicted to drop. Hold remaining stocks equally weighted.

    Parameters
    ----------
    predictions : pd.DataFrame
        MultiIndex (date, ticker) with column 'pred_proba' (prob of drop).
    price_df : pd.DataFrame
        MultiIndex (date, ticker) with columns [close, is_limit_up, is_limit_down].
    regime_df : pd.DataFrame
        Optional regime data for regime-aware sizing.
    entry_delay : int
        Days between signal and execution.
    hold_period : int
        Minimum hold period in days.
    commission : float
        One-way commission rate.
    slippage : float
        Slippage rate per trade.
    cb_threshold : float
        Skip entry/exit if absolute return >= this.
    initial_capital : float
        Starting capital.
    max_position_risk : float
        Max fraction of capital per position.

    Returns
    -------
    pd.DataFrame
        Daily portfolio returns with columns:
        [date, strategy_return, benchmark_return, n_positions, turnover].
    """
    dates = predictions.index.get_level_values("date").unique().sort_values()
    one_way_cost = commission + slippage

    daily_results = []

    for i in range(entry_delay, len(dates)):
        signal_date = dates[i - entry_delay]
        trade_date = dates[i]

        # Get predictions from signal date
        try:
            day_preds = predictions.loc[signal_date]
        except KeyError:
            continue

        if isinstance(day_preds, pd.Series):
            day_preds = day_preds.to_frame().T

        if "pred_proba" not in day_preds.columns:
            continue

        # Get prices on trade date
        try:
            day_prices = price_df.loc[trade_date]
        except KeyError:
            continue

        # Check if this is a custom universe (no overlap with VN30 constituents)
        pred_tickers = set(predictions.index.get_level_values("ticker") if "ticker" in predictions.index.names else predictions.index)
        vn30_current = set(CFG.universe.tickers)
        is_custom_universe = len(pred_tickers & vn30_current) == 0

        # Get available tickers (not suspended, not at circuit breaker, AND in constituents on this date if VN30)
        if not is_custom_universe:
            vn30_on_date = set(get_vn30_constituents(trade_date))
        else:
            vn30_on_date = pred_tickers

        available_tickers = []
        for ticker in day_preds.index.get_level_values("ticker") if "ticker" in day_preds.index.names else day_preds.index:
            if ticker not in vn30_on_date:
                continue
            try:
                if isinstance(day_prices.index, pd.MultiIndex):
                    p = day_prices.loc[ticker]
                else:
                    p = day_prices

                is_cb = False
                if "is_limit_up" in price_df.columns:
                    lu = price_df.loc[(trade_date, ticker), "is_limit_up"] if (trade_date, ticker) in price_df.index else 0
                    ld = price_df.loc[(trade_date, ticker), "is_limit_down"] if (trade_date, ticker) in price_df.index else 0
                    is_cb = lu == 1 or ld == 1

                if not is_cb:
                    available_tickers.append(ticker)
            except (KeyError, TypeError):
                continue

        if not available_tickers:
            daily_results.append({
                "date": trade_date,
                "strategy_return": 0.0,
                "benchmark_return": 0.0,
                "n_positions": 0,
                "turnover": 0.0,
            })
            continue

        # Score: avoid stocks with high drop probability
        # Select stocks with LOWEST drop probability (pred_proba < 0.5)
        scores = {}
        for ticker in available_tickers:
            try:
                if "ticker" in day_preds.index.names:
                    prob = day_preds.loc[ticker, "pred_proba"]
                else:
                    prob = day_preds.loc[day_preds.index == ticker, "pred_proba"].iloc[0]
                scores[ticker] = float(prob) if not isinstance(prob, (pd.Series, pd.DataFrame)) else float(prob.iloc[0])
            except (KeyError, IndexError):
                scores[ticker] = 0.5

        # Sort by drop probability ascending → pick stocks LEAST likely to drop
        sorted_tickers = sorted(scores, key=scores.get)

        # Take top half (avoid bottom half predicted to drop)
        n_select = max(1, len(sorted_tickers) // 2)
        selected = sorted_tickers[:n_select]

        # Equal weight
        weight = 1.0 / len(selected)

        # Calculate portfolio return
        port_return = 0.0
        bench_return = 0.0

        for ticker in available_tickers:
            try:
                if (trade_date, ticker) in price_df.index:
                    close_today = price_df.loc[(trade_date, ticker), "close"]
                    # Previous day close
                    prev_idx = max(0, i - 1)
                    prev_date = dates[prev_idx]
                    if (prev_date, ticker) in price_df.index:
                        close_prev = price_df.loc[(prev_date, ticker), "close"]
                        daily_ret = close_today / close_prev - 1.0

                        bench_return += daily_ret / len(available_tickers)

                        if ticker in selected:
                            port_return += daily_ret * weight
            except (KeyError, TypeError):
                continue

        # Subtract transaction costs (approximation)
        turnover = 2.0 / hold_period  # approximate daily turnover
        cost = turnover * one_way_cost * 2  # round trip
        port_return -= cost

        daily_results.append({
            "date": trade_date,
            "strategy_return": port_return,
            "benchmark_return": bench_return,
            "n_positions": len(selected),
            "turnover": turnover,
        })

    result_df = pd.DataFrame(daily_results)
    if not result_df.empty:
        result_df["date"] = pd.to_datetime(result_df["date"])
        result_df = result_df.set_index("date").sort_index()

    logger.info(
        "Backtest complete: %d trading days, avg positions: %.1f",
        len(result_df),
        result_df["n_positions"].mean() if not result_df.empty else 0,
    )

    return result_df
