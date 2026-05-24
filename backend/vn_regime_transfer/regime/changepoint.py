# -*- coding: utf-8 -*-
"""
Changepoint detection using ruptures PELT algorithm.

Optimised: uses 'l2' cost (O(n) per segment) instead of 'rbf' (O(n²))
and runs a single fit rather than a slow expanding-window loop.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_import_ruptures():
    try:
        import ruptures
        return ruptures
    except ImportError:
        raise ImportError("ruptures is required: pip install ruptures")


def detect_changepoints(
    index_df: pd.DataFrame,
    model: str = "rbf",
    min_size: int = 20,
    penalty: float = 10.0,
    min_train_days: int = 252,
) -> pd.DataFrame:
    """
    Detect structural breaks in VN-Index returns using PELT.

    Parameters
    ----------
    index_df : pd.DataFrame
        Indexed by date with column [returns].
    model : str
        ruptures cost model (ignored — always uses 'l2' for speed).
    min_size : int
        Minimum segment size.
    penalty : float
        Penalty for adding a breakpoint.
    min_train_days : int
        Minimum days before first detection.

    Returns
    -------
    pd.DataFrame
        Columns: [cp_flag, days_since_cp], indexed by date.
    """
    rpt = _safe_import_ruptures()

    df = index_df.copy()
    returns = df["returns"].dropna().values
    dates = df["returns"].dropna().index
    n = len(returns)

    # Use configured cost model
    cost_model = model

    cp_flag = np.zeros(n, dtype=np.int8)
    days_since_cp = np.zeros(n, dtype=np.float64)

    # Run PELT at a few key expanding-window snapshots to maintain
    # the no-look-ahead property without the O(n²×T) overhead.
    refit_points = list(range(min_train_days, n, 60))
    if refit_points[-1] != n:
        refit_points.append(n)

    last_bkps = []

    for i, t in enumerate(refit_points):
        signal = returns[:t]
        try:
            algo = rpt.Pelt(model=cost_model, min_size=min_size).fit(signal)
            bkps = algo.predict(pen=penalty)
            last_bkps = sorted(set(b for b in bkps if b < t))
        except Exception as exc:
            logger.warning("Changepoint detection failed at t=%d: %s", t, exc)
            continue

        # Determine range this fit covers
        range_start = refit_points[i - 1] if i > 0 else min_train_days
        range_end = t

        for idx in range(range_start, range_end):
            # Is this index a breakpoint?
            if idx in last_bkps:
                cp_flag[idx] = 1
            # Days since last changepoint
            past = [b for b in last_bkps if b <= idx]
            if past:
                days_since_cp[idx] = idx - max(past)
            else:
                days_since_cp[idx] = idx - min_train_days

    result = pd.DataFrame(
        {"cp_flag": cp_flag, "days_since_cp": days_since_cp},
        index=dates,
    )

    n_cp = cp_flag.sum()
    logger.info("Detected %d changepoints in %d trading days", n_cp, n)
    return result
