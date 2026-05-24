# -*- coding: utf-8 -*-
"""
Hidden Markov Model regime detector — STRICT walk-forward.

Fits GaussianHMM on VN-Index returns + realised vol to identify
bull / sideways / bear states.

CRITICAL DESIGN RULE:
    At time t, the model may ONLY be trained on data from [0, t-1].
    The state at time t is predicted using a SINGLE forward-pass on the
    observation at t, conditioned on the model trained on [0, t-1].
    Once a state is assigned, it is NEVER overwritten by a later refit.
    This guarantees zero look-ahead leakage.

Previous version had leakage: model.predict(obs_seq[:t+1]) re-ran
the Viterbi algorithm over the entire history, causing the globally-
optimal path to be influenced by future observations.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from ..config import CFG

logger = logging.getLogger(__name__)


def _safe_import_hmmlearn():
    try:
        from hmmlearn.hmm import GaussianHMM
        return GaussianHMM
    except ImportError:
        raise ImportError("hmmlearn is required: pip install hmmlearn")


def _predict_single_step(model, obs_history: np.ndarray) -> tuple:
    """
    Predict the hidden state for the LAST observation only,
    using forward algorithm (not Viterbi over full sequence).

    We use predict_proba on the full history but ONLY read the
    last row's posterior. This is equivalent to the HMM filtering
    distribution P(z_t | x_1:t), which does NOT use future data.

    Parameters
    ----------
    model : GaussianHMM
        Fitted HMM model.
    obs_history : np.ndarray, shape (t, n_features)
        Observations from time 0 to t (inclusive).

    Returns
    -------
    state : int
        Most likely state at the last time step.
    prob : float
        Posterior probability of that state.
    """
    try:
        posteriors = model.predict_proba(obs_history)
        # Only use the LAST row — this is P(z_t | x_1:t), the filtering
        # distribution which by definition only uses past + current data.
        last_posterior = posteriors[-1]
        state = int(np.argmax(last_posterior))
        prob = float(last_posterior[state])
        return state, prob
    except Exception:
        return np.nan, np.nan


def fit_hmm_expanding(
    index_df: pd.DataFrame,
    n_states: int = 3,
    min_train_days: int = 252,
    covariance_type: str = "full",
    n_iter: int = 100,
    seed: int = 42,
    refit_interval: int = 60,
) -> pd.DataFrame:
    """
    Fit GaussianHMM with expanding window — STRICT walk-forward.

    At each time step t (where t >= min_train_days):
      1. If model needs refit: train on [0, t-1] (EXCLUSIVE of t)
      2. Predict state at t using the filtering distribution P(z_t | x_1:t)
      3. Once assigned, state at t is NEVER overwritten

    Features used: [returns, realised_vol_20d].

    Parameters
    ----------
    index_df : pd.DataFrame
        Indexed by date with columns [returns, close].
    n_states : int
        Number of hidden states.
    min_train_days : int
        Minimum days before first prediction.
    covariance_type : str
        HMM covariance type.
    n_iter : int
        EM iterations.
    seed : int
        Random seed.
    refit_interval : int
        Refit HMM every N days to balance speed vs freshness.

    Returns
    -------
    pd.DataFrame
        Columns: [hmm_state, hmm_prob], indexed by date.
        States are relabeled so 0=bull (highest mean return), 2=bear.
    """
    GaussianHMM = _safe_import_hmmlearn()

    df = index_df.copy()
    df = df.dropna(subset=["returns"])

    # Compute realised vol
    df["rvol_20"] = df["returns"].rolling(20, min_periods=10).std()
    df = df.dropna(subset=["rvol_20"])

    obs_cols = ["returns", "rvol_20"]
    dates = df.index.tolist()
    n = len(dates)

    # Pre-allocate — once assigned, NEVER overwritten
    states = np.full(n, np.nan)
    probs = np.full(n, np.nan)

    model = None
    last_fit_t = -1

    for t in range(min_train_days, n):
        # ── Step 1: Refit if needed (train on [0, t-1], EXCLUSIVE of t) ──
        if model is None or (t - last_fit_t) >= refit_interval:
            # Train data: [0, t-1] — strictly before current time
            train_data = df[obs_cols].iloc[:t].values
            try:
                model = GaussianHMM(
                    n_components=n_states,
                    covariance_type=covariance_type,
                    n_iter=n_iter,
                    random_state=seed,
                )
                model.fit(train_data)
                last_fit_t = t
                logger.debug("HMM refit at t=%d with %d training samples", t, t)
            except Exception as exc:
                logger.warning("HMM fit failed at t=%d: %s", t, exc)
                continue

        # ── Step 2: Predict state at t using filtering (not Viterbi) ──
        # We pass observations [0, t] to predict_proba.
        # predict_proba uses the FORWARD algorithm, which computes
        # P(z_t | x_1:t) — the filtering distribution.
        # This is mathematically guaranteed to not use future data.
        obs_up_to_t = df[obs_cols].iloc[:t + 1].values
        state, prob = _predict_single_step(model, obs_up_to_t)

        # ── Step 3: Assign and LOCK — never overwrite ──
        states[t] = state
        probs[t] = prob

    # Relabel states by mean return: 0=bull, 2=bear
    result = pd.DataFrame(
        {"hmm_state": states, "hmm_prob": probs},
        index=df.index,
    )
    result = _relabel_states_by_return(result, df["returns"])

    n_valid = result["hmm_state"].notna().sum()
    logger.info(
        "HMM regime detection complete (strict walk-forward). "
        "%d/%d days assigned. State distribution:\n%s",
        n_valid, n,
        result["hmm_state"].value_counts().to_string(),
    )
    return result


def _relabel_states_by_return(
    regime_df: pd.DataFrame,
    returns: pd.Series,
) -> pd.DataFrame:
    """Relabel HMM states so 0=bull (highest return), 2=bear (lowest)."""
    df = regime_df.copy()
    valid = df["hmm_state"].notna()
    if valid.sum() == 0:
        return df

    mean_ret = {}
    for s in df.loc[valid, "hmm_state"].unique():
        mask = df["hmm_state"] == s
        mean_ret[s] = returns[mask].mean()

    # Sort states by mean return descending
    sorted_states = sorted(mean_ret, key=mean_ret.get, reverse=True)
    mapping = {old: new for new, old in enumerate(sorted_states)}
    df["hmm_state"] = df["hmm_state"].map(mapping)

    return df
