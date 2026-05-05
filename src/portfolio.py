"""
Hierarchical Risk Parity (HRP) — Lopez de Prado, AFML Ch.16.

Steps:
  1. Tree clustering on returns: distance d_ij = sqrt((1 - rho_ij) / 2),
     then hierarchical clustering on the *distance-of-distances* (column-wise
     Euclidean distances of the d matrix).
  2. Quasi-diagonalization: reorder rows/columns of the covariance matrix
     so that similar assets sit together.
  3. Recursive bisection: split the ordered list in halves, allocate weight
     to each half inversely proportional to its inverse-variance-portfolio
     variance.

Output: long-only weights summing to 1, no leverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def _correl_dist(corr: pd.DataFrame) -> pd.DataFrame:
    """d_ij = sqrt((1 - rho_ij) / 2), a proper metric."""
    return ((1 - corr) / 2.0).clip(lower=0.0).pow(0.5)


def _get_quasi_diag(link: np.ndarray) -> list[int]:
    """Sort clustered items by distance.

    Each row of the linkage matrix merges two clusters into one. Walk back
    through the merges, expanding clusters into their constituents.
    """
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _get_ivp(cov: np.ndarray) -> np.ndarray:
    """Inverse-variance portfolio."""
    ivp = 1.0 / np.diag(cov)
    return ivp / ivp.sum()


def _get_cluster_var(cov: np.ndarray, items: list[int]) -> float:
    cov_ = cov[np.ix_(items, items)]
    w_ = _get_ivp(cov_).reshape(-1, 1)
    return float(w_.T @ cov_ @ w_)


def _recursive_bisection(cov: np.ndarray, sort_ix: list[int]) -> pd.Series:
    w = pd.Series(1.0, index=sort_ix)
    c_items: list[list[int]] = [sort_ix]
    while c_items:
        new_c_items = []
        for items in c_items:
            if len(items) <= 1:
                continue
            half = len(items) // 2
            l_items = items[:half]
            r_items = items[half:]
            l_var = _get_cluster_var(cov, l_items)
            r_var = _get_cluster_var(cov, r_items)
            alpha = 1 - l_var / (l_var + r_var)
            w[l_items] *= alpha
            w[r_items] *= 1 - alpha
            new_c_items.append(l_items)
            new_c_items.append(r_items)
        c_items = new_c_items
    return w


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """Compute HRP weights from a returns DataFrame.

    Args:
        returns: T x N DataFrame of asset returns (rows = time, cols = assets).
                 Must have at least N+1 rows and no missing values across columns.

    Returns:
        Series of weights, indexed by column names, summing to 1, all >= 0.
    """
    cov = returns.cov().values
    corr = returns.corr().values
    dist = ((1 - corr) / 2.0).clip(min=0.0) ** 0.5

    # ensure exact symmetry / zero diagonal for squareform
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)

    cond_dist = squareform(dist, checks=False)
    link = linkage(cond_dist, method="single")
    sort_ix = _get_quasi_diag(link)

    w = _recursive_bisection(cov, sort_ix)
    w = w.sort_index()  # restore original asset order
    w.index = returns.columns[w.index]
    # numerical safety
    w = w.clip(lower=0.0)
    if w.sum() > 0:
        w = w / w.sum()
    return w


# -----------------------------------------------------------------------------
# Combining ML scores with HRP
# -----------------------------------------------------------------------------

def select_and_allocate(
    bet_sizes: pd.Series,
    returns_window: pd.DataFrame,
    top_n: int,
    min_history: int = 60,
    fallback_equal: bool = True,
) -> pd.Series:
    """Combine ML bet sizes with HRP.

    1. Pick the top_n stocks by bet_size (ties broken arbitrarily).
    2. Compute HRP weights using the returns_window for those stocks.
    3. Tilt HRP weights by bet_size: w_i ∝ hrp_i * bet_size_i, then renormalize.
       This blends ML conviction with risk-parity diversification.

    Args:
        bet_sizes: Series indexed by ticker, values in [0, 1].
        returns_window: Recent returns (rows = time, cols = tickers) for HRP cov.
        top_n: Number of names to hold.
        min_history: Min number of return observations needed.
        fallback_equal: If True and HRP fails (insufficient data, singular cov),
            fall back to equal weighting on the selected names.

    Returns:
        Series of weights summing to 1 (long-only, no leverage), indexed by
        the selected tickers.
    """
    candidates = bet_sizes[bet_sizes > 0].dropna()
    if len(candidates) == 0:
        return pd.Series(dtype=float)

    top = candidates.sort_values(ascending=False).head(top_n)
    selected = top.index.tolist()

    # restrict returns to selected tickers, drop tickers with too little history
    rw = returns_window[selected].dropna(axis=1, how="any")
    selected = list(rw.columns)
    if len(selected) < 2 or len(rw) < min_history:
        if fallback_equal and len(selected) >= 1:
            w = pd.Series(1.0 / len(selected), index=selected)
            return w
        return pd.Series(dtype=float)

    try:
        w_hrp = hrp_weights(rw)
    except Exception:
        if fallback_equal:
            w_hrp = pd.Series(1.0 / len(selected), index=selected)
        else:
            return pd.Series(dtype=float)

    # tilt by bet size
    w = w_hrp * top.reindex(w_hrp.index).fillna(0.0)
    if w.sum() <= 0:
        if fallback_equal:
            return pd.Series(1.0 / len(selected), index=selected)
        return pd.Series(dtype=float)
    return w / w.sum()


def inverse_vol_score_tilt(
    scores: pd.Series,
    returns_window: pd.DataFrame,
    top_n: int,
    vol_window: int = 63,
    score_power: float = 1.0,
) -> pd.Series:
    """Top-N selection with weights ∝ score / σ.

    Robust alternative to HRP when the covariance estimate is unstable
    (e.g. when many candidates are recently delisted). Equivalent to a naive
    inverse-vol portfolio tilted by the ML score, which behaves well in
    practice and avoids numerical pathologies in the linkage step of HRP.

    Args:
        scores: Series indexed by ticker, values >= 0 (e.g. xs_rank).
        returns_window: Recent daily returns DataFrame for vol estimation.
        top_n: Number of names to hold.
        vol_window: Trailing window for inverse-vol estimation (last
                    `vol_window` rows of `returns_window` are used).
        score_power: Exponent applied to the score before weighting; > 1
                     concentrates into the highest-conviction names.

    Returns:
        Long-only weights summing to 1, indexed by selected tickers.
    """
    s = scores[scores > 0].dropna()
    if len(s) == 0:
        return pd.Series(dtype=float)

    top = s.sort_values(ascending=False).head(top_n)
    selected = top.index.tolist()

    rw = returns_window[selected].iloc[-vol_window:].dropna(axis=1, how="any")
    selected = list(rw.columns)
    if len(selected) == 0:
        return pd.Series(dtype=float)
    if len(rw) < 10:
        # not enough data — fall back to score-only weighting on selected names
        w = top.reindex(selected).fillna(0.0).values ** score_power
    else:
        sigma = rw.std().replace(0, np.nan).values
        s_aligned = top.reindex(selected).fillna(0.0).values ** score_power
        w = np.where(np.isfinite(sigma), s_aligned / sigma, 0.0)

    if w.sum() <= 0:
        w = np.ones(len(selected))
    out = pd.Series(w, index=selected)
    return out / out.sum()


def vol_target_scale(
    weights: pd.Series,
    returns_window: pd.DataFrame,
    target_vol: float = 0.15,
    max_gross: float = 1.0,
    min_history: int = 60,
) -> pd.Series:
    """Scale a long-only weight vector to target an ex-ante annualized
    portfolio volatility, clipped at `max_gross` (no leverage if 1.0).

    The unallocated remainder (1 - sum(w_scaled)) sits in cash.

    Args:
        weights: Pre-scaled weights summing to 1 (output of select_and_allocate).
        returns_window: Trailing daily returns for the held tickers.
        target_vol: Annualized target volatility (e.g. 0.15 = 15%).
        max_gross: Cap on gross exposure (1.0 = fully invested, no leverage).
        min_history: Min observations needed to estimate covariance.

    Returns:
        Scaled weights (sum <= max_gross), indexed like the input.
    """
    if len(weights) == 0:
        return weights
    rw = returns_window[weights.index].dropna(axis=1, how="any")
    if len(rw) < min_history or rw.shape[1] < 2:
        return weights * max_gross  # fallback: just invest at max gross

    cov = rw.cov().values * 252  # annualize
    w = weights.reindex(rw.columns).fillna(0.0).values
    port_var = float(w @ cov @ w)
    if port_var <= 0:
        return weights * max_gross
    port_vol = port_var ** 0.5
    scale = min(target_vol / port_vol, max_gross / max(weights.sum(), 1e-9))
    return weights * scale
