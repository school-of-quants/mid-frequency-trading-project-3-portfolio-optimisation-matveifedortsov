"""Performance metrics: Sharpe, Sortino, Calmar, MaxDD, IPC, etc."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Geometric mean annualized return."""
    n = len(returns)
    if n == 0:
        return np.nan
    cum = float((1.0 + returns).prod())
    if cum <= 0:
        return -1.0
    return cum ** (periods_per_year / n) - 1.0


def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(returns.std()) * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0,
                 periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio."""
    excess = returns - rf / periods_per_year
    sd = excess.std()
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(excess.mean() / sd) * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, rf: float = 0.0,
                  periods_per_year: int = 252) -> float:
    excess = returns - rf / periods_per_year
    downside = excess.clip(upper=0)
    sd = np.sqrt((downside ** 2).mean())
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(excess.mean() / sd) * np.sqrt(periods_per_year)


def max_drawdown(equity_curve: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """Max drawdown (negative number), peak date, trough date."""
    running_max = equity_curve.cummax()
    dd = equity_curve / running_max - 1.0
    trough = dd.idxmin()
    peak = equity_curve.loc[:trough].idxmax()
    return float(dd.min()), peak, trough


def max_drawdown_duration(equity_curve: pd.Series) -> int:
    """Max drawdown duration in trading days."""
    running_max = equity_curve.cummax()
    underwater = equity_curve < running_max
    if not underwater.any():
        return 0
    # find longest consecutive run of True
    grp = (underwater != underwater.shift()).cumsum()
    runs = underwater.groupby(grp).sum()
    return int(runs.max())


def calmar_ratio(returns: pd.Series, equity_curve: pd.Series,
                 periods_per_year: int = 252) -> float:
    ann_ret = annualized_return(returns, periods_per_year)
    mdd, _, _ = max_drawdown(equity_curve)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return ann_ret / abs(mdd)


def intra_portfolio_correlation(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    window: int = 63,
) -> pd.Series:
    """Intra-Portfolio Correlation (IPC).

    For each date t, compute the weighted average of pairwise correlations
    between held assets, where weights are normalized portfolio weights:

        IPC_t = sum_{i != j} w_i w_j rho_ij(t) / sum_{i != j} w_i w_j

    `rho_ij(t)` uses a rolling window ending at t.

    Args:
        weights: Daily weights, T x N (rows sum to <= 1, fraction in cash = 1 - sum).
        returns: Daily simple returns, T x N.
        window: Rolling window for correlation estimation.

    Returns:
        Daily IPC series.
    """
    # align
    weights = weights.reindex_like(returns).fillna(0.0)
    out = pd.Series(index=returns.index, dtype=float)
    rets_vals = returns.values
    w_vals = weights.values

    # we'll compute on a thinned schedule to save time, then forward-fill
    thin = 5  # update IPC every 5 trading days
    last_val = np.nan
    for t in range(len(returns)):
        if t < window:
            out.iloc[t] = np.nan
            continue
        if t % thin != 0:
            out.iloc[t] = last_val
            continue
        # active assets: weight > 0 and sufficient history
        w = w_vals[t]
        active = w > 1e-6
        if active.sum() < 2:
            out.iloc[t] = 0.0
            last_val = 0.0
            continue
        win = rets_vals[t - window + 1: t + 1, active]
        # corrcoef on columns
        if np.any(np.isnan(win)):
            # fill remaining NaN with 0 (asset wasn't tradable in part of window)
            win = np.nan_to_num(win, nan=0.0)
        corr = np.corrcoef(win.T)
        if corr.shape == ():
            out.iloc[t] = np.nan
            continue
        # zero the diagonal to exclude self-correlations
        np.fill_diagonal(corr, 0.0)
        w_active = w[active]
        # normalize weights of active assets
        ws = w_active / w_active.sum()
        # weighted off-diagonal mean
        wmat = np.outer(ws, ws)
        np.fill_diagonal(wmat, 0.0)
        denom = wmat.sum()
        ipc_t = float((corr * wmat).sum() / denom) if denom > 0 else 0.0
        out.iloc[t] = ipc_t
        last_val = ipc_t
    return out.rename("IPC").ffill()


def summarize(
    returns: pd.Series,
    equity_curve: pd.Series,
    name: str = "Strategy",
    periods_per_year: int = 252,
) -> pd.Series:
    """One-shot summary table."""
    mdd, peak, trough = max_drawdown(equity_curve)
    mdd_dur = max_drawdown_duration(equity_curve)
    return pd.Series({
        "Annualized Return": annualized_return(returns, periods_per_year),
        "Annualized Vol": annualized_vol(returns, periods_per_year),
        "Sharpe": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "Sortino": sortino_ratio(returns, periods_per_year=periods_per_year),
        "Calmar": calmar_ratio(returns, equity_curve, periods_per_year),
        "Max Drawdown": mdd,
        "Max DD Duration (days)": mdd_dur,
        "Total Return": equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0,
    }, name=name)
