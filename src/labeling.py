"""
Triple-barrier labeling (Lopez de Prado, AFML Ch.3).

For each (date, ticker) we open a hypothetical long at the close, then watch
the next H trading days. If the cumulative return crosses an upper barrier
(`pt * sigma`), label = +1. If it crosses a lower barrier (`-sl * sigma`),
label = -1. If neither barrier is touched within H days (vertical barrier),
label = 0 (or the sign of the final return, depending on `vertical_sign`).

For the *base* model (binary upward classifier), we collapse to:
    y = 1 if label == +1 else 0
For the *meta-labeler*, we use whether the base prediction was correct.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    close: pd.DataFrame,
    horizon: int = 21,
    pt: float = 2.0,
    sl: float = 2.0,
    vol_window: int = 22,
    min_periods: int | None = None,
    vertical_sign: bool = True,
) -> pd.DataFrame:
    """Compute triple-barrier labels for every (date, ticker).

    Vectorized across tickers and across the horizon (no per-row Python loop):
    we materialize the H-day forward returns matrix and find the first index
    along axis=2 where each barrier is breached.

    Args:
        close: Adjusted close prices, indexed by date, columns = tickers.
        horizon: Vertical barrier (max holding days).
        pt: Profit-take multiple of rolling sigma.
        sl: Stop-loss multiple of rolling sigma.
        vol_window: Window for rolling daily-return sigma.
        min_periods: Min periods for the rolling sigma (defaults to vol_window).
        vertical_sign: If True and barrier untouched, label = sign(final_ret).
                       If False, label = 0 in that case.

    Returns:
        DataFrame of labels in {-1, 0, +1}, shape = close.shape.
        Last `horizon` rows will be NaN (insufficient forward data).
    """
    if min_periods is None:
        min_periods = vol_window

    rets = close.pct_change()
    sigma = rets.rolling(vol_window, min_periods=min_periods).std()

    n, m = close.shape
    # forward log-returns matrix: forward_ret[t, k, j] = log(C[t+k+1] / C[t]) for ticker j
    # we use simple returns for thresholds since sigma is in simple-return units
    log_close = np.log(close.values)

    # build a (horizon, n, m) array of forward returns
    fwd = np.full((horizon, n, m), np.nan)
    for k in range(1, horizon + 1):
        if k >= n:
            break
        fwd[k - 1, :-k, :] = log_close[k:, :] - log_close[:-k, :]
    # convert to simple returns approximation (small returns => log≈simple),
    # but for accuracy use exp(.)-1
    fwd_simple = np.expm1(fwd)

    sig = sigma.values  # (n, m)
    upper = pt * sig
    lower = -sl * sig

    # broadcast: thresholds along axis 0 (horizon) — barriers same for all k
    upper_b = np.broadcast_to(upper, (horizon, n, m))
    lower_b = np.broadcast_to(lower, (horizon, n, m))

    # first hit indices (or horizon if never)
    hit_up = np.where(fwd_simple >= upper_b, np.arange(horizon)[:, None, None],
                      np.iinfo(np.int32).max).min(axis=0)
    hit_dn = np.where(fwd_simple <= lower_b, np.arange(horizon)[:, None, None],
                      np.iinfo(np.int32).max).min(axis=0)

    no_hit = (hit_up == np.iinfo(np.int32).max) & (hit_dn == np.iinfo(np.int32).max)
    labels = np.zeros((n, m), dtype=float)
    labels[hit_up < hit_dn] = 1.0
    labels[hit_dn < hit_up] = -1.0
    # ties (extremely rare; both touched same bar): mark as 0
    labels[(hit_up == hit_dn) & ~no_hit] = 0.0

    if vertical_sign:
        # for no-hit rows use sign of the final-horizon return
        final_ret = fwd_simple[horizon - 1] if horizon > 0 else np.zeros((n, m))
        labels[no_hit] = np.sign(np.where(np.isnan(final_ret), 0.0, final_ret))[no_hit]
    else:
        labels[no_hit] = 0.0

    # invalidate rows where we don't have full horizon ahead
    if horizon > 0:
        labels[-horizon:, :] = np.nan
    # invalidate rows where sigma is NaN
    labels[np.isnan(sig)] = np.nan
    # invalidate where close is NaN
    labels[np.isnan(close.values)] = np.nan

    return pd.DataFrame(labels, index=close.index, columns=close.columns)


def label_event_endpoints(
    close: pd.DataFrame,
    horizon: int,
) -> pd.Series:
    """For each (date, ticker) sample, the event ends at date + horizon.

    Returns a Series indexed (date, ticker) -> end_date. Used by PurgedKFold
    to know which training samples overlap with which test samples in time.
    """
    dates = close.index
    end_idx = np.minimum(np.arange(len(dates)) + horizon, len(dates) - 1)
    end_dates = dates[end_idx]
    end_map = pd.Series(end_dates, index=dates)
    # broadcast to (date, ticker)
    rows = []
    for ticker in close.columns:
        s = end_map.copy()
        rows.append(pd.DataFrame({"end": s.values, "ticker": ticker, "date": s.index}))
    df = pd.concat(rows, ignore_index=True)
    df = df.set_index(["date", "ticker"])["end"]
    return df
