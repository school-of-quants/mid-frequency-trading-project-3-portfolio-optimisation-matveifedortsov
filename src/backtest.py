"""
Vectorized backtest engine.

Models a long-only, no-leverage portfolio with monthly rebalancing.
Between rebalances, weights drift with prices (buy-and-hold dynamics).
Transaction costs (`tc_per_trade`) are charged on the gross turnover at each
rebalance.

We model TWAP execution at the rebalance: trades are executed at the close
of the rebalance day. The brief allows TWAP/VWAP/POV; close-price execution
is a clean approximation given daily data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rebalance_dates(index: pd.DatetimeIndex, freq: str = "MS") -> pd.DatetimeIndex:
    """First trading day of each calendar period (default: month-start)."""
    if freq == "MS":
        flag = index.to_series().groupby(index.to_period("M")).transform("first") == index
    elif freq == "WS":
        flag = index.to_series().groupby(index.to_period("W")).transform("first") == index
    elif freq == "QS":
        flag = index.to_series().groupby(index.to_period("Q")).transform("first") == index
    else:
        raise ValueError(f"Unsupported freq {freq}")
    return index[flag.values]


def backtest_long_only(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    initial_capital: float = 20_000_000,
    tc_per_trade: float = 0.001,
    rebalance_freq: str = "MS",
) -> dict:
    """Run a long-only, no-leverage backtest with monthly rebalance + drift.

    Args:
        prices: Adjusted close prices, T x N (rows = trading days, cols = tickers).
                MUST be forward-filled or contain NaN only where the asset wasn't
                tradable; we treat NaN as not-tradable on that day.
        target_weights: Sparse target-weight DataFrame, T x N. Rows where the
                portfolio should be rebalanced have non-NaN values that sum to
                <= 1; other rows are NaN. The first row of `prices` should
                contain a target (or be all-cash by default).
        initial_capital: Starting cash in dollars.
        tc_per_trade: Per-side transaction cost as fraction of trade notional
                (0.001 = 10 bps).
        rebalance_freq: Used only for sanity check / annotation.

    Returns:
        Dict with keys:
            equity_curve : pd.Series of total portfolio value
            returns      : pd.Series of daily portfolio returns
            weights      : pd.DataFrame of actual portfolio weights (drifted)
            turnover     : pd.Series of one-way turnover at each rebalance
            trade_log    : pd.DataFrame of (date, ticker, weight_change)
    """
    prices = prices.copy().sort_index()
    target_weights = target_weights.reindex(index=prices.index, columns=prices.columns)
    n_days, n_assets = prices.shape

    # daily simple returns; NaN -> 0 for the recursion (asset not held anyway)
    daily_ret = prices.pct_change().fillna(0.0).values
    px = prices.values

    actual_w = np.zeros((n_days, n_assets))
    cash = np.zeros(n_days)
    nav = np.zeros(n_days)

    nav[0] = initial_capital
    cash[0] = initial_capital
    turnover_list = []
    rebal_dates = []
    trade_records = []

    for t in range(n_days):
        # 1) drift today's weights forward by today's returns
        if t > 0:
            w_prev = actual_w[t - 1]
            r = daily_ret[t]
            # mark NaN prices: the asset can't return; drop its weight to 0 (delisted)
            valid = ~np.isnan(prices.values[t]) & ~np.isnan(prices.values[t - 1])
            r_safe = np.where(valid, r, 0.0)
            growth = 1.0 + r_safe
            asset_val = w_prev * nav[t - 1] * growth
            cash_val = cash[t - 1]  # cash earns 0% (conservative)
            nav[t] = asset_val.sum() + cash_val
            if nav[t] <= 0:
                # blown up — keep 0 going forward
                nav[t:] = 0
                actual_w[t:] = 0
                cash[t:] = 0
                break
            actual_w[t] = asset_val / nav[t]
            cash[t] = cash_val

        # 2) if today is a rebalance day, trade to the target
        target_row = target_weights.values[t]
        if not np.all(np.isnan(target_row)):
            target = np.where(np.isnan(target_row), 0.0, target_row)
            # ensure no leverage and drop weights on assets with NaN price today
            tradable = ~np.isnan(prices.values[t])
            target = np.where(tradable, target, 0.0)
            tot = target.sum()
            if tot > 1.0:
                target = target / tot  # normalize, never leveraged
                tot = 1.0

            delta = target - actual_w[t]
            traded_notional = np.abs(delta).sum() * nav[t]
            cost = traded_notional * tc_per_trade

            # apply costs by reducing NAV; reallocate to target after cost
            nav[t] = nav[t] - cost
            actual_w[t] = target
            cash[t] = nav[t] * (1.0 - tot)

            turnover_list.append(np.abs(delta).sum())
            rebal_dates.append(prices.index[t])
            for j in np.where(np.abs(delta) > 1e-9)[0]:
                trade_records.append((prices.index[t], prices.columns[j], delta[j]))

    equity_curve = pd.Series(nav, index=prices.index, name="equity")
    returns = equity_curve.pct_change().fillna(0.0).rename("returns")
    weights_df = pd.DataFrame(actual_w, index=prices.index, columns=prices.columns)
    turnover = pd.Series(turnover_list, index=pd.DatetimeIndex(rebal_dates), name="turnover")
    trade_log = pd.DataFrame(trade_records, columns=["date", "ticker", "weight_change"])

    return {
        "equity_curve": equity_curve,
        "returns": returns,
        "weights": weights_df,
        "turnover": turnover,
        "trade_log": trade_log,
    }


def benchmark_buy_and_hold(
    prices_or_index: pd.Series,
    initial_capital: float = 20_000_000,
) -> dict:
    """Buy-and-hold a single instrument (e.g. SPY) for benchmark comparison."""
    if isinstance(prices_or_index, pd.DataFrame):
        prices_or_index = prices_or_index.iloc[:, 0]
    p = prices_or_index.dropna()
    eq = initial_capital * (p / p.iloc[0])
    return {
        "equity_curve": eq.rename("equity"),
        "returns": eq.pct_change().fillna(0.0).rename("returns"),
    }
