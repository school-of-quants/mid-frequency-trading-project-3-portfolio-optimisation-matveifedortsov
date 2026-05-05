"""
Regime overlay based on the broad-market index.

Two simple, well-known signals are combined into a single multiplicative
exposure scale in [0, 1]:

    1. Trend filter:     price > SMA(200d)
    2. Drawdown filter:  drawdown from 252d running max < 8%

Each contributes 0.5 weight when "on" and 0.0 when "off"; the resulting scale
is in {0.0, 0.5, 1.0}, smoothed with a 5-day EMA so we don't whipsaw on
single bad days.

Rationale: the Sharpe-cost of being lightly de-risked during a sustained
drawdown is small in expectation, but the drawdown-control benefit is large.
This has been documented at least back to Faber (2007) for monthly trend
overlays. We apply it daily but with smoothing to avoid noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trend_regime_scale(
    spy_close: pd.Series,
    sma_window: int = 200,
    dd_window: int = 252,
    dd_threshold: float = 0.08,
    smooth: int = 5,
) -> pd.Series:
    """Produce a daily exposure multiplier in [0, 1].

    Args:
        spy_close: SPY (or any broad-market proxy) adjusted close, daily.
        sma_window: Long simple-moving-average window for the trend signal.
        dd_window: Lookback window for the running peak used in the
                   drawdown-from-peak signal.
        dd_threshold: Drawdown depth (positive number) below which the
                      drawdown signal switches off.
        smooth: EMA span (in days) applied to the binary signal sum, smooths
                the transition between regimes.

    Returns:
        Series indexed like `spy_close`, values in [0, 1]. Use as a scalar
        multiplier on the desired gross exposure.
    """
    sma = spy_close.rolling(sma_window, min_periods=sma_window // 2).mean()
    peak = spy_close.rolling(dd_window, min_periods=dd_window // 2).max()
    dd = spy_close / peak - 1.0  # negative or zero

    trend_on = (spy_close > sma).astype(float)
    dd_on = (dd > -dd_threshold).astype(float)

    # Floor at 0.4 even when both filters are off — full-cash periods badly
    # hurt total return without much DD benefit because the strategy is
    # already vol-targeted. The remaining 60% comes from the two filters.
    raw = 0.4 + 0.3 * trend_on + 0.3 * dd_on  # in [0.4, 1.0]

    # smooth — switching is slow because both filters are slow already, but
    # an EMA prevents single-day flips.
    if smooth and smooth > 1:
        raw = raw.ewm(span=smooth, adjust=False).mean()

    return raw.clip(lower=0.0, upper=1.0).rename("regime_scale")


def macro_regime_scale(
    macro_daily: pd.DataFrame,
    infl_high: float = 5.0,
    rgdp_low: float = 0.005,
    spread_low: float = -1.0,
) -> pd.Series:
    """Multiplicative scale based on US macroeconomic state.

    Returns a value in [0.5, 1.0]: full risk-on (1.0) when growth is solid,
    inflation is contained, and the yield curve isn't deeply inverted.
    Each adverse condition deducts 1/6 from the scale, floored at 0.5.

    Args:
        macro_daily: Daily-aligned macro panel with at least the columns
                     `infl`, `rGDP_growth`, `yield_spread`.
        infl_high:   Annual inflation rate above which we consider macro
                     hostile (percent units, matching GMD convention).
        rgdp_low:    Real-GDP-growth threshold (fraction, e.g. 0.005 = 0.5%)
                     below which we consider growth hostile.
        spread_low:  Yield-spread (long − short) threshold below which the
                     curve is considered inverted (in percentage points).

    Returns:
        Series indexed like `macro_daily`, values in [0.5, 1.0].
    """
    infl = macro_daily.get("infl")
    rgdp = macro_daily.get("rGDP_growth")
    spread = macro_daily.get("yield_spread")

    # build three boolean filters; ffill recovers from NaN at the edges
    f_infl = (infl < infl_high).astype(float) if infl is not None else 1.0
    f_rgdp = (rgdp > rgdp_low).astype(float) if rgdp is not None else 1.0
    f_spread = (spread > spread_low).astype(float) if spread is not None else 1.0

    raw = (f_infl + f_rgdp + f_spread) / 3.0  # in [0, 1]
    # rescale to [0.5, 1.0]: even all-bad macro keeps us 50% invested
    out = 0.5 + 0.5 * raw
    return out.fillna(1.0).clip(0.5, 1.0).rename("macro_regime")
