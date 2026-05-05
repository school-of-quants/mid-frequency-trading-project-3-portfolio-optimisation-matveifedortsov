"""
SPY-return forecasting via Prophet.

Strategic role: feed a forecast of next-21-day SPY return into the regime
overlay. When Prophet (which decomposes into trend + weekly/yearly seasonality)
projects a positive trend, hold full exposure; when projecting flat or
negative, scale down. This is one of three independent regime overlays we
combine multiplicatively (price-trend filter, macro filter, Prophet filter)
so that no single signal can dominate the de-risking decision.

We retrain Prophet on a sliding window ending at each rebalance date, using
only data observable at that time (no look-ahead). Prophet trains in
~1-3 seconds on 5 years of data so monthly retraining costs ≈ 1 minute total.
"""
from __future__ import annotations

import warnings
from contextlib import redirect_stderr, redirect_stdout
import os

import numpy as np
import pandas as pd

# Prophet emits a sea of `cmdstanpy` chatter at INFO level — silence it.
import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


def prophet_log_return_forecast(
    spy_close: pd.Series,
    rebalance_dates: pd.DatetimeIndex,
    train_window_days: int = 252 * 5,
    forecast_horizon: int = 21,
) -> pd.Series:
    """Forecast cumulative log-return for the next `forecast_horizon` days at
    every rebalance date, using a Prophet model fit on the trailing window.

    Args:
        spy_close: Daily SPY adjusted close (Series).
        rebalance_dates: Dates at which we want a forecast.
        train_window_days: Trailing window length used to fit each Prophet.
        forecast_horizon: Forward horizon for the cumulative return forecast.

    Returns:
        Series indexed by `rebalance_dates`, values are forecast cumulative
        log-returns over the next `forecast_horizon` trading days.
    """
    from prophet import Prophet

    log_close = np.log(spy_close.dropna())

    out = {}
    for d in rebalance_dates:
        # data available at d (use < d to be strictly point-in-time)
        history = log_close.loc[:d].iloc[:-1]
        if len(history) < 252:
            out[d] = 0.0
            continue
        history = history.iloc[-train_window_days:]
        df = pd.DataFrame({
            "ds": history.index,
            "y":  history.values,
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(os.devnull, "w") as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    m = Prophet(
                        daily_seasonality=False,
                        weekly_seasonality=True,
                        yearly_seasonality=True,
                        changepoint_prior_scale=0.05,
                        n_changepoints=15,
                    )
                    m.fit(df)
                    future = m.make_future_dataframe(
                        periods=forecast_horizon, freq="B", include_history=False,
                    )
                    fcst = m.predict(future)
        end_log = float(fcst["yhat"].iloc[-1])
        last_log = float(history.iloc[-1])
        out[d] = end_log - last_log  # forecast cumulative log return
    return pd.Series(out, name="prophet_logret_fcst").astype(float)


def prophet_regime_scale(
    fcst_logret: pd.Series,
    daily_index: pd.DatetimeIndex,
    pos_floor: float = -0.02,   # 21d forecast < -2% → 0.7 scale
    pos_ceil:  float =  0.005,  # 21d forecast > +0.5% → 1.0 scale
    floor:     float = 0.7,
) -> pd.Series:
    """Convert per-rebalance forecast log-return into a daily exposure scalar.

    Linear interpolation between [pos_floor, pos_ceil] mapping to
    [floor, 1.0]. The default range and floor are intentionally mild —
    Prophet on a noisy daily series produces forecasts that mostly hover
    near zero, and a hard de-risk on every slightly-negative forecast
    would crush Sharpe in bull markets. Used as a *third* overlay on top
    of the price-trend and macro regimes, so even the worst Prophet
    forecast still leaves ~70% of the de-risking decision to the others.

    Args:
        fcst_logret: Output of prophet_log_return_forecast.
        daily_index: Trading-day index to align onto.
        pos_floor: Forecast level at which scale hits its floor.
        pos_ceil:  Forecast level at which scale hits 1.0.
        floor:     Minimum scale for very negative forecasts.

    Returns:
        Daily Series in [floor, 1.0].
    """
    raw = fcst_logret.reindex(daily_index, method="ffill")
    scale = (raw - pos_floor) / max(pos_ceil - pos_floor, 1e-9)
    scale = scale.clip(lower=0.0, upper=1.0)
    return (floor + (1 - floor) * scale).rename("prophet_regime")
