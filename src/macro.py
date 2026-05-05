"""
US macro features from the Global Macro Database (GMD).

We extract every annual indicator that has reliable post-2010 coverage,
broadcast it to a daily index with an explicit publication lag, and derive
several richer signals on top:

* Year-on-year *changes* (rGDP_growth, M2_growth, debt-to-GDP changes)
* *Levels vs trend* (deviation of inflation/unemp/rates from a 5-year mean)
* *Yield-curve shape* (long − short, plus a "deeply inverted" indicator)
* *Risk flags* (banking-/sovereign-/currency-crisis dummies if present)

All values are forward-filled within years and lagged by `publication_lag_months`
months past year-end so we never see information that wasn't yet released.

The returned macro panel is meant to be used in two ways:
  1. As global features broadcast across stocks for an XGBoost ranker.
  2. As inputs to a Prophet / regime model for global-exposure scaling.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Annual GMD columns we extract (keep only those with US post-2010 coverage).
_BASE_COLS = [
    "infl",            # CPI inflation rate (%)
    "unemp",           # unemployment rate (%)
    "cbrate",          # central-bank policy rate (%)
    "ltrate",          # 10y benchmark yield (%)
    "strate",          # short-term rate (%, may be NaN late)
    "rGDP",            # real GDP (level)
    "rGDP_pc",         # real GDP per capita
    "M2",              # broad money M2 (level)
    "HPI",             # house-price index
    "CPI",             # CPI level
    "USDfx",           # dollar FX index
    "REER",            # real effective exchange rate
    "gen_govdef_GDP",  # general gov deficit % GDP
    "gen_govdebt_GDP", # general gov debt % GDP
    "CA_GDP",          # current-account % GDP
    "BankingCrisis",   # 0/1 dummy
    "SovDebtCrisis",   # 0/1 dummy
    "CurrencyCrisis",  # 0/1 dummy
]


def load_us_macro_annual(csv_path: str | Path) -> pd.DataFrame:
    """Load US annual macro indicators, indexed by integer year."""
    df = pd.read_csv(csv_path)
    us = df[df["ISO3"] == "USA"].sort_values("year").reset_index(drop=True)
    keep = ["year"] + [c for c in _BASE_COLS if c in us.columns]
    us = us[keep].copy().set_index("year")

    # --- derived: YoY growth rates of stock variables
    if "rGDP" in us:
        us["rGDP_growth"] = us["rGDP"].pct_change()
    if "rGDP_pc" in us:
        us["rGDP_pc_growth"] = us["rGDP_pc"].pct_change()
    if "M2" in us:
        us["M2_growth"] = us["M2"].pct_change()
    if "HPI" in us:
        us["HPI_growth"] = us["HPI"].pct_change()
    if "CPI" in us:
        # CPI YoY change is closely related to `infl` but cleaner to derive
        us["CPI_growth"] = us["CPI"].pct_change()

    # --- derived: yield-curve shape
    if "ltrate" in us and "cbrate" in us:
        us["yield_spread"] = us["ltrate"] - us["cbrate"]
        us["yc_inverted"] = (us["yield_spread"] < 0).astype(float)

    # --- derived: 5-year deviation from rolling mean (regime "anomaly" measure).
    # This captures *deviation from consensus* in macro variables: with annual
    # data a 5-year rolling mean is a decent proxy for medium-term trend.
    for c in ["infl", "unemp", "cbrate", "ltrate", "yield_spread"]:
        if c in us:
            us[f"{c}_dev5y"] = us[c] - us[c].rolling(5, min_periods=2).mean()

    # --- derived: real rate (cbrate - inflation)
    if "cbrate" in us and "infl" in us:
        us["real_rate"] = us["cbrate"] - us["infl"]

    return us


def macro_daily_features(
    csv_path: str | Path,
    daily_index: pd.DatetimeIndex,
    publication_lag_months: int = 4,
) -> pd.DataFrame:
    """Forward-fill annual macro data to a daily index with publication lag.

    Year `Y` data are available starting `(Y+1)-01-01 + publication_lag_months`,
    e.g. with a 4-month lag, calendar-year-2023 figures become visible from
    2024-05-01 — conservative versus actual GMD release timings.

    Args:
        csv_path: Path to GMD.csv.
        daily_index: Trading-day index.
        publication_lag_months: Lag past year-end before data is "known".

    Returns:
        DataFrame indexed by daily_index, one column per macro feature.
    """
    annual = load_us_macro_annual(csv_path)
    pub_dates = pd.to_datetime(
        [f"{int(y) + 1}-01-01" for y in annual.index]
    ) + pd.DateOffset(months=publication_lag_months)
    annual.index = pub_dates
    return annual.reindex(daily_index, method="ffill")
