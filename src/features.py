"""
Cross-sectional feature engineering.

We build per-(date, ticker) features computed from price/volume history.
All operations are vectorized across tickers — no per-ticker Python loops.

The output is a long-format DataFrame indexed by (date, ticker) with one
column per feature, suitable for direct ingestion into XGBoost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore_xs(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score (per row), robust to all-NaN rows."""
    mu = x.mean(axis=1)
    sd = x.std(axis=1).replace(0, np.nan)
    return x.sub(mu, axis=0).div(sd, axis=0)


def _rank_xs(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank in [0, 1]."""
    return x.rank(axis=1, pct=True)


def build_features(
    close: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
    market_proxy: pd.Series | None = None,
) -> pd.DataFrame:
    """Build cross-sectional features.

    Features include:
      Momentum:
        - mom_1m, mom_3m, mom_6m, mom_12m_1m (12-month skipping last month)
        - mom_1w_rev (1-week reversal)
      Volatility:
        - vol_22, vol_63, vol_252
        - vol_ratio_22_252 (short vs long)
      Price-MA deviations:
        - dev_5, dev_22, dev_63, dev_252
      Cross-sectional ranks of the above (helps tree models)
      Liquidity / size proxy:
        - dollar_volume_22, log_dvol_22 (rank)
      Range / efficiency:
        - hl_range_22 (avg high-low / close)
      Beta to market (if market_proxy provided):
        - beta_252

    Each feature is shifted by 1 day so that the feature available at time t
    uses only data up to t-1 (no look-ahead).

    Args:
        close: Adjusted close prices, indexed by date, columns = tickers.
        volume: Volume DataFrame (same shape) — used for liquidity features.
        high, low: Optional, for range features.
        market_proxy: Optional Series of market returns (e.g. SPY) for beta.

    Returns:
        Long-format DataFrame indexed by (date, ticker), one column per feature.
    """
    out: dict[str, pd.DataFrame] = {}
    rets = close.pct_change()

    # --- Momentum ---
    out["mom_1m"] = close.pct_change(21)
    out["mom_3m"] = close.pct_change(63)
    out["mom_6m"] = close.pct_change(126)
    # 12-1 momentum: return from t-252 to t-21 (skip the last month)
    out["mom_12m_1m"] = (close.shift(21) / close.shift(252)) - 1
    # short-term reversal (1 week)
    out["mom_1w_rev"] = close.pct_change(5)

    # --- Volatility ---
    out["vol_22"] = rets.rolling(22).std()
    out["vol_63"] = rets.rolling(63).std()
    out["vol_252"] = rets.rolling(252).std()
    out["vol_ratio_22_252"] = out["vol_22"] / out["vol_252"].replace(0, np.nan)

    # --- Deviation from MAs (price minus MA, normalized by price) ---
    for w in (5, 22, 63, 252):
        ma = close.rolling(w).mean()
        out[f"dev_{w}"] = (close - ma) / close.replace(0, np.nan)

    # MA crossover signal
    out["ma_50_200"] = (
        close.rolling(50).mean() - close.rolling(200).mean()
    ) / close.replace(0, np.nan)

    # --- Liquidity ---
    if volume is not None:
        # align: volume should match close columns
        volume = volume.reindex_like(close)
        dvol = (close * volume).rolling(22).mean()
        out["dollar_volume_22"] = dvol
        out["log_dvol_22"] = np.log1p(dvol)

    # --- Range / efficiency ---
    if high is not None and low is not None:
        high = high.reindex_like(close)
        low = low.reindex_like(close)
        out["hl_range_22"] = ((high - low) / close.replace(0, np.nan)).rolling(22).mean()

    # --- Beta to market ---
    if market_proxy is not None:
        mkt = market_proxy.reindex(close.index).pct_change()
        # rolling 252-day beta: cov(r, mkt) / var(mkt)
        # vectorized via rolling apply isn't great -> compute manually
        var_mkt = mkt.rolling(252).var()
        # cov(r_i, mkt) per column
        mkt_centered = mkt - mkt.rolling(252).mean()
        rets_centered = rets.sub(rets.rolling(252).mean())
        cov = (rets_centered.mul(mkt_centered, axis=0)).rolling(252).mean()
        beta = cov.div(var_mkt, axis=0)
        out["beta_252"] = beta

        # Residual (idiosyncratic) momentum — Blitz, Huij, Martens 2011.
        # Take 12-1 momentum of beta-adjusted returns: r_i - beta_i * r_mkt.
        # Idiosyncratic momentum has historically been a stronger signal than
        # raw price momentum on US large-caps because it strips out the
        # market-regime drift that contaminates the latter.
        resid = rets.sub(beta.mul(mkt, axis=0))
        # 12-1 month residual momentum: sum of last 252 daily residuals,
        # skipping the most recent 21 (lagged via .shift(21)).
        out["resid_mom_12m_1m"] = resid.rolling(231).sum().shift(21)
        # Idiosyncratic volatility — well-known anomaly (low idio vol earns
        # higher risk-adjusted returns; Ang et al 2006).
        out["idio_vol_63"] = resid.rolling(63).std()

    # --- Acceleration / shape features
    out["mom_acc_3_12"] = out["mom_3m"] - out["mom_12m_1m"] / 4.0
    out["vol_change"]   = out["vol_22"] / out["vol_63"].replace(0, np.nan)

    # --- Cross-sectional ranks (often more useful for trees than raw values) ---
    rank_cols = ["mom_1m", "mom_3m", "mom_6m", "mom_12m_1m", "mom_1w_rev",
                 "vol_22", "vol_63", "vol_252", "dev_22", "ma_50_200",
                 "resid_mom_12m_1m", "idio_vol_63",
                 "mom_acc_3_12", "vol_change"]
    for c in rank_cols:
        if c in out:
            out[f"{c}_xsrank"] = _rank_xs(out[c])

    # --- "Deviation from consensus": rank minus its 63-day rolling mean.
    # When a stock's rank is far above its own recent average, momentum is
    # accelerating; when far below, it's reverting. This is a deviation from
    # the cross-section's collective view of the stock — i.e. the "consensus"
    # the user asked for in feature form.
    consensus_cols = ["mom_12m_1m_xsrank", "resid_mom_12m_1m_xsrank",
                      "mom_3m_xsrank", "vol_252_xsrank", "idio_vol_63_xsrank"]
    for c in consensus_cols:
        if c in out:
            out[f"{c}_dev63"] = out[c] - out[c].rolling(63, min_periods=21).mean()

    # --- Stack to long format ---
    pieces = []
    for name, df in out.items():
        # shift to avoid look-ahead (feature at time t uses data up to t-1)
        df_shifted = df.shift(1)
        s = df_shifted.stack()
        s.name = name
        pieces.append(s)

    feats = pd.concat(pieces, axis=1)
    feats.index.names = ["date", "ticker"]
    return feats
