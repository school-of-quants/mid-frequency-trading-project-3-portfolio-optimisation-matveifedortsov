"""
Data loading utilities.

The project requires us to avoid survivorship bias by using the *historical*
S&P 500 membership at each point in time. We load Anthropic-provided
historical components CSV (which gives us, for a sequence of dates, the list
of tickers in the index on that date) and fetch full OHLCV history for the
universe of all tickers that ever appeared.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Historical S&P 500 components
# -----------------------------------------------------------------------------

def load_historical_components(csv_path: str | Path) -> pd.Series:
    """Load historical S&P 500 components.

    The CSV has two columns: `date`, `tickers` (comma-separated).
    We return a Series indexed by date, whose values are *sets* of tickers.

    Args:
        csv_path: Path to the components CSV file.

    Returns:
        pd.Series indexed by datetime, values are frozenset[str] of tickers
        in the index on that date.
    """
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return pd.Series(
        [frozenset(t.split(",")) for t in df["tickers"]],
        index=df["date"],
        name="components",
    )


def universe_at_date(components: pd.Series, date: pd.Timestamp) -> frozenset[str]:
    """Return the S&P 500 universe as of `date` (the most recent snapshot
    on or before `date`)."""
    idx = components.index.searchsorted(date, side="right") - 1
    idx = max(idx, 0)
    return components.iloc[idx]


def all_tickers_ever(components: pd.Series) -> list[str]:
    """Union of all tickers that ever appeared in the index."""
    s: set[str] = set()
    for tickers in components.values:
        s.update(tickers)
    return sorted(s)


# -----------------------------------------------------------------------------
# Price data
# -----------------------------------------------------------------------------

def _normalize_ticker_for_yf(t: str) -> str:
    """yfinance uses '-' instead of '.' in some tickers (e.g. BRK.B -> BRK-B)."""
    return t.replace(".", "-")


def fetch_prices_yf(
    tickers: Iterable[str],
    start: str,
    end: str,
    cache_path: str | Path | None = None,
    batch_size: int = 50,
    sleep_between: float = 0.5,
) -> pd.DataFrame:
    """Fetch OHLCV data via yfinance with on-disk parquet caching.

    The result has a MultiIndex on columns: (Field, Ticker), where Field is one
    of {Open, High, Low, Close, Adj Close, Volume}. Tickers that fail to
    download are silently skipped.

    Args:
        tickers: Iterable of ticker symbols.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD), exclusive.
        cache_path: If provided, parquet file used as a persistent cache.
        batch_size: Number of tickers per yfinance request.
        sleep_between: Seconds to wait between batches (rate limiting).

    Returns:
        Wide DataFrame indexed by trading date, columns = (Field, Ticker).
    """
    import yfinance as yf  # local import keeps dependency optional

    tickers = sorted(set(tickers))
    cache_path = Path(cache_path) if cache_path else None

    if cache_path is not None and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached.columns = pd.MultiIndex.from_tuples(
            [tuple(c.split("|")) for c in cached.columns]
        )
        cached.index = pd.to_datetime(cached.index)
        have = set(cached.columns.get_level_values(1))
        missing = [t for t in tickers if t not in have]
        if not missing:
            return cached.loc[start:end]
        tickers_to_fetch = missing
        print(f"Cache hit for {len(have)} tickers; fetching {len(missing)} more.")
    else:
        cached = None
        tickers_to_fetch = tickers

    chunks = []
    for i in range(0, len(tickers_to_fetch), batch_size):
        batch = tickers_to_fetch[i : i + batch_size]
        yf_batch = [_normalize_ticker_for_yf(t) for t in batch]
        print(f"  Batch {i // batch_size + 1}: {len(batch)} tickers...")
        try:
            data = yf.download(
                yf_batch,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="column",
            )
        except Exception as exc:  # pragma: no cover
            print(f"    Batch failed: {exc}")
            continue
        if data is None or data.empty:
            continue
        # rename back to dotted tickers
        if isinstance(data.columns, pd.MultiIndex):
            new_cols = []
            for field, tk in data.columns:
                # invert the BRK-B -> BRK.B mapping
                orig = tk.replace("-", ".") if tk not in batch else tk
                if orig not in batch:
                    # try direct match first
                    orig = tk if tk in batch else orig
                new_cols.append((field, orig))
            data.columns = pd.MultiIndex.from_tuples(new_cols)
        chunks.append(data)
        time.sleep(sleep_between)

    if not chunks and cached is None:
        raise RuntimeError("No price data could be fetched.")

    if chunks:
        new_data = pd.concat(chunks, axis=1)
    else:
        new_data = None

    if cached is not None and new_data is not None:
        out = pd.concat([cached, new_data], axis=1)
    elif cached is not None:
        out = cached
    else:
        out = new_data

    out = out.sort_index()
    out = out.loc[~out.index.duplicated(keep="last")]

    if cache_path is not None:
        # parquet doesn't love multiindex columns: flatten with '|'
        flat = out.copy()
        flat.columns = ["|".join(c) for c in out.columns]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        flat.to_parquet(cache_path)

    return out.loc[start:end]


def build_active_mask(
    prices: pd.DataFrame,
    components: pd.Series,
) -> pd.DataFrame:
    """Build a boolean mask of (date, ticker) -> True if ticker was in the
    S&P 500 on that date AND has a non-null price.

    This is the point-in-time investible universe filter — critical for
    avoiding survivorship and look-ahead biases.

    Args:
        prices: Adjusted close prices, indexed by date, columns = tickers.
        components: Output of load_historical_components.

    Returns:
        Boolean DataFrame, same shape as prices.
    """
    # for each price date, find the most recent components snapshot
    snap_dates = components.index
    pos = np.searchsorted(snap_dates, prices.index, side="right") - 1
    pos = np.clip(pos, 0, len(snap_dates) - 1)

    mask = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    for i, p in enumerate(pos):
        members = components.iloc[p]
        present = [c for c in prices.columns if c in members]
        mask.iloc[i][present] = True

    # also require non-null price
    mask &= prices.notna()
    return mask
