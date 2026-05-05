"""End-to-end smoke test on synthetic data.

This tests every module without needing yfinance / xgboost. Where xgboost is
needed, we substitute sklearn's HistGradientBoostingClassifier (drop-in for
the pipeline). The notebook itself uses xgboost as the project intends.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from features import build_features
from labeling import triple_barrier_labels
from cv import PurgedKFold, cpcv_split, cpcv_paths
from portfolio import hrp_weights, select_and_allocate
from backtest import backtest_long_only, _rebalance_dates, benchmark_buy_and_hold
from metrics import summarize, intra_portfolio_correlation


# ---------------------------------------------------------------------------
# 1. Generate synthetic OHLCV
# ---------------------------------------------------------------------------
np.random.seed(0)
n_days = 252 * 8  # 8 years of daily data
n_assets = 60

dates = pd.bdate_range("2018-01-02", periods=n_days)

# Generate a single-factor model with idiosyncratic noise.
# Add some momentum (autocorrelation) so the ML model has signal to find.
mkt = np.random.normal(0.0004, 0.012, size=n_days)
beta = np.random.uniform(0.5, 1.6, size=n_assets)
idio_vol = np.random.uniform(0.01, 0.025, size=n_assets)

# Add stock-specific momentum: positive AR(1) on returns
ar = 0.05
rets = np.zeros((n_days, n_assets))
for t in range(n_days):
    base = beta * mkt[t] + np.random.normal(0, idio_vol, size=n_assets)
    if t > 0:
        base = base + ar * rets[t - 1]
    rets[t] = base

# Add stock-specific drift varying over time (regime changes that ML can pick up)
drift = np.cumsum(np.random.normal(0, 0.0005, size=(n_days, n_assets)), axis=0)
rets = rets + np.diff(drift, axis=0, prepend=drift[:1]) * 0.5

prices = 100 * np.exp(np.cumsum(rets, axis=0))
tickers = [f"S{i:03d}" for i in range(n_assets)]
close = pd.DataFrame(prices, index=dates, columns=tickers)
volume = pd.DataFrame(np.random.lognormal(15, 0.5, size=close.shape),
                      index=dates, columns=tickers)
high = close * (1 + np.abs(np.random.normal(0, 0.005, size=close.shape)))
low = close * (1 - np.abs(np.random.normal(0, 0.005, size=close.shape)))

print(f"[1/8] Generated synthetic data: {close.shape}")

# ---------------------------------------------------------------------------
# 2. Features
# ---------------------------------------------------------------------------
mkt_proxy = pd.Series((1 + rets.mean(axis=1)).cumprod() * 100, index=dates)
features = build_features(close, volume=volume, high=high, low=low, market_proxy=mkt_proxy)
print(f"[2/8] Built features: {features.shape}, NaN frac = {features.isna().mean().mean():.3f}")

# ---------------------------------------------------------------------------
# 3. Labels
# ---------------------------------------------------------------------------
labels = triple_barrier_labels(close, horizon=21, pt=2.0, sl=2.0, vol_window=22)
print(f"[3/8] Labels: shape={labels.shape}, value_counts={labels.stack().value_counts().to_dict()}")

# Long format
y = labels.stack().rename("label")
y.index.names = ["date", "ticker"]

# Align features with labels
common = features.index.intersection(y.index)
X = features.loc[common].dropna()
y = y.loc[X.index]

# Binary target for base classifier: did the upper barrier hit first?
y_bin = (y == 1).astype(int)
print(f"[3/8] Aligned X={X.shape}, y_bin pos rate={y_bin.mean():.3f}")

# ---------------------------------------------------------------------------
# 4. Train / val / test split
# ---------------------------------------------------------------------------
split_dates = X.index.get_level_values("date")
test_start = pd.Timestamp("2024-01-01")
train_val_mask = split_dates < test_start
test_mask = ~train_val_mask

X_trval, y_trval = X[train_val_mask], y_bin[train_val_mask]
X_test, y_test = X[test_mask], y_bin[test_mask]

# split train_val into train (80%) and val (20%) by time
trval_dates = X_trval.index.get_level_values("date")
cut = trval_dates.unique()[int(0.8 * trval_dates.nunique())]
train_mask = trval_dates < cut
X_train, y_train = X_trval[train_mask], y_trval[train_mask]
X_val, y_val = X_trval[~train_mask], y_trval[~train_mask]
print(f"[4/8] Train={len(X_train)} Val={len(X_val)} Test={len(X_test)}")

# ---------------------------------------------------------------------------
# 5. CV: PurgedKFold + CPCV (just verify split logic)
# ---------------------------------------------------------------------------
# build event end dates per sample
event_horizon = 21
date_arr = X_trval.index.get_level_values("date")
end_dates = (
    pd.DatetimeIndex(date_arr).to_series().shift(-event_horizon)
    .reindex(date_arr).fillna(date_arr.max())
)
t1 = pd.Series(end_dates.values, index=X_trval.index)

pkf = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.01)
fold_sizes = []
for tr, te in pkf.split(X_trval):
    fold_sizes.append((len(tr), len(te)))
print(f"[5/8] PurgedKFold splits OK: train sizes = {[s[0] for s in fold_sizes]}")

# Test CPCV combinatorics
combos, paths = cpcv_paths(n_groups=6, k_test_groups=2)
expected_combos = 15  # C(6, 2) = 15
expected_paths = 5    # C(5, 1) = 5
assert len(combos) == expected_combos, f"CPCV combos: expected {expected_combos}, got {len(combos)}"
assert paths.shape == (expected_paths, 6), f"CPCV paths shape: expected ({expected_paths}, 6), got {paths.shape}"
print(f"[5/8] CPCV: {expected_combos} splits, {expected_paths} paths — verified")

# Run a few CPCV splits
n_split_iter = 0
for tr, te, grp in cpcv_split(X_trval, t1, n_groups=6, k_test_groups=2, embargo_pct=0.01):
    assert len(set(tr) & set(te)) == 0
    n_split_iter += 1
    if n_split_iter >= 3:
        break
print(f"[5/8] CPCV split iterator OK ({n_split_iter} runs verified)")

# ---------------------------------------------------------------------------
# 6. Two-stage model — using sklearn's HGB as a stand-in (no xgboost in env)
# ---------------------------------------------------------------------------
from sklearn.ensemble import HistGradientBoostingClassifier

base = HistGradientBoostingClassifier(max_iter=200, max_depth=5, learning_rate=0.05,
                                      random_state=42)
base.fit(X_train, y_train)
p_base_val = base.predict_proba(X_val)[:, 1]
p_base_test = base.predict_proba(X_test)[:, 1]
print(f"[6/8] Base model: train acc={base.score(X_train, y_train):.3f} "
      f"val acc={base.score(X_val, y_val):.3f}")

# Meta features
def make_meta_features_local(X, p_base):
    out = X.copy()
    out["p_base"] = p_base
    out["base_side"] = (p_base > 0.55).astype(int)
    out["p_base_rank"] = (
        pd.Series(p_base, index=X.index).groupby(level="date").rank(pct=True).values
    )
    return out

X_val_meta = make_meta_features_local(X_val, p_base_val)
side_val = (p_base_val > 0.55).astype(int)
y_meta = ((side_val == 1) & (y_val.values == 1)).astype(int)
long_mask = side_val == 1
print(f"[6/8] Meta training: {long_mask.sum()} long calls in val, "
      f"{y_meta[long_mask].sum()} correct")

if long_mask.sum() > 50:
    meta = HistGradientBoostingClassifier(max_iter=100, max_depth=4, learning_rate=0.05,
                                          random_state=42)
    meta.fit(X_val_meta[long_mask], y_meta[long_mask])
    X_test_meta = make_meta_features_local(X_test, p_base_test)
    p_meta = meta.predict_proba(X_test_meta)[:, 1]
else:
    p_meta = p_base_test  # fallback
print(f"[6/8] Meta model trained, mean p_meta on test={p_meta.mean():.3f}")

# Bet size
def meta_prob_to_size(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    z = (2 * p - 1) / (2 * np.sqrt(p * (1 - p)))
    return 2 * norm.cdf(z) - 1

bet = meta_prob_to_size(p_meta)
side_test = (p_base_test > 0.55).astype(int)
bet_long = np.clip(bet, 0, 1) * side_test

bet_series = pd.Series(bet_long, index=X_test.index, name="bet")
print(f"[6/8] Test bet sizes: positive frac={(bet_series > 0).mean():.3f}, "
      f"mean (when positive)={bet_series[bet_series > 0].mean():.3f}")

# ---------------------------------------------------------------------------
# 7. Build target weights schedule
# ---------------------------------------------------------------------------
test_dates = close.index[close.index >= test_start]
rebal_dates = _rebalance_dates(test_dates, freq="MS")
print(f"[7/8] Test span: {test_dates[0].date()} to {test_dates[-1].date()}, "
      f"{len(rebal_dates)} rebalances")

# bet_series is indexed (date, ticker) over the test set
bet_unstack = bet_series.unstack("ticker").reindex(test_dates).fillna(0.0)

target_weights = pd.DataFrame(np.nan, index=test_dates, columns=close.columns)

ret_for_hrp = close.pct_change()

top_n = 20
for d in rebal_dates:
    if d not in bet_unstack.index:
        continue
    todays_bets = bet_unstack.loc[d]
    if (todays_bets > 0).sum() < 2:
        continue
    # use trailing 252 days for HRP
    window = ret_for_hrp.loc[:d].iloc[-252:]
    w = select_and_allocate(todays_bets, window, top_n=top_n)
    if len(w) > 0:
        # broadcast w across columns of target_weights row
        target_weights.loc[d, :] = 0.0
        target_weights.loc[d, w.index] = w.values

# need at least one non-NaN row at the start; if not, equal-weight on rebal_dates[0]
if target_weights.dropna(how="all").empty:
    raise RuntimeError("No rebalances scheduled — pipeline produced no longs.")

print(f"[7/8] Scheduled rebalances with positions: "
      f"{(target_weights.sum(axis=1) > 0.5).sum()}")

# ---------------------------------------------------------------------------
# 8. Backtest
# ---------------------------------------------------------------------------
test_prices = close.loc[test_dates]
result = backtest_long_only(
    prices=test_prices,
    target_weights=target_weights,
    initial_capital=20_000_000,
    tc_per_trade=0.001,
)

bench_proxy = pd.Series(
    (1 + rets.mean(axis=1)).cumprod() * 100, index=close.index,
).loc[test_dates]
bench = benchmark_buy_and_hold(bench_proxy)

strat_summary = summarize(result["returns"], result["equity_curve"], "Strategy")
bench_summary = summarize(bench["returns"], bench["equity_curve"], "Benchmark")

# IPC
ipc_strat = intra_portfolio_correlation(result["weights"],
                                        test_prices.pct_change().fillna(0))
print(f"[8/8] Backtest done. NAV start={result['equity_curve'].iloc[0]:,.0f}, "
      f"end={result['equity_curve'].iloc[-1]:,.0f}")
print()
print(pd.concat([strat_summary, bench_summary], axis=1))
print()
print(f"IPC mean = {ipc_strat.mean():.3f}")
print(f"Total turnover (one-way): {result['turnover'].sum():.2f}")
print(f"Average rebalance turnover: {result['turnover'].mean():.3f}")

# Sanity asserts
assert result["weights"].sum(axis=1).max() <= 1.001, "Leverage detected!"
assert result["weights"].min().min() >= -1e-9, "Negative weight!"
print()
print("ALL SMOKE TESTS PASSED.")
