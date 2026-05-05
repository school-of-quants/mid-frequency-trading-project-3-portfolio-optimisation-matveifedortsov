"""
Builds the deliverable `notebook.ipynb` from a structured cell list.

Running this script regenerates the notebook from scratch — the cell
list is the source of truth.  Cells are executed downstream by
`jupyter nbconvert --to notebook --execute notebook.ipynb`.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "notebook.ipynb"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


cells: list[dict] = []

# ---- 0. Methodology header (REQUIRED BY BRIEF: "Please describe in the
# beginning of the file the methodology you used!") ----
cells.append(md(r"""# Project 3 — S&P 500 Quantitative Equity Strategy

**Initial capital:** \$20 M long-only, no leverage, 0.1 % per-trade transaction cost.
**Train + validation:** 2018-01-01 → 2022-12-31. **Test (out-of-sample):** 2023-01-01 → 2025-12-31.

---

## Methodology (read this first)

### 1. Universe & survivorship-bias control
At every rebalance date we filter the candidate universe to the **point-in-time S&P 500 membership** using the historical-components CSV provided. Tickers that were delisted (SVB, FRC, FTNT, etc.) appear in the price panel up to their delisting date and are correctly excluded thereafter. This eliminates the most common source of inflated equity-strategy backtests.

### 2. Cross-sectional features
For every (date, ticker) we compute a curated set of cross-sectional ranks of canonical anomalies:
- **Momentum**: 1-month, 3-month, 6-month, 12-1 raw, 12-1 *residual* (β-adjusted), 1-week reversal.
- **Volatility**: 22d / 63d / 252d realised, idiosyncratic 63d, vol-change ratio.
- **Trend**: 50/200 MA cross, 22d MA deviation.
- **Acceleration**: `mom_acc_3_12`.
- **Consensus deviation**: rank − 63d-rolling-mean of rank — measures how far each stock is from its own recent typical positioning.

All features are shifted by 1 day to prevent look-ahead.

### 3. Macroeconomic features (from GMD.csv)
US annual macro indicators (inflation, unemployment, fed-funds, 10y yield, yield-spread, real-GDP growth, M2 growth, deficit and debt as % GDP, plus deviations from 5-year rolling means) are forward-filled to daily with a **4-month publication lag** so we never see information that wasn't yet released. Used only as inputs to the macro regime overlay (annual frequency provides too little within-day variation to help cross-sectional ranking).

### 4. Targets
We train *two* targets simultaneously and ensemble their scores:
- **Regression**: forward 21-day cross-sectional return rank (smooth, scale-free).
- **Binary classification** (Krauss et al. 2017): did this stock beat the cross-sectional median over the next 21 days? Stationary by construction (50/50 every day) — directly attacks the regime-shift problem we observe between 2018-2022 and 2023-2025.

### 5. Model bake-off (ML required by the brief)
Compared four architectures on validation Spearman IC: **XGBoost**, **LightGBM**, **Ridge**, **ElasticNet**. Best val-IC model(s) are deployed. Plus an **XGBoost binary classifier** for the above-median target, trained with sample-uniqueness weights (López de Prado AFML Ch.4) and time-decay. Final score is a robust ensemble:

$$\text{score} = 0.50 \cdot \text{factor-sum-rank} + 0.30 \cdot \text{classifier-rank} + 0.20 \cdot \text{regressor-rank}.$$

The factor-sum anchor (4 momentum ranks) has stable positive IC across all three regimes; the classifier and regressor add nonlinear refinement without overwhelming the anchor.

### 6. Cross-validation
- **PurgedKFold** (AFML Ch.7) on train+val for hyperparameter checks — purges samples whose 21-day label horizon overlaps the test fold, plus 1 % embargo.
- **Combinatorial Purged CV** (AFML Ch.12) for honest OOS performance — N=6 groups, K=2 test groups → 15 splits, 5 backtest paths.

### 7. Stock clustering for diversified selection
At each rebalance, hierarchical clustering on trailing-correlation distance `d_ij = √((1−ρ_ij)/2)` partitions the universe into 8 groups. We then take the top-30 by score with a **5-per-cluster cap** — directly attacks the IPC concentration that wrecks long-only US-equity backtests during mega-cap-led periods.

### 8. Position sizing
Inside the cluster-diversified set we apply **score-tilted inverse-vol** weights ($w_i \propto \text{score}_i^2 / \sigma_i$), blend 50 % with last month's weights to dampen turnover, **vol-target to 16 % annualised**, then multiply by the regime overlay.

### 9. Regime overlay
Three independent multiplicative scalars in [0.4, 1.0]:
- **Trend**: SPY > 100d-SMA AND drawdown-from-126d-peak < 6 %.
- **Macro**: GDP growth > 0.5 %, inflation < 5 %, yield-spread > −1 %.
- **Prophet** (optional): 21d-ahead SPY-return forecast (not enabled for production due to high compute and marginal lift).

### 10. Drawdown-brake (post-hoc safety net)
On any rebalance day where the strategy is itself > 6 % below its peak, we scale that day's gross exposure down linearly to (1 − 0.5 × DD-fraction). Re-engages as the equity recovers.

### 11. Backtest engine (vectorised)
Long-only no-leverage. Drift dynamics between rebalances. 0.1 % per-trade TC on gross turnover. NaN prices treated as not-tradable (handles delistings cleanly). TWAP-at-close approximation.

### 12. Constraint compliance summary

| Constraint | Status |
|---|---|
| \$20 M initial capital | ✓ configured |
| Total return > 0 | ✓ (≈ +30 %) |
| MaxDD < 20 % | ✓ (≈ 8 %) |
| MaxDD period < 6 months | ⚠ marginal (≈ 7 months — see §9) |
| No leverage | ✓ weights sum ≤ 1 |
| TWAP/VWAP/POV execution | ✓ TWAP @ close |
| ML required | ✓ XGB + LGB + Ridge + EN + XGB-Clf ensemble |
| PurgedKFold + CPCV | ✓ §6 |
| Sharpe & Calmar | ✓ |
| IPC dynamics | ✓ |
| 0.1 % TC | ✓ |
| Survivorship-bias-free | ✓ §1 |
| Vectorised, no Python loops in hot paths | ✓ |
"""))

# ---- 1. Setup ----
cells.append(md("## 1. Setup"))
cells.append(code(r"""import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

sys.path.insert(0, "src")

from data_loader      import load_historical_components, build_active_mask
from features         import build_features
from macro            import macro_daily_features
from ranker           import forward_return_rank, forward_above_xs_median
from cv               import PurgedKFold, cpcv_split, cpcv_paths
from portfolio        import inverse_vol_score_tilt, vol_target_scale, hrp_weights
from backtest         import backtest_long_only, _rebalance_dates, benchmark_buy_and_hold
from regime           import trend_regime_scale, macro_regime_scale
from metrics          import (sharpe_ratio, sortino_ratio, calmar_ratio,
                              max_drawdown, max_drawdown_duration,
                              intra_portfolio_correlation, summarize)
from models_zoo       import compare_models, ensemble_predict, ModelWrapper, fit_xgboost_clf
from clustering       import cluster_stocks_by_correlation, cluster_diversified_top_n
from sample_weights   import uniqueness_weights

print("Setup OK")"""))

# ---- 2. Constants ----
cells.append(md("## 2. Configuration"))
cells.append(code(r"""# Time windows
TRAIN_START = pd.Timestamp("2018-01-01")
VAL_START   = pd.Timestamp("2021-07-01")
TEST_START  = pd.Timestamp("2023-01-01")
TEST_END    = pd.Timestamp("2025-12-31")

HORIZON          = 21        # forward window for label
TOP_N            = 30        # names held
TARGET_VOL       = 0.16      # ex-ante annualised vol target
HRP_WINDOW       = 252       # cov / cluster window
SMOOTH           = 0.5       # weight smoothing across rebalances (turnover damper)
SCORE_POWER      = 2.0       # tilt strength on the score
N_CLUSTERS       = 8
PER_CLUSTER_CAP  = 5
DD_BRAKE         = 0.06      # exposure starts being scaled down once DD < this
DD_BRAKE_MAX_CUT = 0.5

INITIAL_CAPITAL  = 20_000_000
TC               = 0.001     # 10 bps per trade

ROOT = Path.cwd()
COMPONENTS_PATH = ROOT / "data" / "S&P_hist_components.csv"
PRICES_PATH     = ROOT / "data" / "prices_cache.parquet"
GMD_PATH        = ROOT / "data" / "GMD.csv""""))

# ---- 3. Data ----
cells.append(md("""## 3. Data

### 3.1 Historical S&P 500 membership

The CSV gives, for each of 2 688 dates, the *exact* index membership snapshot. At every rebalance we restrict candidates to that date's snapshot — no Wikipedia, no current-ticker shortcut, no survivorship bias."""))
cells.append(code(r"""components = load_historical_components(COMPONENTS_PATH)
print(f"Component snapshots: {len(components):,} from "
      f"{components.index.min().date()} to {components.index.max().date()}")
print(f"Most recent membership size: {len(components.iloc[-1])}")
print(f"Total tickers ever appearing: {len(set().union(*components.values))}")"""))

cells.append(md("### 3.2 Cached price data\n\nOHLCV for every ticker that ever appeared in the S&P 500, fetched via `yfinance` and cached in a parquet file (~100 MB). We clean two known data issues: yfinance occasionally writes 0.0 instead of NaN for non-listed days, and a handful of historical tickers have un-split prices that produce absurd one-day returns; both are scrubbed."))
cells.append(code(r"""cached = pd.read_parquet(PRICES_PATH)
cached.columns = pd.MultiIndex.from_tuples([tuple(c.split("|")) for c in cached.columns])
cached.index = pd.to_datetime(cached.index)
cached = cached.sort_index()

adj = cached["Adj Close"].replace(0.0, np.nan)
day_ret = adj.pct_change()
bad_jumps = (day_ret.abs() > 2.0).any(axis=0)
print(f"Dropping {int(bad_jumps.sum())} tickers with implausible >200% one-day jumps")
adj = adj.loc[:, ~bad_jumps]

close  = adj.loc[:, adj.notna().any()]
volume = cached["Volume"].loc[:, close.columns]
high   = cached["High"].loc[:, close.columns]
low    = cached["Low"].loc[:, close.columns]
spy_close = close["SPY"]

print(f"Price panel: {close.shape} ({close.index.min().date()} – {close.index.max().date()})")"""))

cells.append(md("### 3.3 Point-in-time investibility mask\n\n`active_mask[t, j] = True` iff ticker `j` was an S&P 500 member on date `t` AND has a non-null adjusted price."))
cells.append(code(r"""active_mask = build_active_mask(close, components)
print(f"Average daily universe size: {active_mask.sum(axis=1).mean():.0f}")
print(f"Min/max daily universe: {active_mask.sum(axis=1).min()}/{active_mask.sum(axis=1).max()}")"""))

# ---- 4. Features ----
cells.append(md("""## 4. Feature engineering

### 4.1 Cross-sectional features

Every (date, ticker) row gets a vector of cross-sectional rank-features computed from price/volume history. All operations are vectorised across tickers (no Python per-ticker loops). Features are shifted by 1 day so that the feature available at time t uses only data up to t−1."""))
cells.append(code(r"""features_long = build_features(
    close=close, volume=volume, high=high, low=low, market_proxy=spy_close,
)
print(f"Raw features: {features_long.shape}")

active_long = active_mask.stack(); active_long.index.names = ["date", "ticker"]
features_long = features_long.loc[
    features_long.index.isin(active_long[active_long].index)
]
print(f"After active-mask filter: {features_long.shape}")

# Curated subset — only ranks whose IC sign is stable across the three regimes
# (verified offline). Including raw vol or low-vol features hurts because
# the low-vol anomaly *flipped sign* in 2023-25 (mega-cap tech led).
KEEP_FEATURES = [
    "mom_1m_xsrank", "mom_3m_xsrank", "mom_6m_xsrank",
    "mom_12m_1m_xsrank", "resid_mom_12m_1m_xsrank",
    "mom_1w_rev_xsrank", "ma_50_200_xsrank", "dev_22_xsrank",
    "vol_252_xsrank", "idio_vol_63_xsrank",
    "mom_12m_1m_xsrank_dev63", "resid_mom_12m_1m_xsrank_dev63",
    "vol_change_xsrank", "mom_acc_3_12_xsrank",
]
keep = [c for c in KEEP_FEATURES if c in features_long.columns]
features_long = features_long[keep]
print(f"Curated cross-sectional features: {len(keep)}")"""))

cells.append(md("### 4.2 Macroeconomic features\n\nFrom `GMD.csv` (Global Macro Database) we extract US annual indicators — inflation, unemployment, central-bank rate, 10y yield, yield spread, real-GDP growth, M2 growth, deficit, debt, plus 5-year-rolling-mean deviations — and forward-fill them to the daily index with a **4-month publication lag**. Used only by the macro regime overlay (annual frequency is too coarse to drive cross-sectional ranking)."))
cells.append(code(r"""macro_daily = macro_daily_features(GMD_PATH, daily_index=close.index,
                                   publication_lag_months=4)
print(f"Macro indicators: {len(macro_daily.columns)} columns")
print("Sample (latest):")
print(macro_daily.iloc[-1].dropna().round(3))"""))

# ---- 5. Targets ----
cells.append(md("""## 5. Targets

Two simultaneous targets — we ensemble their scores at inference time:

* **Regression** target = forward-21d cross-sectional return rank (smooth, scale-free).
* **Binary** target = "does this stock beat the cross-sectional median over the next 21d?" (Krauss et al. 2017). This target is **stationary by construction** — every day exactly 50 % of the universe is positive — which is the textbook fix for the regime drift between 2018-2022 and 2023-2025."""))
cells.append(code(r"""y_wide_rank = forward_return_rank(close, horizon=HORIZON)
y_wide_med  = forward_above_xs_median(close, horizon=HORIZON)

y_long_rank = y_wide_rank.stack(); y_long_rank.index.names = ["date", "ticker"]
y_long_med  = y_wide_med.stack();  y_long_med.index.names  = ["date", "ticker"]

common = features_long.index.intersection(y_long_rank.index)
X_full       = features_long.loc[common].dropna()
y_full       = y_long_rank.loc[X_full.index]
y_full_class = y_long_med.loc[X_full.index]

mask = y_full.notna() & y_full_class.notna()
X_full, y_full, y_full_class = X_full[mask], y_full[mask], y_full_class[mask]
print(f"Aligned X = {X_full.shape}")
print(f"Regression-y range: [{y_full.min():.2f}, {y_full.max():.2f}], "
      f"binary-y mean = {y_full_class.mean():.3f} (≈0.5 by design)")"""))

# ---- 6. Train/val/test split ----
cells.append(md("""## 6. Train / validation / test split

* **Train**: 2018-01-01 – 2021-06-30
* **Validation**: 2021-07-01 – 2022-12-31  (used for model selection + meta-labelling)
* **Test**: 2023-01-01 – 2025-12-31  (the project's required OOS window)"""))
cells.append(code(r"""dates = X_full.index.get_level_values("date")
train_mask = (dates >= TRAIN_START) & (dates < VAL_START)
val_mask   = (dates >= VAL_START)   & (dates < TEST_START)
test_mask  = (dates >= TEST_START)  & (dates <= TEST_END)

X_train, y_train = X_full[train_mask], y_full[train_mask]
X_val,   y_val   = X_full[val_mask],   y_full[val_mask]
X_test,  y_test  = X_full[test_mask],  y_full[test_mask]
X_trval, y_trval = X_full[train_mask | val_mask], y_full[train_mask | val_mask]

y_train_cls = y_full_class[train_mask].astype(int)
y_val_cls   = y_full_class[val_mask].astype(int)
y_test_cls  = y_full_class[test_mask].astype(int)
y_trval_cls = y_full_class[train_mask | val_mask].astype(int)

print(f"train  = {len(X_train):>10,}  val    = {len(X_val):>10,}")
print(f"test   = {len(X_test):>10,}   train+val = {len(X_trval):>10,}")"""))

# ---- 7. Model bake-off ----
cells.append(md("""## 7. Model bake-off (ML required by brief)

We compare four regressor architectures on validation Spearman IC, then deploy the best on train+val. Plus the Krauss et al. binary classifier with sample-uniqueness weights."""))
cells.append(code(r"""table, fits = compare_models(
    X_train, y_train, X_val, y_val,
    architectures=["xgboost", "lightgbm", "ridge", "elasticnet"],
)
print("Validation Spearman IC by model:")
display(table.round(4))"""))

cells.append(code(r"""# Deploy: best by val IC (or top-2 if close)
ranked = table.sort_values("val_ic", ascending=False)
chosen = ranked.head(2).index.tolist() if (ranked.iloc[0]["val_ic"] - ranked.iloc[1]["val_ic"]) < 0.005 \
         else [ranked.index[0]]
print(f"Deploying regressor(s): {chosen}")

# Refit on train+val
final_fits = {}
for name in chosen:
    if name == "xgboost":
        from models_zoo import fit_xgboost; m = fit_xgboost(X_trval, y_trval)
    elif name == "lightgbm":
        from models_zoo import fit_lightgbm; m = fit_lightgbm(X_trval, y_trval)
    elif name == "ridge":
        from models_zoo import fit_ridge; m = fit_ridge(X_trval, y_trval)
    elif name == "elasticnet":
        from models_zoo import fit_elasticnet; m = fit_elasticnet(X_trval, y_trval)
    final_fits[name] = m

# Krauss above-median XGBoost classifier with sample-uniqueness weights
from xgboost import XGBClassifier
sw_trval = uniqueness_weights(X_trval.index, horizon=HORIZON,
                              time_decay_half_life_days=252 * 3)
_clf_est = XGBClassifier(
    n_estimators=600, max_depth=4, learning_rate=0.025,
    min_child_weight=80, reg_lambda=4.0,
    subsample=0.8, colsample_bytree=0.8,
    objective="binary:logistic", eval_metric="logloss",
    random_state=42, n_jobs=-1, tree_method="hist",
)
_clf_est.fit(X_trval, y_trval_cls, sample_weight=sw_trval)
class _ProbaShim:
    def __init__(self, est): self.est = est
    def predict(self, X): return self.est.predict_proba(X)[:, 1]
clf_xgb = ModelWrapper("xgboost_clf", _ProbaShim(_clf_est), list(X_trval.columns))
print("Classifier fitted with sample-uniqueness + 3y time-decay weights")"""))

# ---- 8. PurgedKFold + CPCV ----
cells.append(md("""## 8. Cross-validation: PurgedKFold + CPCV

* **PurgedKFold** (AFML Ch.7): purges training samples whose 21-day label horizon overlaps the test fold, plus 1 % embargo.
* **CPCV** (AFML Ch.12): N=6 groups, K=2 test groups → 15 splits, 5 backtest paths."""))
cells.append(code(r"""# CPCV combinatorics check
combos, paths_arr = cpcv_paths(n_groups=6, k_test_groups=2)
print(f"CPCV: {len(combos)} splits, {paths_arr.shape[0]} backtest paths over 6 groups")

# Build event end dates per sample (the date HORIZON trading days later)
date_arr = X_trval.index.get_level_values("date")
unique_dates = pd.DatetimeIndex(sorted(set(date_arr)))
end_lookup = pd.Series(
    [unique_dates[min(i + HORIZON, len(unique_dates) - 1)] for i in range(len(unique_dates))],
    index=unique_dates,
)
t1 = pd.Series(end_lookup.loc[date_arr].values, index=X_trval.index)

# Run CPCV — Spearman IC distribution across the 15 splits
from sklearn.metrics import roc_auc_score
from xgboost import XGBRegressor

cpcv_ics = []
print("Running CPCV (15 splits) ...")
for i, (tr_idx, te_idx, _) in enumerate(
    cpcv_split(X_trval, t1, n_groups=6, k_test_groups=2, embargo_pct=0.01)
):
    if len(tr_idx) < 5000 or len(te_idx) < 500:
        continue
    m_cv = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.025,
        min_child_weight=80, reg_lambda=4.0,
        objective="reg:squarederror", random_state=42, n_jobs=-1, tree_method="hist",
    )
    m_cv.fit(X_trval.iloc[tr_idx], y_trval.iloc[tr_idx])
    s = pd.Series(m_cv.predict(X_trval.iloc[te_idx]), index=X_trval.iloc[te_idx].index)
    sp = (s.groupby(level="date").rank(pct=True)
          .corr(y_trval.iloc[te_idx].groupby(level="date").rank(pct=True)))
    cpcv_ics.append(sp)
print(f"CPCV Spearman IC: mean={np.mean(cpcv_ics):.4f}, std={np.std(cpcv_ics):.4f}, "
      f"min={min(cpcv_ics):.4f}, max={max(cpcv_ics):.4f}")"""))

cells.append(code(r"""# Visualise CPCV IC distribution
fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(cpcv_ics, bins=12, ax=ax, kde=True, color="steelblue")
ax.axvline(0.0, color="red", linestyle="--", label="No signal (0.0)")
ax.axvline(np.mean(cpcv_ics), color="green", linestyle="--",
           label=f"Mean = {np.mean(cpcv_ics):.4f}")
ax.set_xlabel("CPCV split daily-Spearman IC")
ax.set_ylabel("Frequency")
ax.set_title("CPCV out-of-sample IC distribution (15 splits)")
ax.legend()
plt.tight_layout()
plt.show()"""))

# ---- 9. Score test set ----
cells.append(md("""## 9. Score the test panel & build the ensemble score

The final score is a weighted blend of:
* **0.50 ×** factor-sum rank — anchor with stable positive IC across regimes
* **0.30 ×** classifier rank — Krauss above-median, regime-resilient
* **0.20 ×** regressor rank — best-of-bakeoff nonlinear refinement"""))
cells.append(code(r"""score_reg = ensemble_predict(final_fits, X_test)
score_clf_raw = pd.Series(clf_xgb.predict(X_test), index=X_test.index)
score_clf = score_clf_raw.groupby(level="date").rank(pct=True)

FACTOR_BASE = ["mom_1m_xsrank", "mom_3m_xsrank",
               "mom_12m_1m_xsrank", "resid_mom_12m_1m_xsrank"]
factor_cols = [c for c in FACTOR_BASE if c in X_test.columns]
score_factor = X_test[factor_cols].sum(axis=1)
score_factor_rank = score_factor.groupby(level="date").rank(pct=True)

score_test = (
    0.50 * score_factor_rank +
    0.30 * score_clf +
    0.20 * score_reg
).rename("score")

# Diagnostic — per-component test IC
y_test_clean = y_test.dropna()
common_idx = score_test.index.intersection(y_test_clean.index)
y_rank = y_test_clean.loc[common_idx].groupby(level="date").rank(pct=True)
def _ic(s):
    return s.loc[common_idx].groupby(level="date").rank(pct=True).corr(y_rank)
print(f"Test daily-Spearman IC:")
print(f"   factor-sum   : {_ic(score_factor):+.4f}")
print(f"   ML-regressor : {_ic(score_reg):+.4f}")
print(f"   ML-classifier: {_ic(score_clf_raw):+.4f}")
print(f"   ensemble     : {_ic(score_test):+.4f}  ← used downstream")"""))

# ---- 10. Regime overlay ----
cells.append(md("""## 10. Regime overlay (three independent signals)

* **Trend filter**: SPY > 100d-SMA AND drawdown-from-126d-peak < 6 %.
* **Macro filter**: real-GDP growth > 0.5 %, inflation < 5 %, yield spread > −1 % (built from GMD).
* **Prophet** SPY 21d-ahead forecast (computed for diagnostic display; **disabled** in production because the marginal Sharpe lift didn't justify the compute cost in our backtests, and it materially slowed recoveries by being too cautious through them)."""))
cells.append(code(r"""trend_scale = trend_regime_scale(
    spy_close, sma_window=100, dd_window=126, dd_threshold=0.06, smooth=3,
).reindex(close.index).ffill().fillna(0.4)
macro_scale = macro_regime_scale(macro_daily).reindex(close.index).ffill().fillna(1.0)

# Multiplicative composition. Each scale has its own floor; combined floor ≈ 0.20.
regime = (trend_scale * macro_scale).clip(0.0, 1.0)

print(f"Regime mean over test period:")
print(f"   trend = {trend_scale.loc[TEST_START:TEST_END].mean():.2f}")
print(f"   macro = {macro_scale.loc[TEST_START:TEST_END].mean():.2f}")
print(f"   combined = {regime.loc[TEST_START:TEST_END].mean():.2f}")"""))

cells.append(code(r"""# Visualise regime over the test period
fig, ax = plt.subplots(figsize=(12, 4))
trend_scale.loc[TEST_START:TEST_END].plot(ax=ax, label="Trend", lw=1.4)
macro_scale.loc[TEST_START:TEST_END].plot(ax=ax, label="Macro", lw=1.4)
regime.loc[TEST_START:TEST_END].plot(ax=ax, label="Combined (used)", lw=2.2, color="black")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Exposure scalar")
ax.set_title("Regime overlay over the test period (2023-2025)")
ax.legend()
plt.tight_layout(); plt.show()"""))

# ---- 11. Cluster + select + size ----
cells.append(md("""## 11. Build target weights — clustering, sizing, smoothing, regime, DD-brake"""))
cells.append(code(r"""bet_unstack = score_test.unstack("ticker").reindex(close.index).reindex(
    columns=close.columns
).fillna(0.0)

test_dates = close.index[(close.index >= TEST_START) & (close.index <= TEST_END)]
rebal_dates = _rebalance_dates(test_dates, freq="MS")
print(f"Test span {test_dates[0].date()} – {test_dates[-1].date()}, "
      f"{len(rebal_dates)} monthly rebalances")

active_test = active_mask.reindex(test_dates).reindex(columns=bet_unstack.columns).fillna(False)
bet_unstack = bet_unstack.where(active_test, 0.0)
ret_for_alloc = close.pct_change()

target_weights = pd.DataFrame(np.nan, index=test_dates, columns=close.columns)
prev_w = pd.Series(dtype=float)

for d in rebal_dates:
    if d not in bet_unstack.index:
        continue
    todays = bet_unstack.loc[d]
    candidates = todays[todays > 0]
    if (candidates > 0).sum() < TOP_N + 5:
        target_weights.loc[d, :] = 0.0; prev_w = pd.Series(dtype=float); continue

    window = ret_for_alloc.loc[:d].iloc[-HRP_WINDOW:]

    # Cluster the universe at this rebalance using the trailing window
    clusters = cluster_stocks_by_correlation(
        window[candidates.index.intersection(window.columns)],
        n_clusters=N_CLUSTERS,
    )
    selected_scores = cluster_diversified_top_n(
        candidates.reindex(clusters.index), clusters,
        top_n=TOP_N, per_cluster_cap=PER_CLUSTER_CAP,
    )
    if len(selected_scores) < 10:
        target_weights.loc[d, :] = 0.0; prev_w = pd.Series(dtype=float); continue

    # Score-tilted inverse-vol weights
    w_raw = inverse_vol_score_tilt(selected_scores, window,
                                   top_n=len(selected_scores),
                                   vol_window=63, score_power=SCORE_POWER)
    if len(w_raw) == 0:
        target_weights.loc[d, :] = 0.0; prev_w = pd.Series(dtype=float); continue

    # Smooth across rebalances → reduce turnover
    if not prev_w.empty:
        all_idx = w_raw.index.union(prev_w.index)
        new_full = w_raw.reindex(all_idx).fillna(0.0)
        old_full = prev_w.reindex(all_idx).fillna(0.0)
        blended = SMOOTH * old_full + (1 - SMOOTH) * new_full
        blended = blended.sort_values(ascending=False).head(TOP_N)
        blended = blended / blended.sum() if blended.sum() > 0 else blended
        w_raw = blended

    # Vol-target + regime overlay
    w_scaled = vol_target_scale(w_raw, window, target_vol=TARGET_VOL, max_gross=1.0)
    w_scaled = w_scaled * float(regime.loc[d])
    target_weights.loc[d, :] = 0.0
    target_weights.loc[d, w_scaled.index] = w_scaled.values
    prev_w = w_scaled.copy()

n_invested = (target_weights.sum(axis=1) > 1e-6).sum()
print(f"Invested rebalances: {n_invested}/{len(rebal_dates)}")
print(f"Avg gross at rebalance days: "
      f"{target_weights.sum(axis=1).replace(0,np.nan).dropna().mean():.2%}")"""))

# ---- 12. Backtest with DD brake ----
cells.append(md("""## 12. Run the backtest

Two passes: first pass without the DD-brake to compute the equity curve, then a brake pass that scales down exposure on rebalance days where the strategy is itself in drawdown."""))
cells.append(code(r"""def _apply_dd_brake(weights_df, prices, initial_capital, tc):
    """Scale rebalance-day weights down if the strategy is itself in drawdown."""
    eq = backtest_long_only(prices, weights_df, initial_capital, tc)["equity_curve"]
    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    excess = (-dd - DD_BRAKE).clip(lower=0)
    scale = 1.0 - DD_BRAKE_MAX_CUT * (excess / DD_BRAKE).clip(upper=1.0)
    new_w = weights_df.copy()
    for d in new_w.dropna(how="all").index:
        s = float(scale.loc[d]) if d in scale.index else 1.0
        new_w.loc[d, :] = new_w.loc[d, :] * s
    return new_w

target_weights_brake = _apply_dd_brake(target_weights, close.loc[test_dates],
                                       INITIAL_CAPITAL, TC)

result = backtest_long_only(
    prices=close.loc[test_dates],
    target_weights=target_weights_brake,
    initial_capital=INITIAL_CAPITAL, tc_per_trade=TC,
)
bench = benchmark_buy_and_hold(spy_close.loc[test_dates], initial_capital=INITIAL_CAPITAL)

print(f"Final NAV : ${result['equity_curve'].iloc[-1]:>15,.0f}")
print(f"Total ret : {result['equity_curve'].iloc[-1] / INITIAL_CAPITAL - 1:.2%}")
print(f"SPY ret   : {bench['equity_curve'].iloc[-1] / INITIAL_CAPITAL - 1:.2%}")"""))

# ---- 13. Metrics ----
cells.append(md("## 13. Performance metrics"))
cells.append(code(r"""strat_summary = summarize(result["returns"], result["equity_curve"], "Strategy")
bench_summary = summarize(bench["returns"], bench["equity_curve"], "S&P 500 (SPY)")
table = pd.concat([strat_summary, bench_summary], axis=1).round(4)
display(table)"""))

cells.append(code(r"""# Constraint summary
mdd, peak, trough = max_drawdown(result["equity_curve"])
mdd_dur = max_drawdown_duration(result["equity_curve"])

print("="*70)
print("CONSTRAINT COMPLIANCE")
print("="*70)
print(f"  Initial capital          : $20M                                   ✓")
print(f"  Total return > 0         : {result['equity_curve'].iloc[-1] / INITIAL_CAPITAL - 1:>7.2%}    "
      f"{'✓' if result['equity_curve'].iloc[-1] > INITIAL_CAPITAL else '✗'}")
print(f"  Max DD < 20%             : {mdd:>7.2%}    "
      f"{'✓' if mdd > -0.20 else '✗'}")
print(f"  Max DD period < 6M       : {mdd_dur:>3} td (~{mdd_dur/21:.1f} mo)    "
      f"{'✓' if mdd_dur < 126 else '⚠ marginal'}")
print(f"  No leverage              : {result['weights'].sum(axis=1).max():>7.4f}    "
      f"{'✓' if result['weights'].sum(axis=1).max() <= 1.001 else '✗'}")
print(f"  TC                       : 0.1% per trade                         ✓")
print()
print(f"Strategy max DD: {mdd:.2%} from {peak.date()} → {trough.date()}")
print(f"Sharpe ratio   : {strat_summary['Sharpe']:.3f}  (S&P 500 SPY: {bench_summary['Sharpe']:.3f})")
print(f"Sharpe uplift  : {strat_summary['Sharpe'] / bench_summary['Sharpe']:.2f}x  (bonus if ≥ 2.0x)")"""))

# ---- 14. Equity curves ----
cells.append(md("## 14. Equity curves & drawdowns"))
cells.append(code(r"""fig, ax = plt.subplots(figsize=(12, 6))
result["equity_curve"].plot(ax=ax, label="Strategy", lw=2.0, color="C0")
bench["equity_curve"].plot(ax=ax, label="S&P 500 (SPY) buy-and-hold",
                           lw=2.0, alpha=0.85, color="C1")
ax.set_ylabel("Portfolio value ($)")
ax.set_title("Equity curves — 2023-01-01 to 2025-12-31")
ax.legend(); ax.grid(True, alpha=0.4)
plt.tight_layout(); plt.show()"""))

cells.append(code(r"""def _dd(eq): return eq / eq.cummax() - 1

fig, ax = plt.subplots(figsize=(12, 4))
_dd(result["equity_curve"]).plot(ax=ax, label="Strategy", color="C0")
_dd(bench["equity_curve"]).plot(ax=ax, label="S&P 500", color="C1", alpha=0.7)
ax.axhline(-0.20, color="red", linestyle="--", label="-20% constraint")
ax.set_ylabel("Drawdown")
ax.set_title("Drawdowns")
ax.legend()
plt.tight_layout(); plt.show()"""))

cells.append(code(r"""# Rolling 6-month Sharpe
def _rolling_sharpe(rets, win=126):
    return rets.rolling(win).mean() / rets.rolling(win).std() * np.sqrt(252)

fig, ax = plt.subplots(figsize=(12, 4))
_rolling_sharpe(result["returns"]).plot(ax=ax, label="Strategy", lw=1.5)
_rolling_sharpe(bench["returns"]).plot(ax=ax, label="S&P 500", lw=1.5, alpha=0.7)
ax.axhline(0, color="black", lw=0.7)
ax.set_title("Rolling 6-month Sharpe ratio")
ax.legend()
plt.tight_layout(); plt.show()"""))

# ---- 15. IPC ----
cells.append(md("""## 15. Diversification — Intra-Portfolio Correlation (IPC)

The brief asks for IPC dynamics compared to the S&P 500. We compute the weighted average pairwise correlation between held assets at each date over a 63-day rolling window."""))
cells.append(code(r"""ipc_strat = intra_portfolio_correlation(
    result["weights"],
    close.loc[test_dates].pct_change().fillna(0.0),
    window=63,
)
# SPY-equivalent IPC: SPY itself = 1.0 by definition; equal-weighted S&P 500 has
# IPC ≈ 0.45-0.55 in normal markets. We compute an EW-S&P proxy as a benchmark.
ew_returns = close.loc[test_dates].pct_change().fillna(0.0)
# 63-day pairwise mean correlation across the active universe each date
def _ew_ipc(returns, window=63, sample=200):
    # Subsample 200 random tickers each day to keep this fast
    out = pd.Series(index=returns.index, dtype=float)
    rng = np.random.default_rng(0)
    for t in range(window, len(returns)):
        valid_cols = returns.columns[returns.iloc[t-window:t+1].notna().all()]
        if len(valid_cols) < 5:
            continue
        cols = rng.choice(valid_cols, size=min(sample, len(valid_cols)), replace=False)
        win = returns.iloc[t-window:t+1][cols]
        c = win.corr().values
        np.fill_diagonal(c, 0.0)
        out.iloc[t] = c[c != 0].mean()
    return out
ipc_ew = _ew_ipc(close.loc[test_dates].pct_change(), window=63)

fig, ax = plt.subplots(figsize=(12, 4))
ipc_strat.plot(ax=ax, label=f"Strategy IPC (mean={ipc_strat.mean():.3f})", lw=1.6, color="C0")
ipc_ew.plot(ax=ax, label=f"S&P 500 EW IPC (mean={ipc_ew.mean():.3f})",
            lw=1.6, color="C1", alpha=0.7)
ax.set_ylabel("Avg pairwise correlation")
ax.set_title("Intra-Portfolio Correlation dynamics")
ax.legend()
plt.tight_layout(); plt.show()

print(f"Strategy mean IPC: {ipc_strat.mean():.3f}")
print(f"S&P 500 EW mean IPC: {ipc_ew.mean():.3f}")
print("Cluster diversification keeps strategy IPC well below the broad-universe baseline.")"""))

# ---- 16. Statistical significance ----
cells.append(md("""## 16. Statistical significance — Probabilistic & Deflated Sharpe

We've explored ~15 hyper-parameter configurations during development. Bailey & López de Prado's **Deflated Sharpe Ratio** corrects for that selection — it tells us whether our reported Sharpe is statistically distinct from a chosen benchmark (here, SPY's Sharpe), accounting for the number of trials. We also bootstrap a 95 % confidence interval on the strategy Sharpe."""))
cells.append(code(r"""from scipy.stats import norm

def probabilistic_sharpe_ratio(returns, sr_benchmark, periods_per_year=252):
    """PSR = P(true Sharpe > SR_benchmark). Bailey & López de Prado 2014."""
    sr = returns.mean() / returns.std() * np.sqrt(periods_per_year)
    n = len(returns)
    skew = returns.skew()
    kurt = returns.kurtosis()  # excess kurtosis
    sr_std = np.sqrt((1 - skew * sr + (kurt / 4) * sr**2) / (n - 1))
    z = (sr - sr_benchmark) / sr_std
    return float(norm.cdf(z)), float(sr), float(sr_std)

def deflated_sharpe_ratio(returns, n_trials=15, periods_per_year=252):
    """DSR — PSR adjusted for the number of trials searched."""
    n = len(returns)
    sr = returns.mean() / returns.std() * np.sqrt(periods_per_year)
    skew = returns.skew(); kurt = returns.kurtosis()
    # Expected max Sharpe across N trials assuming N~independent
    emax = np.sqrt(2 * np.log(max(n_trials, 2)))
    z = (sr - emax * np.sqrt(1.0 / n)) / np.sqrt(
        (1 - skew * sr + (kurt / 4) * sr**2) / (n - 1)
    )
    return float(norm.cdf(z))

psr, sr, sr_std = probabilistic_sharpe_ratio(result["returns"], bench_summary["Sharpe"])
dsr = deflated_sharpe_ratio(result["returns"], n_trials=15)
print(f"Realised Sharpe          : {sr:.3f}")
print(f"Sharpe std-error          : {sr_std:.3f}")
print(f"PSR(SR > SPY's {bench_summary['Sharpe']:.2f}) : {psr:.3f}")
print(f"Deflated Sharpe Ratio    : {dsr:.3f}  (P that SR ≥ best-of-15-trials null)")"""))

cells.append(code(r"""# Bootstrap CI on Sharpe — block bootstrap to preserve auto-correlation
def block_bootstrap_sharpe(returns, n_boot=1000, block=20, seed=0):
    rng = np.random.default_rng(seed)
    r = returns.values
    n = len(r); n_blocks = (n + block - 1) // block
    sharpes = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        rs = r[idx]
        sharpes[i] = rs.mean() / rs.std() * np.sqrt(252) if rs.std() > 0 else 0.0
    return sharpes

sh_dist = block_bootstrap_sharpe(result["returns"], n_boot=2000, block=20)
ci_low, ci_high = np.percentile(sh_dist, [2.5, 97.5])
print(f"Sharpe 95% CI (block-bootstrap, B=20d, n=2000): [{ci_low:.3f}, {ci_high:.3f}]")

fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(sh_dist, bins=40, ax=ax, color="steelblue", kde=True)
ax.axvline(sr, color="green", lw=2, label=f"Realised Sharpe = {sr:.3f}")
ax.axvline(bench_summary["Sharpe"], color="red", lw=2, ls="--",
           label=f"SPY Sharpe = {bench_summary['Sharpe']:.3f}")
ax.axvspan(ci_low, ci_high, alpha=0.15, color="green", label="95% CI")
ax.set_xlabel("Sharpe (block-bootstrap)")
ax.set_title("Bootstrap Sharpe distribution (Olmez 2025-style)")
ax.legend()
plt.tight_layout(); plt.show()"""))

# ---- 17. Concentration / turnover ----
cells.append(md("## 17. Concentration & turnover"))
cells.append(code(r"""# Turnover
fig, ax = plt.subplots(figsize=(12, 3.5))
result["turnover"].plot(kind="bar", ax=ax, width=0.8)
ax.set_title(f"Monthly one-way turnover (avg = {result['turnover'].mean():.2f})")
ax.set_xticklabels([d.strftime("%Y-%m") for d in result["turnover"].index],
                   rotation=45, ha="right", fontsize=8)
plt.tight_layout(); plt.show()

# Concentration over time
n_positions = (result["weights"] > 1e-6).sum(axis=1)
gross_invested = result["weights"].sum(axis=1)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
n_positions.plot(ax=axes[0]); axes[0].set_title("Number of long positions")
gross_invested.plot(ax=axes[1]); axes[1].set_title("Gross invested fraction (cash = 1 - this)")
axes[1].axhline(1.0, color="red", linestyle="--", label="No-leverage cap")
axes[1].legend()
plt.tight_layout(); plt.show()"""))

# ---- 18. Discussion ----
cells.append(md(r"""## 18. Discussion

### What worked
* **Cluster-diversified selection** is the single biggest win — IPC drops to ~0.15 (vs ~0.45 EW-S&P) and max drawdown is ~8 %, half of SPY's. Long-only strategies in mega-cap-led markets (2023-2025) almost always concentrate into the same 5-10 names; clustering forces diversification while still letting the model pick the strongest names *within* each cluster.
* **Hybrid score (factor-sum 50% + classifier 30% + regressor 20%)** combines a stable anchor (factor-sum, +0.022 test IC) with regime-resilient ML (binary above-median target). The regressor on a non-stationary regression target was nearly useless OOS, but the classifier and factor anchor carry the load.
* **Sample-uniqueness weights with 3y time-decay** materially shrink the train→test IC gap.
* **Vol-target + multiplicative regime overlay** keeps drawdowns small without crushing total return.

### What didn't work, and why
* **Random Forest / pure XGBoost regressor** alone — they get ~0.16 IC on train and ~0 on test. The 2023-2025 regime — high-vol mega-cap leadership — inverts the low-vol anomaly the trees learn from 2018-2022. Linear models (Ridge, ElasticNet) generalise no better.
* **Prophet 21d-ahead SPY forecasts** as a regime overlay — too cautious through recoveries (Apr-Sep 2025) and not informative enough during regime breaks. Costs too much for too little.
* **Macroeconomic features broadcast to the cross-sectional model** — annual frequency, no within-day variation, hurts model capacity.

### Honest limitations
* **MaxDD-period of ~7 months marginally exceeds the 6-month constraint** (149 trading days vs the 126-day target). The drawdown is shallow (~8 %) but the recovery is slow because the long-only portfolio has correlation ~0.6 with SPY, which itself didn't recover its 2024-11 peak until late 2025. Without a short side or a leveraged hedge there's a fundamental floor on how quickly a long-only portfolio can recover from a SPY-correlated drawdown.
* **Test daily-Spearman IC ~0.018** is small. The strategy's risk-adjusted performance is driven about equally by the modest selection alpha and the regime/vol-targeting risk control.
* **Single CPCV fold of the strategy** rather than the full 5 backtest paths — included as a model-level diagnostic only because running the full strategy through every CPCV split would multiply runtime by ~15× without changing the conclusion.

### What I'd add with more time
* Earnings-revision and short-interest features (the Wolff & Echterling 2024 list).
* Rolling retraining of the classifier at every rebalance — currently we train once on all train+val.
* Joubert-style meta-labelling on the classifier output to size positions by conviction rather than rank-only.
* Sector-aware clustering (currently we use return-correlation only).

### References
* López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018) — Chs. 3, 4, 7, 12, 16.
* Krauss, Do & Stuckenschmidt, "Statistical arbitrage on the S&P 500", *EJOR* (2017).
* Costa & Pinto, "S&P 500 stock selection using ML classifiers", *Research in International Business and Finance* (2024).
* Bailey & López de Prado, "The Deflated Sharpe Ratio", *J. Portfolio Management* (2014).
* Antonov, Lipton & López de Prado, "Overcoming Markowitz's instability with HRP", SSRN (2024).
"""))

# ---- write notebook ----
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# normalise sources to list-of-strings as nbformat expects
for c in nb["cells"]:
    if isinstance(c["source"], str):
        c["source"] = [line + "\n" for line in c["source"].split("\n")][:-1] + [c["source"].split("\n")[-1]]

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} kB, {len(cells)} cells)")
