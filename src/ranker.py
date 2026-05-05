"""
Cross-sectional ranking model.

Predicts a per-stock score whose cross-sectional rank correlates with
forward 21-day returns. We train an XGBoost regressor on the *residual*
forward return (return minus cross-sectional mean on each forward day),
then convert the model's continuous output into a per-day percentile rank
which serves as the bet size.

Why a regressor on residuals rather than a triple-barrier classifier:
  * The signal-to-noise ratio of binary triple-barrier labels is too low
    on broadly diversified equity universes — base AUC ~0.52 in our prior
    run. Continuous residual-return targets retain more information.
  * Cross-sectional ranks (rather than raw scores) are naturally
    market-neutral and dollar-neutral when used to *select* names —
    perfect for a top-N long-only portfolio.
  * The model's output is calibrated by construction (ranks are uniform).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------------

def forward_residual_returns(
    close: pd.DataFrame,
    horizon: int = 21,
) -> pd.DataFrame:
    """Forward residual return: future H-day return minus the cross-sectional
    mean on the same forward window.

    For each (date, ticker), the value is:
        r_{t -> t+H} - mean_j r_{j, t -> t+H}
    where the mean is over all tickers with valid prices at both endpoints.

    Last `horizon` rows are NaN (insufficient forward data).

    Args:
        close: Adjusted close prices, T x N.
        horizon: Forward window in trading days.

    Returns:
        DataFrame of residual forward returns, same shape as `close`.
    """
    fwd = close.shift(-horizon) / close - 1.0
    xs_mean = fwd.mean(axis=1)
    return fwd.sub(xs_mean, axis=0)


def forward_return_rank(
    close: pd.DataFrame,
    horizon: int = 21,
) -> pd.DataFrame:
    """Cross-sectional percentile rank of forward H-day returns.

    Useful as a smooth alternative target when you want the model to learn
    *relative* outperformance without being sensitive to scale.
    """
    fwd = close.shift(-horizon) / close - 1.0
    return fwd.rank(axis=1, pct=True)


def forward_above_xs_median(
    close: pd.DataFrame,
    horizon: int = 21,
) -> pd.DataFrame:
    """Binary cross-sectional target: did this stock beat the cross-sectional
    median over the next H trading days?

    This target is **stationary by construction** — every day the
    distribution is exactly 50/50 across the active universe, regardless
    of regime. That makes it the right choice when the absolute level of
    forward returns shifts dramatically between train and test (which is
    precisely the train-on-2018-2022 / test-on-2023-2025 problem).

    Reference: Krauss, Do & Stuckenschmidt (2017), "Deep neural networks,
    gradient-boosted trees, random forests: Statistical arbitrage on the
    S&P 500", *European Journal of Operational Research*.

    Returns:
        Boolean DataFrame, same shape as `close`. NaN where the cross-
        sectional median is undefined or the forward window is incomplete.
    """
    fwd = close.shift(-horizon) / close - 1.0
    med = fwd.median(axis=1)
    out = fwd.gt(med, axis=0).astype(float)
    out[fwd.isna()] = np.nan
    return out


# -----------------------------------------------------------------------------
# Model wrapper
# -----------------------------------------------------------------------------

@dataclass
class CrossSectionalRanker:
    """Thin wrapper around an XGBoost regressor that produces per-day
    cross-sectional ranks suitable for use as bet sizes.

    Attributes:
        model: A fitted regressor with .predict(X) → 1-D numpy array.
        feature_names: Ordered list of features used at fit time.
    """

    model: object
    feature_names: list[str]

    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        """Return raw model scores (unscaled, signed)."""
        scores = self.model.predict(X[self.feature_names])
        return pd.Series(scores, index=X.index, name="score")

    def predict_xs_rank(self, X: pd.DataFrame) -> pd.Series:
        """Return per-day percentile ranks in [0, 1] of the raw scores.

        On any given date all included tickers are jointly ranked; ranks are
        scale-invariant so they're directly usable as a bet size or as input
        to a top-N selection rule.
        """
        s = self.predict_score(X)
        if "date" in X.index.names:
            return s.groupby(level="date").rank(pct=True).rename("xs_rank")
        # fall back to a single-day rank
        return s.rank(pct=True).rename("xs_rank")


def train_ranker(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    params: dict | None = None,
    early_stopping_rounds: int | None = None,
) -> CrossSectionalRanker:
    """Train an XGBoost regressor for cross-sectional ranking.

    Args:
        X_train, y_train: Training features (long format) and continuous target.
        X_val, y_val: Optional validation set. If both provided AND
            `early_stopping_rounds` is set, training uses early stopping.
        params: XGBRegressor hyperparameters — sensible defaults are merged in.
        early_stopping_rounds: If set with X_val/y_val, enables early stopping.

    Returns:
        Fitted CrossSectionalRanker.
    """
    from xgboost import XGBRegressor

    default_params = {
        "n_estimators": 600,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    p = {**default_params, **(params or {})}

    fit_kwargs: dict = {}
    if (
        early_stopping_rounds is not None
        and X_val is not None
        and y_val is not None
    ):
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False
        # xgboost ≥ 1.6 accepts `early_stopping_rounds` in fit; ≥ 2.0 deprecates
        # it in favour of the constructor.  We try fit first and fall back.
        try:
            model = XGBRegressor(**p)
            model.fit(X_train, y_train,
                      early_stopping_rounds=early_stopping_rounds, **fit_kwargs)
        except TypeError:
            p2 = {**p, "early_stopping_rounds": early_stopping_rounds}
            model = XGBRegressor(**p2)
            model.fit(X_train, y_train, **fit_kwargs)
    else:
        model = XGBRegressor(**p)
        model.fit(X_train, y_train)

    return CrossSectionalRanker(model=model, feature_names=list(X_train.columns))
