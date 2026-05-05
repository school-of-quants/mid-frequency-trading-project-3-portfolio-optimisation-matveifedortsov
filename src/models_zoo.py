"""
Model zoo for cross-sectional ranking.

A common interface around several model architectures so we can compare
them apples-to-apples on per-day Spearman IC. Every model exposes:

    fit(X_train, y_train, X_val=None, y_val=None) -> self
    predict(X) -> np.ndarray

This lets the strategy notebook score, say, XGBoost vs LightGBM vs Ridge,
choose whichever has the best validation IC, and then deploy that one to
the test panel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


@dataclass
class ModelWrapper:
    """Common-interface wrapper holding a fitted estimator."""
    name: str
    estimator: object
    feature_names: list[str]
    scaler: object | None = None  # for linear models that benefit from scaling

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xa = X[self.feature_names]
        if self.scaler is not None:
            Xa = self.scaler.transform(Xa)
        return self.estimator.predict(Xa)

    def predict_xs_rank(self, X: pd.DataFrame) -> pd.Series:
        s = pd.Series(self.predict(X), index=X.index, name=f"{self.name}_score")
        if "date" in X.index.names:
            return s.groupby(level="date").rank(pct=True)
        return s.rank(pct=True)


def _ic_spearman(scores: pd.Series, y: pd.Series) -> float:
    """Per-day Spearman correlation, then averaged via Pearson on the per-day
    pairs. Returns the daily-Spearman IC standard in finance literature.
    """
    df = pd.concat([scores.rename("s"), y.rename("y")], axis=1).dropna()
    if df.empty:
        return float("nan")
    sp = (df.groupby(level="date")["s"].rank(pct=True)
          .corr(df.groupby(level="date")["y"].rank(pct=True)))
    return float(sp)


def fit_xgboost_clf(X_tr, y_tr, X_val=None, y_val=None, **params) -> ModelWrapper:
    """Binary XGBoost classifier — predicts P(target=1).

    For our use the target is the Krauss et al. above-cross-sectional-median
    indicator — a 50/50-stationary target that's resilient to regime shift.
    The wrapper's `predict` returns `P(class=1)` so it composes cleanly with
    the regressor outputs in our ensemble.
    """
    from xgboost import XGBClassifier
    p = dict(
        n_estimators=600, max_depth=4, learning_rate=0.025,
        min_child_weight=80, reg_lambda=4.0,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=-1, tree_method="hist",
    )
    p.update(params)
    m = XGBClassifier(**p)
    if X_val is not None and y_val is not None:
        try:
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=40, verbose=False)
        except TypeError:
            p["early_stopping_rounds"] = 40
            m = XGBClassifier(**p)
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    else:
        m.fit(X_tr, y_tr)

    # The wrapper's .predict() pulls P(class=1); we monkey-patch the
    # predict method on the estimator to return probabilities so the
    # rest of the ensemble code is identical.
    class _ProbaShim:
        def __init__(self, est): self.est = est
        def predict(self, X): return self.est.predict_proba(X)[:, 1]
    return ModelWrapper(name="xgboost_clf",
                        estimator=_ProbaShim(m),
                        feature_names=list(X_tr.columns))


def fit_xgboost(X_tr, y_tr, X_val=None, y_val=None, **params) -> ModelWrapper:
    from xgboost import XGBRegressor
    p = dict(
        n_estimators=600, max_depth=4, learning_rate=0.025,
        min_child_weight=80, reg_lambda=4.0,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42, n_jobs=-1,
        tree_method="hist",
    )
    p.update(params)
    m = XGBRegressor(**p)
    if X_val is not None and y_val is not None:
        try:
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=40, verbose=False)
        except TypeError:
            p["early_stopping_rounds"] = 40
            m = XGBRegressor(**p)
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    else:
        m.fit(X_tr, y_tr)
    return ModelWrapper(name="xgboost", estimator=m,
                        feature_names=list(X_tr.columns))


def fit_lightgbm(X_tr, y_tr, X_val=None, y_val=None, **params) -> ModelWrapper:
    import lightgbm as lgb
    p = dict(
        n_estimators=600, max_depth=-1, num_leaves=31, learning_rate=0.025,
        min_child_samples=200, reg_lambda=2.0,
        subsample=0.8, colsample_bytree=0.8,
        objective="regression", random_state=42, n_jobs=-1, verbosity=-1,
    )
    p.update(params)
    m = lgb.LGBMRegressor(**p)
    if X_val is not None and y_val is not None:
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(40, verbose=False)])
    else:
        m.fit(X_tr, y_tr)
    return ModelWrapper(name="lightgbm", estimator=m,
                        feature_names=list(X_tr.columns))


def fit_ridge(X_tr, y_tr, X_val=None, y_val=None,
              alpha: float = 1.0, **kwargs) -> ModelWrapper:
    """Ridge regression on standardized features. Robust baseline — limited
    capacity means it tends to generalize where trees overfit."""
    scaler = StandardScaler().fit(X_tr)
    Xs = scaler.transform(X_tr)
    m = Ridge(alpha=alpha, random_state=42)
    m.fit(Xs, y_tr)
    return ModelWrapper(name="ridge", estimator=m,
                        feature_names=list(X_tr.columns), scaler=scaler)


def fit_elasticnet(X_tr, y_tr, X_val=None, y_val=None,
                   alpha: float = 0.001, l1_ratio: float = 0.3,
                   **kwargs) -> ModelWrapper:
    scaler = StandardScaler().fit(X_tr)
    Xs = scaler.transform(X_tr)
    m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                   max_iter=5000, random_state=42)
    m.fit(Xs, y_tr)
    return ModelWrapper(name="elasticnet", estimator=m,
                        feature_names=list(X_tr.columns), scaler=scaler)


def compare_models(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    architectures: list[str] | None = None,
) -> pd.DataFrame:
    """Fit every architecture, score by validation Spearman IC, return a table.

    Returns a DataFrame indexed by model name with columns:
        train_ic, val_ic, fit_time_s
    """
    import time
    archs = architectures or ["xgboost", "lightgbm", "ridge", "elasticnet"]
    rows = []
    fits = {}
    for a in archs:
        t0 = time.time()
        if a == "xgboost":
            m = fit_xgboost(X_train, y_train, X_val, y_val)
        elif a == "lightgbm":
            m = fit_lightgbm(X_train, y_train, X_val, y_val)
        elif a == "ridge":
            m = fit_ridge(X_train, y_train)
        elif a == "elasticnet":
            m = fit_elasticnet(X_train, y_train)
        else:
            raise ValueError(f"unknown architecture: {a}")
        elapsed = time.time() - t0

        score_tr = pd.Series(m.predict(X_train), index=X_train.index)
        score_va = pd.Series(m.predict(X_val),   index=X_val.index)
        rows.append({
            "model": a,
            "train_ic": _ic_spearman(score_tr, y_train),
            "val_ic":   _ic_spearman(score_va, y_val),
            "fit_time_s": elapsed,
        })
        fits[a] = m
    table = pd.DataFrame(rows).set_index("model")
    return table, fits


def ensemble_predict(models: dict[str, ModelWrapper], X: pd.DataFrame,
                     weights: dict[str, float] | None = None) -> pd.Series:
    """Weighted average of per-day cross-sectional ranks across models.

    Ranking each model's output before averaging keeps the ensemble
    invariant to scale differences between architectures (e.g. Ridge
    output vs XGBoost output).
    """
    parts = []
    for name, m in models.items():
        w = (weights or {}).get(name, 1.0)
        if w == 0:
            continue
        s = m.predict_xs_rank(X)
        parts.append(w * s)
    if not parts:
        raise ValueError("no models to ensemble")
    avg = sum(parts) / sum((weights or {}).get(n, 1.0) for n in models)
    return avg.rename("ensemble_rank")
