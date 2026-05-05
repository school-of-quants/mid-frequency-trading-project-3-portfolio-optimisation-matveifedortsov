"""
Two-stage ML model: base classifier + meta-labeler.

Stage 1 (base): XGBoost binary classifier predicting P(triple-barrier label = +1)
                given cross-sectional features.
Stage 2 (meta): XGBoost binary classifier predicting whether the base model's
                "long" call will be correct, given the base features + base
                model's probability output.

Training pipeline:
    1. Train base on train set.
    2. Score base on validation set.
    3. Build meta target = (base predicted long) AND (base prediction correct).
       Restrict meta training to rows where base predicted long.
    4. Train meta on the validation set with augmented features.

At inference time:
    p_base    = base.predict_proba(X)[:, 1]
    side      = (p_base > base_threshold).astype(int)         # 0 or 1
    X_meta    = augment(X, p_base)
    p_meta    = meta.predict_proba(X_meta)[:, 1]
    bet_size  = meta_prob_to_size(p_meta) * side               # ∈ [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


# -----------------------------------------------------------------------------
# Bet sizing (Lopez de Prado, AFML Ch.10)
# -----------------------------------------------------------------------------

def meta_prob_to_size(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Translate meta-labeler probability to position size in [-1, 1].

    z = (2p - 1) / (2 sqrt(p(1-p)))
    size = 2 * Phi(z) - 1

    Args:
        p: Probability of meta-label = 1 (base call correct).
        eps: Floor for p(1-p) to avoid divide-by-zero.

    Returns:
        Bet size in [-1, 1]; for our long-only system we clip negatives to 0.
    """
    p = np.clip(p, eps, 1 - eps)
    z = (2 * p - 1) / (2 * np.sqrt(p * (1 - p)))
    return 2 * norm.cdf(z) - 1


# -----------------------------------------------------------------------------
# Model wrapper
# -----------------------------------------------------------------------------

@dataclass
class TwoStageModel:
    """Container for the two trained models and their hyperparameters."""

    base_model: object   # xgboost.XGBClassifier (or sklearn-compatible)
    meta_model: object
    base_features: list[str]
    meta_features: list[str]
    base_threshold: float = 0.5

    def predict_bet_size(self, X: pd.DataFrame) -> pd.Series:
        """Score a feature DataFrame and return long-only bet size in [0, 1]."""
        p_base = self.base_model.predict_proba(X[self.base_features])[:, 1]
        side = (p_base > self.base_threshold).astype(int)

        X_meta = X[self.base_features].copy()
        X_meta["p_base"] = p_base
        X_meta["base_side"] = side
        # rank within day
        if "date" in X.index.names:
            X_meta["p_base_rank"] = (
                pd.Series(p_base, index=X.index)
                .groupby(level="date").rank(pct=True).values
            )
        else:
            X_meta["p_base_rank"] = pd.Series(p_base).rank(pct=True).values

        p_meta = self.meta_model.predict_proba(X_meta[self.meta_features])[:, 1]
        size = meta_prob_to_size(p_meta)
        size = np.clip(size, 0.0, 1.0) * side  # long-only
        return pd.Series(size, index=X.index, name="bet_size")


def make_meta_features(
    X: pd.DataFrame,
    p_base: np.ndarray,
) -> pd.DataFrame:
    """Augment base features with base model output, for meta-labeler input."""
    out = X.copy()
    out["p_base"] = p_base
    out["base_side"] = (p_base > 0.5).astype(int)
    if "date" in X.index.names:
        out["p_base_rank"] = (
            pd.Series(p_base, index=X.index)
            .groupby(level="date").rank(pct=True).values
        )
    else:
        out["p_base_rank"] = pd.Series(p_base).rank(pct=True).values
    return out


def train_two_stage(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    base_params: dict | None = None,
    meta_params: dict | None = None,
    base_threshold: float = 0.55,
    sample_weight_train: np.ndarray | None = None,
) -> TwoStageModel:
    """Train base + meta models.

    Args:
        X_train, y_train: Base model training data. y_train ∈ {0, 1}
            (we use binary "future ret > 0 with triple-barrier criterion").
        X_val, y_val: Meta-labeler training data (held-out, time-after-train).
        base_params, meta_params: Hyperparameters for XGBClassifier.
        base_threshold: Probability threshold defining a "long" base call.
        sample_weight_train: Optional sample weights for base (e.g. to address
            time-of-event uniqueness; AFML Ch.4).

    Returns:
        TwoStageModel ready for inference.
    """
    from xgboost import XGBClassifier

    base_params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    } | (base_params or {})

    meta_params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    } | (meta_params or {})

    base_features = list(X_train.columns)

    base = XGBClassifier(**base_params)
    base.fit(X_train, y_train, sample_weight=sample_weight_train)

    # Build meta-labeler training set on validation fold
    p_base_val = base.predict_proba(X_val)[:, 1]
    long_mask = p_base_val > base_threshold

    X_val_aug = make_meta_features(X_val, p_base_val)
    # meta target: "did the base side call equal the true label?"
    # i.e. base says long (side=1) AND y=1 => correct (1); else 0.
    side_val = (p_base_val > base_threshold).astype(int)
    y_meta = ((side_val == 1) & (y_val.values == 1)).astype(int)

    # Train meta only on rows where base went long
    if long_mask.sum() < 50:
        # too few long calls to train a useful meta — fall back to a model
        # trained on all rows
        train_idx = np.arange(len(X_val_aug))
    else:
        train_idx = np.where(long_mask)[0]

    meta_features = list(X_val_aug.columns)

    meta = XGBClassifier(**meta_params)
    meta.fit(X_val_aug.iloc[train_idx], y_meta[train_idx])

    return TwoStageModel(
        base_model=base,
        meta_model=meta,
        base_features=base_features,
        meta_features=meta_features,
        base_threshold=base_threshold,
    )
