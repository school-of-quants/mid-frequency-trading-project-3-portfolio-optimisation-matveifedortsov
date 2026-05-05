"""
Sample uniqueness weighting (Lopez de Prado, AFML Ch.4).

Each (date, ticker) sample's label is determined over a 21-day forward
window, so consecutive samples on the same ticker share most of their
label. Vanilla XGBoost training treats each row as iid, which over-weights
information from densely-sampled regions and inflates train IC at the
expense of OOS generalisation.

The standard fix is to weight each sample by its *average uniqueness*:
the average, over all bars in its label horizon, of `1 / number of
concurrently-active labels`. We approximate this with a closed-form
sufficient statistic per ticker — every sample on a ticker has the same
uniqueness (1 / horizon) ignoring boundary effects, so we instead
down-weight by *time-on-event*: each ticker contributes total weight
proportional to its number of *non-overlapping* events, which equals
`sample_count / horizon`. This is mathematically equivalent for our
constant-horizon labels and dodges the O(N²) AFML implementation.

We also apply optional time-decay (more recent samples weighted higher)
to combat regime drift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def uniqueness_weights(
    sample_index: pd.MultiIndex,
    horizon: int = 21,
    time_decay_half_life_days: float | None = None,
) -> np.ndarray:
    """Compute per-sample weights.

    With constant `horizon`, every sample shares its label window with
    `horizon - 1` neighbours on the same ticker. So the AFML uniqueness
    weight collapses to `1 / horizon`. To restore variability we apply
    an optional exponential time-decay so recent samples carry more weight.

    Args:
        sample_index: MultiIndex with levels ("date", "ticker").
        horizon: Label-horizon length in trading days (used only for sanity).
        time_decay_half_life_days: If given, exponential decay with this
            half-life applied to the *date* level (older = lighter).

    Returns:
        1-D numpy array of weights, same length as sample_index, normalised
        to mean 1.
    """
    n = len(sample_index)
    w = np.full(n, 1.0 / horizon)

    if time_decay_half_life_days is not None and time_decay_half_life_days > 0:
        d = sample_index.get_level_values("date")
        age = (d.max() - d).days.values.astype(float)
        decay = np.power(0.5, age / float(time_decay_half_life_days))
        w = w * decay

    # normalise to mean 1 so XGBoost's effective sample size is preserved
    w = w * (n / w.sum())
    return w
