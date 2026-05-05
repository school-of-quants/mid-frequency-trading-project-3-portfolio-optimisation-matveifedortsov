"""
Cross-validation respecting the time-overlap of triple-barrier samples.

Standard K-fold leaks information when samples have overlapping label
horizons: a training sample whose label is determined over [t, t+H] shares
information with any test sample drawn from that interval. Lopez de Prado's
fix is to *purge* training samples whose event horizon overlaps the test
fold, plus an *embargo* of E days after the test fold to defend against
serial correlation.

Combinatorial Purged CV (CPCV): split time into N groups, choose K of them
to form a test path, repeat for all C(N, K) combinations. This produces
many backtest paths from a single sample, giving a distribution of OOS
metrics rather than a single point estimate.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


# -----------------------------------------------------------------------------
# PurgedKFold
# -----------------------------------------------------------------------------

class PurgedKFold(BaseCrossValidator):
    """K-fold CV with purging and embargo (Lopez de Prado, AFML Ch.7).

    Splits are made on the *time* axis; each (date, ticker) sample inherits
    the date's fold assignment. Training samples whose event horizon
    [t1_start, t1_end] overlaps the test fold are purged. An embargo of
    `embargo_pct` of the total length is applied after each test fold.

    Args:
        n_splits: Number of folds.
        t1: Series indexed by sample index; values are event end dates.
            t1.index must align with the X passed to .split().
            t1.values must be Timestamps.
        embargo_pct: Fraction of total samples to embargo after each test fold.
    """

    def __init__(self, n_splits: int = 5, t1: pd.Series | None = None,
                 embargo_pct: float = 0.01):
        if t1 is None:
            raise ValueError("t1 is required (event end dates per sample).")
        self.n_splits = n_splits
        self.t1 = t1
        self.embargo_pct = embargo_pct

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if not (X.index == self.t1.index).all():
            raise ValueError("X.index must match t1.index.")
        n = len(X)
        indices = np.arange(n)
        embargo = int(n * self.embargo_pct)

        # split into n_splits contiguous chunks by time order
        # (samples are assumed already sorted by time outside)
        test_ranges = [(i[0], i[-1] + 1) for i in
                       np.array_split(indices, self.n_splits)]

        # event start time t0 = the index level "date"; event end = self.t1
        # we use the date level if MultiIndex, else the index itself
        if isinstance(X.index, pd.MultiIndex):
            t0 = X.index.get_level_values("date")
        else:
            t0 = X.index
        t0 = pd.DatetimeIndex(t0)
        t1 = pd.DatetimeIndex(self.t1.values)

        for start, end in test_ranges:
            test_idx = indices[start:end]
            test_t0_min = t0[start]
            test_t1_max = t1[start:end].max()

            # purge: drop training samples whose event horizon [t0_i, t1_i]
            # overlaps with the test horizon [test_t0_min, test_t1_max]
            train_mask = (t1 < test_t0_min) | (t0 > test_t1_max)
            train_idx = indices[train_mask.values if hasattr(train_mask, "values")
                                else train_mask]

            # exclude the test indices themselves
            train_idx = np.setdiff1d(train_idx, test_idx, assume_unique=False)

            # embargo: drop training samples in [end, end + embargo)
            if embargo > 0:
                emb_lo = end
                emb_hi = min(n, end + embargo)
                train_idx = np.setdiff1d(train_idx, indices[emb_lo:emb_hi])

            yield train_idx, test_idx


# -----------------------------------------------------------------------------
# Combinatorial Purged CV
# -----------------------------------------------------------------------------

def cpcv_paths(n_groups: int, k_test_groups: int) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """Compute CPCV split combinations and the resulting backtest paths.

    Args:
        n_groups: Number of equal-time groups (N).
        k_test_groups: Test groups per split (K).

    Returns:
        combos: list of group-index tuples, one per split (length C(N, K)).
        paths: 2D array (n_paths, n_groups) where each row is a "path" — a
               sequence of split indices that, taken together, give an OOS
               prediction for every group exactly once. n_paths = C(N-1, K-1).
    """
    combos = list(combinations(range(n_groups), k_test_groups))
    # for each (split_idx, group) record whether group is in test
    is_test = np.zeros((len(combos), n_groups), dtype=bool)
    for i, c in enumerate(combos):
        is_test[i, list(c)] = True

    # Each path picks, for each group, one split where that group was test.
    # We need to ensure that for each path, the chosen splits for different
    # groups are *different splits*.  Lopez de Prado shows there are exactly
    # C(N-1, K-1) such paths and constructs them as follows:
    #   path p, group g -> split index = the p-th combo containing g
    # We just enumerate.
    # Each group appears in C(N-1, K-1) splits; that is the number of paths.
    n_paths = 1
    for i in range(k_test_groups - 1):
        n_paths = n_paths * (n_groups - 1 - i) // (i + 1)
    paths = np.full((n_paths, n_groups), -1, dtype=int)
    for g in range(n_groups):
        # all splits where g is test
        splits_with_g = np.where(is_test[:, g])[0]
        if len(splits_with_g) != n_paths:  # pragma: no cover
            raise RuntimeError(
                f"CPCV path count mismatch for group {g}: "
                f"got {len(splits_with_g)}, expected {n_paths}"
            )
        paths[:, g] = splits_with_g
    return combos, paths


def cpcv_split(
    X: pd.DataFrame,
    t1: pd.Series,
    n_groups: int = 6,
    k_test_groups: int = 2,
    embargo_pct: float = 0.01,
) -> Iterator[tuple[np.ndarray, np.ndarray, tuple[int, ...]]]:
    """Yield (train_idx, test_idx, group_tuple) for every CPCV split.

    Args:
        X: Sample DataFrame (must be sorted by date).
        t1: Series indexed like X, values are event end dates.
        n_groups: Total time groups N.
        k_test_groups: Test groups per split K.
        embargo_pct: Embargo as fraction of total length.

    Yields:
        Tuples (train_idx, test_idx, groups_used_as_test).
    """
    if not (X.index == t1.index).all():
        raise ValueError("X.index must match t1.index.")

    n = len(X)
    indices = np.arange(n)
    embargo = int(n * embargo_pct)

    if isinstance(X.index, pd.MultiIndex):
        t0 = X.index.get_level_values("date")
    else:
        t0 = X.index
    t0 = pd.DatetimeIndex(t0)
    t1_arr = pd.DatetimeIndex(t1.values)

    group_ranges = [(g[0], g[-1] + 1) for g in np.array_split(indices, n_groups)]

    combos, _ = cpcv_paths(n_groups, k_test_groups)

    for groups_in_test in combos:
        test_idx_parts = []
        test_t0_intervals = []
        for g in groups_in_test:
            s, e = group_ranges[g]
            test_idx_parts.append(indices[s:e])
            test_t0_intervals.append((t0[s], t1_arr[s:e].max()))
        test_idx = np.sort(np.concatenate(test_idx_parts))

        # purge training: remove samples whose event horizon overlaps any test interval
        train_mask = np.ones(n, dtype=bool)
        for t0_lo, t1_hi in test_t0_intervals:
            overlap = ~((t1_arr < t0_lo) | (t0 > t1_hi))
            train_mask &= ~overlap.values if hasattr(overlap, "values") else ~overlap
        train_mask[test_idx] = False

        # embargo after each test group
        if embargo > 0:
            for g in groups_in_test:
                _, e = group_ranges[g]
                emb_hi = min(n, e + embargo)
                train_mask[e:emb_hi] = False

        yield indices[train_mask], test_idx, groups_in_test
