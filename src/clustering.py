"""
Stock clustering for diversified portfolio construction.

Even with perfect cross-sectional ranks, picking the top-N by score concentrates
into clusters of correlated names — in 2023-25 that means most picks land in
mega-cap tech. The Intra-Portfolio Correlation (IPC) of such a portfolio is
high; its drawdowns are deep and slow to recover.

This module groups stocks into clusters by trailing-return correlation
(hierarchical clustering on the López de Prado / Mantegna distance metric
`d_ij = sqrt((1 - rho_ij) / 2)`), then offers a cluster-diversified top-N
selector that takes only the best-scored stock per cluster up to a cap.

The clusters are recomputed at each rebalance using only data observable at
that time — no look-ahead — so the cluster assignment adapts to regime
changes (e.g. when energy stops moving with materials, or when consumer
discretionary decouples from tech).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def cluster_stocks_by_correlation(
    returns_window: pd.DataFrame,
    n_clusters: int = 10,
    min_history: int = 60,
) -> pd.Series:
    """Hierarchical clustering on the correlation-distance metric.

    Args:
        returns_window: T x N daily returns. Tickers with missing data
                        across the window are dropped (cluster assignment
                        can't be done without enough history).
        n_clusters: Target number of clusters. The actual number can be
                    smaller when the universe is small.
        min_history: Minimum non-NaN observations per ticker required to
                     include it in clustering.

    Returns:
        Series indexed by ticker, values are integer cluster labels in
        [1, n_clusters]. Tickers without enough history are absent.
    """
    rw = returns_window.dropna(axis=1, thresh=min_history)
    if rw.shape[1] < n_clusters or rw.shape[1] < 2:
        # too few names — assign each to its own cluster
        return pd.Series(np.arange(1, rw.shape[1] + 1, dtype=int),
                         index=rw.columns, name="cluster")
    rw = rw.fillna(0.0)
    corr = rw.corr().values
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt((1.0 - corr) / 2.0)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    cond = squareform(dist, checks=False)
    link = linkage(cond, method="average")
    labels = fcluster(link, t=n_clusters, criterion="maxclust")
    return pd.Series(labels.astype(int), index=rw.columns, name="cluster")


def cluster_diversified_top_n(
    scores: pd.Series,
    clusters: pd.Series,
    top_n: int,
    per_cluster_cap: int = 2,
) -> pd.Series:
    """Select up to top_n stocks with at most `per_cluster_cap` per cluster.

    Greedy by score: the highest-scored available stock from each cluster
    is added in turn, then second-best per cluster, until we have top_n
    or run out of candidates.

    Args:
        scores: Series indexed by ticker, higher = better.
        clusters: Series indexed by ticker, integer cluster labels.
        top_n: Maximum number of names to select.
        per_cluster_cap: Maximum names from any single cluster.

    Returns:
        Series of selected tickers and their scores, length ≤ top_n.
    """
    df = pd.concat([scores.rename("s"), clusters.rename("c")], axis=1).dropna()
    df = df.sort_values("s", ascending=False)

    selected = []
    counts: dict[int, int] = {}
    for ticker, row in df.iterrows():
        c = int(row["c"])
        if counts.get(c, 0) >= per_cluster_cap:
            continue
        selected.append(ticker)
        counts[c] = counts.get(c, 0) + 1
        if len(selected) >= top_n:
            break

    return df.loc[selected, "s"]
