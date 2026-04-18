"""Causal discovery utilities.

Provides wrappers around PC, DirectLiNGAM, and GES algorithms,
plus helpers for preprocessing, edge extraction, consensus graphs,
and bootstrap confidence estimation.
"""

import sys
import types
from collections import Counter

import numpy as np
import pandas as pd

# lingam needs semopy/psy which fail to build in this environment;
# stub them so the DirectLiNGAM import succeeds.
for _mod in ("semopy", "psy", "psy.cfa"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)
if not hasattr(sys.modules["psy"], "cfa"):
    sys.modules["psy"].cfa = None  # type: ignore[attr-defined]

import lingam  # noqa: E402
from causallearn.search.ConstraintBased.PC import pc  # noqa: E402
from causallearn.search.ScoreBased.GES import ges  # noqa: E402


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_for_causal(df: pd.DataFrame, sentinel: float = 999.0) -> tuple[np.ndarray, list[str]]:
    """Prepare a numeric matrix for causal discovery.

    * Drops ``id`` and ``Country`` (constant).
    * Encodes ``City`` as binary (Zaragoza=1, Albacete=0).
    * Replaces *sentinel* values with the column median.
    * Returns ``(data_array, column_labels)``.
    """
    df = df.copy()
    df = df.drop(columns=["id", "Country"], errors="ignore")

    if "City" in df.columns:
        df["City"] = (df["City"] == "Zaragoza").astype(float)

    # Replace sentinel with median per column
    num_cols = df.select_dtypes("number").columns
    for col in num_cols:
        mask = df[col] == sentinel
        if mask.any():
            median_val = df.loc[~mask, col].median()
            df.loc[mask, col] = median_val

    labels = list(df.columns)
    data = df[labels].to_numpy(dtype=np.float64)
    return data, labels


# ---------------------------------------------------------------------------
# Causal discovery wrappers
# ---------------------------------------------------------------------------

def run_pc(data: np.ndarray, labels: list[str], alpha: float = 0.05) -> np.ndarray:
    """Run the PC algorithm and return a numeric adjacency matrix.

    The returned matrix uses the causal-learn convention:
    * ``adj[i, j] = -1`` and ``adj[j, i] = 1`` means ``i -> j``.
    * ``adj[i, j] = adj[j, i] = -1`` means undirected ``i -- j``.
    """
    cg = pc(data, alpha=alpha, indep_test="fisherz", show_progress=False)
    return cg.G.graph.astype(float)


def run_direct_lingam(data: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[int]]:
    """Run DirectLiNGAM and return (adjacency_matrix, causal_order).

    ``adjacency_matrix[i, j]`` is the effect of ``x_j`` on ``x_i``.
    """
    model = lingam.DirectLiNGAM()
    model.fit(data)
    return model.adjacency_matrix_.astype(float), list(model.causal_order_)


def run_ges(data: np.ndarray, labels: list[str], score_func: str = "local_score_CV_general") -> np.ndarray:
    """Run GES (Greedy Equivalence Search) and return a numeric adjacency matrix.

    Uses cross-validated local score by default (avoids a numpy-compat bug in
    the BIC scorer).  Same edge encoding as :func:`run_pc`.

    .. note:: CV scoring can be slow for >10 variables.  Consider using
       ``score_func="local_score_BDeu"`` for faster (approximate) results.
    """
    record = ges(data, score_func=score_func)
    return record["G"].graph.astype(float)


# ---------------------------------------------------------------------------
# Edge extraction helpers
# ---------------------------------------------------------------------------

def adjacency_to_edges(
    adj: np.ndarray,
    labels: list[str],
    threshold: float = 0.01,
) -> list[tuple[str, str, float]]:
    """Convert a weighted adjacency matrix to a list of ``(cause, effect, weight)`` tuples.

    Only entries with ``|weight| >= threshold`` are included.
    Uses standard convention: ``adj[i, j] != 0`` means edge ``i -> j``.
    """
    edges: list[tuple[str, str, float]] = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(n):
            w = adj[i, j]
            if abs(w) >= threshold:
                edges.append((labels[i], labels[j], float(w)))
    return edges


def consensus_graph(
    edge_lists: list[list[tuple[str, str, ...]]],
    min_methods: int = 2,
) -> list[tuple[str, str]]:
    """Return directed edges that appear in at least *min_methods* edge lists.

    Each edge list is a sequence of tuples whose first two elements are
    ``(cause, effect)``. Extra elements (weights) are ignored.
    """
    counter: Counter[tuple[str, str]] = Counter()
    for edges in edge_lists:
        # Deduplicate within a single method
        seen: set[tuple[str, str]] = set()
        for e in edges:
            pair = (e[0], e[1])
            if pair not in seen:
                seen.add(pair)
                counter[pair] += 1
    return [pair for pair, count in counter.items() if count >= min_methods]


# ---------------------------------------------------------------------------
# Bootstrap confidence
# ---------------------------------------------------------------------------

def bootstrap_edges(
    data: np.ndarray,
    labels: list[str],
    method: str = "pc",
    n_boot: int = 200,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Bootstrap resample *data* and run a causal method each time.

    Returns a DataFrame with columns ``cause, effect, frequency`` where
    *frequency* is the fraction of bootstrap runs in which the edge appeared.
    """
    n = data.shape[0]
    rng = np.random.default_rng(0)
    edge_counts: Counter[tuple[str, str]] = Counter()

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sample = data[idx]
        try:
            if method == "pc":
                adj = run_pc(sample, labels, alpha=alpha)
                edges = _pc_adj_to_pairs(adj, labels)
            elif method == "lingam":
                adj, _ = run_direct_lingam(sample, labels)
                # LiNGAM convention: adj[i,j] = effect of j on i (j->i)
                edges = {
                    (labels[j], labels[i])
                    for i in range(adj.shape[0])
                    for j in range(adj.shape[1])
                    if abs(adj[i, j]) >= 0.01 and i != j
                }
            elif method == "ges":
                adj = run_ges(sample, labels)
                edges = _pc_adj_to_pairs(adj, labels)
            else:
                raise ValueError(f"Unknown method: {method}")
            for e in edges:
                edge_counts[e] += 1
        except Exception:
            continue  # skip failed resamples

    rows = [
        {"cause": c, "effect": e, "frequency": cnt / n_boot}
        for (c, e), cnt in edge_counts.items()
    ]
    df = pd.DataFrame(rows, columns=["cause", "effect", "frequency"])
    return df.sort_values("frequency", ascending=False).reset_index(drop=True)


def _pc_adj_to_pairs(adj: np.ndarray, labels: list[str]) -> set[tuple[str, str]]:
    """Extract directed edge pairs from a PC/GES adjacency matrix.

    Convention: adj[i,j]=-1, adj[j,i]=1 means i->j.
    Undirected (both -1) recorded as both directions.
    """
    pairs: set[tuple[str, str]] = set()
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == -1 and adj[j, i] == 1:
                pairs.add((labels[i], labels[j]))
            elif adj[i, j] == 1 and adj[j, i] == -1:
                pairs.add((labels[j], labels[i]))
            elif adj[i, j] == -1 and adj[j, i] == -1:
                # Undirected — record both
                pairs.add((labels[i], labels[j]))
                pairs.add((labels[j], labels[i]))
    return pairs
