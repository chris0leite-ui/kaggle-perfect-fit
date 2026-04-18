import numpy as np
import pandas as pd
import pytest

from src.causal import (
    preprocess_for_causal,
    run_pc,
    run_direct_lingam,
    run_ges,
    adjacency_to_edges,
    consensus_graph,
    bootstrap_edges,
)


# ---------------------------------------------------------------------------
# preprocess_for_causal
# ---------------------------------------------------------------------------

def _sample_df():
    """Minimal dataframe mimicking the competition schema."""
    return pd.DataFrame({
        "id": [0, 1, 2, 3, 4],
        "x1": [0.1, -0.2, 0.3, 0.0, -0.1],
        "x2": [0.2, -0.3, 0.1, 0.4, 0.0],
        "Country": ["Spain"] * 5,
        "City": ["Zaragoza", "Albacete", "Zaragoza", "Albacete", "Zaragoza"],
        "x4": [0.5, -0.3, 0.1, 0.2, -0.4],
        "x5": [10.0, 999.0, 11.0, 999.0, 9.0],
        "x6": [1.0, 2.0, 3.0, 4.0, 5.0],
        "x7": [5.0, 6.0, 7.0, 8.0, 9.0],
        "x8": [0.1, 0.2, 0.3, 0.4, 0.5],
        "x9": [4.0, 5.0, 6.0, 7.0, 3.0],
        "x10": [1.0, 2.0, 3.0, 4.0, 5.0],
        "x11": [2.0, 3.0, 4.0, 5.0, 6.0],
        "target": [-10.0, -20.0, -30.0, -15.0, -5.0],
    })


def test_preprocess_drops_country():
    data, labels = preprocess_for_causal(_sample_df())
    assert "Country" not in labels


def test_preprocess_drops_id():
    data, labels = preprocess_for_causal(_sample_df())
    assert "id" not in labels


def test_preprocess_encodes_city():
    data, labels = preprocess_for_causal(_sample_df())
    assert "City" in labels
    idx = labels.index("City")
    # Binary values only
    assert set(np.unique(data[:, idx])).issubset({0.0, 1.0})


def test_preprocess_handles_sentinels():
    data, labels = preprocess_for_causal(_sample_df())
    # No 999.0 should remain anywhere
    assert not np.any(data == 999.0)


def test_preprocess_output_shape():
    data, labels = preprocess_for_causal(_sample_df())
    assert data.ndim == 2
    assert data.shape[1] == len(labels)
    # id and Country dropped; all others kept
    assert len(labels) == 12  # x1,x2,City,x4,x5,x6,x7,x8,x9,x10,x11,target


def test_preprocess_includes_target():
    data, labels = preprocess_for_causal(_sample_df())
    assert "target" in labels


# ---------------------------------------------------------------------------
# Causal discovery methods (smoke tests on synthetic data)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Small synthetic dataset with known causal structure: x0 -> x1 -> x2."""
    rng = np.random.default_rng(42)
    n = 300
    x0 = rng.standard_normal(n)
    x1 = 0.8 * x0 + rng.standard_normal(n) * 0.3
    x2 = 0.6 * x1 + rng.standard_normal(n) * 0.3
    data = np.column_stack([x0, x1, x2])
    labels = ["x0", "x1", "x2"]
    return data, labels


def test_pc_returns_adjacency(synthetic_data):
    data, labels = synthetic_data
    adj = run_pc(data, labels, alpha=0.05)
    assert isinstance(adj, np.ndarray)
    assert adj.shape == (3, 3)


def test_pc_detects_edges(synthetic_data):
    data, labels = synthetic_data
    adj = run_pc(data, labels, alpha=0.05)
    # x0-x1 and x1-x2 should be connected (nonzero)
    assert adj[0, 1] != 0 or adj[1, 0] != 0
    assert adj[1, 2] != 0 or adj[2, 1] != 0


def test_lingam_returns_adjacency(synthetic_data):
    data, labels = synthetic_data
    adj, order = run_direct_lingam(data, labels)
    assert isinstance(adj, np.ndarray)
    assert adj.shape == (3, 3)
    assert len(order) == 3


def test_ges_returns_adjacency(synthetic_data):
    data, labels = synthetic_data
    adj = run_ges(data, labels)
    assert isinstance(adj, np.ndarray)
    assert adj.shape == (3, 3)


# ---------------------------------------------------------------------------
# adjacency_to_edges
# ---------------------------------------------------------------------------

def test_adjacency_to_edges_basic():
    adj = np.array([
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.8],
        [0.0, 0.0, 0.0],
    ])
    labels = ["a", "b", "c"]
    edges = adjacency_to_edges(adj, labels, threshold=0.01)
    assert ("a", "b", 0.5) in edges
    assert ("b", "c", 0.8) in edges
    assert len(edges) == 2


def test_adjacency_to_edges_threshold():
    adj = np.array([
        [0.0, 0.005, 0.0],
        [0.0, 0.0, 0.8],
        [0.0, 0.0, 0.0],
    ])
    labels = ["a", "b", "c"]
    edges = adjacency_to_edges(adj, labels, threshold=0.01)
    # 0.005 is below threshold
    assert len(edges) == 1


# ---------------------------------------------------------------------------
# consensus_graph
# ---------------------------------------------------------------------------

def test_consensus_graph_intersection():
    edges_a = [("x0", "x1"), ("x1", "x2"), ("x0", "x2")]
    edges_b = [("x0", "x1"), ("x1", "x2")]
    edges_c = [("x0", "x1"), ("x1", "x2"), ("x2", "x3")]
    result = consensus_graph([edges_a, edges_b, edges_c], min_methods=2)
    # x0->x1 and x1->x2 appear in all three; x0->x2 in 1, x2->x3 in 1
    assert ("x0", "x1") in result
    assert ("x1", "x2") in result
    assert ("x0", "x2") not in result
    assert ("x2", "x3") not in result


def test_consensus_graph_min_methods():
    edges_a = [("x0", "x1"), ("x1", "x2")]
    edges_b = [("x0", "x1")]
    result = consensus_graph([edges_a, edges_b], min_methods=1)
    assert ("x0", "x1") in result
    assert ("x1", "x2") in result


# ---------------------------------------------------------------------------
# bootstrap_edges
# ---------------------------------------------------------------------------

def test_bootstrap_edges_returns_dataframe(synthetic_data):
    data, labels = synthetic_data
    df = bootstrap_edges(data, labels, method="pc", n_boot=5, alpha=0.05)
    assert isinstance(df, pd.DataFrame)
    assert "cause" in df.columns
    assert "effect" in df.columns
    assert "frequency" in df.columns
    assert all(df["frequency"] >= 0) and all(df["frequency"] <= 1)
