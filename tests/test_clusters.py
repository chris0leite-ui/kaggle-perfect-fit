import pandas as pd
import numpy as np
import pytest
from src.clusters import find_x4_gap, assign_clusters, replace_sentinels, cluster_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bimodal_x4():
    """Synthetic x4 with a clear gap around 0."""
    low = np.array([-0.45, -0.40, -0.35, -0.30, -0.25, -0.20])
    high = np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
    return pd.Series(np.concatenate([low, high]))


def _cluster_df():
    """Synthetic DataFrame with 4 clusters (5 rows each)."""
    return pd.DataFrame({
        "City": (["Albacete"] * 5 + ["Albacete"] * 5
                 + ["Zaragoza"] * 5 + ["Zaragoza"] * 5),
        "x4": ([-0.4, -0.3, -0.35, -0.25, -0.45]      # Albacete low
               + [0.4, 0.3, 0.35, 0.25, 0.45]          # Albacete high
               + [-0.4, -0.3, -0.35, -0.25, -0.45]     # Zaragoza low
               + [0.4, 0.3, 0.35, 0.25, 0.45]),        # Zaragoza high
        "target": ([10, 12, 11, 13, 14]
                   + [30, 32, 31, 33, 34]
                   + [-10, -12, -11, -13, -14]
                   + [0, 2, 1, 3, 4]),
        "x1": np.random.default_rng(0).standard_normal(20),
        "x2": np.random.default_rng(1).standard_normal(20),
        "x5": [5, 6, 7, 8, 999.0] + [10, 11, 12, 13, 14] + [1, 2, 3, 4, 5] + [20, 21, 22, 23, 24],
        "x8": np.random.default_rng(2).standard_normal(20),
        "x10": np.random.default_rng(3).standard_normal(20),
        "x11": np.random.default_rng(4).standard_normal(20),
    })


# ---------------------------------------------------------------------------
# find_x4_gap
# ---------------------------------------------------------------------------

def test_find_x4_gap_bimodal():
    gap_start, gap_end, midpoint = find_x4_gap(_bimodal_x4())
    assert gap_start < 0 < gap_end
    assert gap_start >= -0.20
    assert gap_end <= 0.20


def test_find_x4_gap_midpoint():
    gap_start, gap_end, midpoint = find_x4_gap(_bimodal_x4())
    assert midpoint == pytest.approx((gap_start + gap_end) / 2)


def test_find_x4_gap_uniform():
    """Uniform data has no large gap — function still returns without error."""
    series = pd.Series(np.linspace(-1, 1, 100))
    gap_start, gap_end, midpoint = find_x4_gap(series)
    assert gap_end - gap_start < 0.1  # no big gap


# ---------------------------------------------------------------------------
# assign_clusters
# ---------------------------------------------------------------------------

def test_assign_clusters_four_labels():
    df = _cluster_df()
    labels = assign_clusters(df, x4_threshold=0.0)
    assert set(labels.unique()) == {
        "Albacete_high", "Albacete_low",
        "Zaragoza_high", "Zaragoza_low",
    }


def test_assign_clusters_correct_assignment():
    df = pd.DataFrame({"City": ["Zaragoza", "Albacete"], "x4": [0.3, -0.3]})
    labels = assign_clusters(df, x4_threshold=0.0)
    assert labels.iloc[0] == "Zaragoza_high"
    assert labels.iloc[1] == "Albacete_low"


def test_assign_clusters_returns_series():
    df = _cluster_df()
    labels = assign_clusters(df, x4_threshold=0.0)
    assert isinstance(labels, pd.Series)
    assert list(labels.index) == list(df.index)


def test_assign_clusters_auto_threshold():
    """Without explicit threshold, uses find_x4_gap to auto-detect."""
    df = _cluster_df()
    labels = assign_clusters(df)
    assert len(labels.unique()) == 4


# ---------------------------------------------------------------------------
# replace_sentinels
# ---------------------------------------------------------------------------

def test_replace_sentinels_replaces_999():
    s = pd.Series([1.0, 2.0, 999.0, 3.0])
    result = replace_sentinels(s)
    expected = pd.Series([1.0, 2.0, 2.0, 3.0])
    pd.testing.assert_series_equal(result, expected)


def test_replace_sentinels_no_change():
    s = pd.Series([1.0, 2.0, 3.0])
    result = replace_sentinels(s)
    pd.testing.assert_series_equal(result, s)


def test_replace_sentinels_custom_sentinel():
    s = pd.Series([10.0, -99.0, 20.0])
    result = replace_sentinels(s, sentinel=-99.0)
    expected = pd.Series([10.0, 15.0, 20.0])
    pd.testing.assert_series_equal(result, expected)


# ---------------------------------------------------------------------------
# cluster_stats
# ---------------------------------------------------------------------------

def test_cluster_stats_has_all_clusters():
    df = _cluster_df()
    df["cluster"] = assign_clusters(df, x4_threshold=0.0)
    result = cluster_stats(df, "cluster", ["target"])
    assert set(result.index) == {
        "Albacete_high", "Albacete_low",
        "Zaragoza_high", "Zaragoza_low",
    }


def test_cluster_stats_correct_count():
    df = _cluster_df()
    df["cluster"] = assign_clusters(df, x4_threshold=0.0)
    result = cluster_stats(df, "cluster", ["target"])
    assert result.loc["Albacete_high", ("target", "count")] == 5
    assert result.loc["Zaragoza_low", ("target", "count")] == 5


def test_cluster_stats_correct_mean():
    df = _cluster_df()
    df["cluster"] = assign_clusters(df, x4_threshold=0.0)
    result = cluster_stats(df, "cluster", ["target"])
    # Albacete_high target: [30, 32, 31, 33, 34] → mean = 32.0
    assert result.loc["Albacete_high", ("target", "mean")] == pytest.approx(32.0)


def test_cluster_stats_cleans_x5_sentinels():
    df = _cluster_df()
    df["cluster"] = assign_clusters(df, x4_threshold=0.0)
    result = cluster_stats(df, "cluster", ["x5"], sentinel_cols={"x5": 999.0})
    # Global x5 non-sentinel values: [5,6,7,8, 10,11,12,13,14, 1,2,3,4,5, 20,21,22,23,24]
    # Global median = 10.0, so 999.0 → 10.0
    # Albacete_low x5: [5, 6, 7, 8, 10.0] → mean = 7.2
    assert result.loc["Albacete_low", ("x5", "mean")] == pytest.approx(7.2)


def test_cluster_stats_columns():
    df = _cluster_df()
    df["cluster"] = assign_clusters(df, x4_threshold=0.0)
    variables = ["target", "x1"]
    result = cluster_stats(df, "cluster", variables)
    expected_stats = ["count", "mean", "median", "std", "min", "max"]
    for var in variables:
        for stat in expected_stats:
            assert (var, stat) in result.columns
