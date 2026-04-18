import pytest
import pandas as pd
from pathlib import Path
from src.data import split_holdout


DATA_PATH = Path("data/dataset.csv")


def test_split_sizes():
    train, holdout = split_holdout(DATA_PATH, holdout_frac=0.2, seed=42)
    assert len(train) + len(holdout) == 1500
    assert abs(len(holdout) - 300) <= 1


def test_split_no_overlap():
    train, holdout = split_holdout(DATA_PATH, holdout_frac=0.2, seed=42)
    assert len(set(train["id"]) & set(holdout["id"])) == 0


def test_split_reproducible():
    train1, _ = split_holdout(DATA_PATH, holdout_frac=0.2, seed=42)
    train2, _ = split_holdout(DATA_PATH, holdout_frac=0.2, seed=42)
    assert list(train1["id"]) == list(train2["id"])
