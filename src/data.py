import pandas as pd
from pathlib import Path


def split_holdout(path: Path, holdout_frac: float = 0.2, seed: int = 42):
    df = pd.read_csv(path)
    holdout = df.sample(frac=holdout_frac, random_state=seed)
    train = df.drop(holdout.index).reset_index(drop=True)
    holdout = holdout.reset_index(drop=True)
    return train, holdout
