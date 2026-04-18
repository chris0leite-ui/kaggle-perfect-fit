import pandas as pd
from pathlib import Path


def split_holdout(path: Path, holdout_frac: float = 0.2, seed: int = 42):
    df = pd.read_csv(path)
    holdout = df.sample(frac=holdout_frac, random_state=seed)
    train = df.drop(holdout.index).reset_index(drop=True)
    holdout = holdout.reset_index(drop=True)
    return train, holdout


def load_train_holdout(data_dir=None):
    """Load train.csv and holdout.csv, creating them from dataset.csv if needed."""
    if data_dir is None:
        data_dir = Path("data")
    else:
        data_dir = Path(data_dir)

    train_path = data_dir / "train.csv"
    holdout_path = data_dir / "holdout.csv"

    if train_path.exists() and holdout_path.exists():
        return pd.read_csv(train_path), pd.read_csv(holdout_path)

    train, holdout = split_holdout(data_dir / "dataset.csv")
    train.to_csv(train_path, index=False)
    holdout.to_csv(holdout_path, index=False)
    return train, holdout
