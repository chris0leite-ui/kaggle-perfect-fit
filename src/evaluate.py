"""Evaluation framework: CV, holdout split, MAE scoring, model comparison."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score

TARGET = "target"


def split_val_test(holdout_df, val_frac=0.5, seed=42):
    """Split holdout into val and test sets, stratified by City if present."""
    if "City" in holdout_df.columns:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - val_frac,
                                     random_state=seed)
        val_idx, test_idx = next(sss.split(holdout_df, holdout_df["City"]))
    else:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(holdout_df))
        n_val = int(len(holdout_df) * val_frac)
        val_idx, test_idx = idx[:n_val], idx[n_val:]

    val = holdout_df.iloc[val_idx].reset_index(drop=True)
    test = holdout_df.iloc[test_idx].reset_index(drop=True)
    return val, test


def score_mae(y_true, y_pred):
    """Compute Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def cross_validate_model(model, train_df, n_splits=5, seed=42):
    """Run k-fold CV, return {cv_scores, cv_mean, cv_std} (MAE, lower=better)."""
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=kf,
                             scoring="neg_mean_absolute_error")
    mae_scores = -scores  # convert to positive MAE
    return {
        "cv_scores": mae_scores,
        "cv_mean": float(mae_scores.mean()),
        "cv_std": float(mae_scores.std()),
    }


def evaluate_on_holdout(model, train_df, val_df, target=TARGET):
    """Fit on train_df, predict on val_df, return {mae, predictions}."""
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_val = val_df.drop(columns=[target])
    y_val = val_df[target]

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mae = score_mae(y_val, preds)
    return {"mae": mae, "predictions": preds}


def compare_models(results):
    """Build a comparison table sorted by val_mae ascending.

    results: dict of {model_name: {cv_mean, cv_std, val_mae}}
    """
    rows = []
    for name, metrics in results.items():
        rows.append({
            "model": name,
            "cv_mean": metrics["cv_mean"],
            "cv_std": metrics["cv_std"],
            "val_mae": metrics["val_mae"],
        })
    df = pd.DataFrame(rows).sort_values("val_mae").reset_index(drop=True)
    return df


def final_test_evaluation(model, train_df, val_df, test_df, target=TARGET):
    """Retrain on train+val, evaluate on test. Called once for final model."""
    combined = pd.concat([train_df, val_df], ignore_index=True)
    X_train = combined.drop(columns=[target])
    y_train = combined[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = score_mae(y_test, preds)
    return {"test_mae": mae, "predictions": preds}
