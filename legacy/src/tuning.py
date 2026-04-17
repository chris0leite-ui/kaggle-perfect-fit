"""Hyperparameter grid search with cross-validation (MAE scoring)."""

import itertools

from src.evaluate import cross_validate_model


def grid_search_cv(builder_fn, param_grid, train_df, n_splits=5, seed=42):
    """Run grid search over param_grid using builder_fn to create models.

    Args:
        builder_fn: callable(**params) -> sklearn Pipeline
        param_grid: dict of {param_name: [values]}
        train_df: training DataFrame with 'target' column
        n_splits: number of CV folds
        seed: random seed for CV

    Returns:
        dict with keys: best_params, best_score, all_results
        all_results is a list of {params, cv_mean, cv_std, cv_scores}
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    all_results = []
    best_score = float("inf")
    best_params = None

    for combo in combos:
        params = dict(zip(keys, combo))
        model = builder_fn(**params)
        cv_result = cross_validate_model(model, train_df, n_splits=n_splits,
                                         seed=seed)
        entry = {
            "params": params,
            "cv_mean": cv_result["cv_mean"],
            "cv_std": cv_result["cv_std"],
            "cv_scores": cv_result["cv_scores"],
        }
        all_results.append(entry)

        if cv_result["cv_mean"] < best_score:
            best_score = cv_result["cv_mean"]
            best_params = params

    return {
        "best_params": best_params,
        "best_score": best_score,
        "all_results": all_results,
    }
