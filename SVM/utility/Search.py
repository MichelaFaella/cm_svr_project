import itertools
import random

from sklearn.metrics import mean_squared_error

from SVM.Svr import SupportVectorRegression


def grid_search_svr(X_train, y_train, X_val, y_val, param_grid):
    """
    Performs an exhaustive grid search for SVR hyperparameter optimization.

    Parameters:
    - X_train, y_train: Training dataset.
    - X_val, y_val: Validation dataset.
    - param_grid: Dictionary containing hyperparameter ranges.

    Returns:
    - best_params: Best hyperparameter combination.
    - best_score: Best validation MSE.
    """
    best_score = float("inf")
    best_params = None

    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for params in param_combinations:
        model = SupportVectorRegression(
            C=params["C"],
            epsilon=params["epsilon"],
            kernel_type=params["kernel_type"],
            sigma=params.get("sigma", 1.0),
            degree=params.get("degree", 3),
            coef=params.get("coef", 1),
            learning_rate=params["learning_rate"],
            momentum=params.get("momentum", 0.9)  # default momentum if not provided
        )

        model.fit(X_train, y_train)

        y_pred_val = model.predict(X_val)
        mse_val = mean_squared_error(y_val, y_pred_val)

        print(f"Params: {params}, Val MSE: {mse_val:.4f}")

        if mse_val < best_score:
            best_score = mse_val
            best_params = params

    return best_params, best_score


def random_search_svr(X_train, y_train, X_val, y_val, param_grid, n_iter=10):
    """
    Performs randomized search for SVR hyperparameter optimization.

    Parameters:
    - X_train: Training dataset features.
    - y_train: Training targets.
    - X_val: Validation features.
    - y_val: Validation targets.
    - param_grid: Dictionary containing hyperparameter ranges.
    - n_iter: Number of random combinations to test.

    Returns:
    - best_params: Dictionary with the best combination.
    - best_score: Best validation MSE.
    """
    best_score = float("inf")
    best_params = None

    keys = list(param_grid.keys())

    for _ in range(n_iter):
        params = {key: random.choice(param_grid[key]) for key in keys}

        model = SupportVectorRegression(
            C=params["C"],
            epsilon=params["epsilon"],
            kernel_type=params["kernel_type"],
            sigma=params.get("sigma", 1.0),
            degree=params.get("degree", 3),
            coef=params.get("coef", 1),
            learning_rate=params["learning_rate"],
            momentum=params.get("momentum", 0.9)  # add momentum
        )

        model.fit(X_train, y_train)

        y_pred_val = model.predict(X_val)
        mse_val = mean_squared_error(y_val, y_pred_val)

        print(f"Params: {params}, Val MSE: {mse_val:.4f}")

        if mse_val < best_score:
            best_score = mse_val
            best_params = params

    return best_params, best_score
