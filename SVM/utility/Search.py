import itertools
from sklearn.metrics import mean_squared_error
from SVM.Svr import SupportVectorRegression
import random

from SVM.utility.Enum import LossFunctionType


def grid_search_svr(X_train, y_train, X_val, y_val, param_grid, loss_type=LossFunctionType.HUBER):
    """
    Performs an exhaustive grid search for hyperparameter optimization.

    Parameters:
    - X_train: numpy array of shape (n_samples_train, n_features)
        The training dataset features.
    - y_train: numpy array of shape (n_samples_train,)
        The training dataset targets.
    - X_val: numpy array of shape (n_samples_val, n_features)
        The validation dataset features.
    - y_val: numpy array of shape (n_samples_val,)
        The validation dataset targets.
    - param_grid: dict
        Dictionary containing hyperparameter values to search over.
        Keys should be: "kernel_type", "C", "epsilon", "sigma", "degree", "coef".

    Returns:
    - best_params: dict
        The best hyperparameter combination found.
    - best_score: float
        The lowest Mean Squared Error (MSE) achieved.
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
            loss_function=loss_type,
            learning_rate=params.get("learning rate")
        )

        model.fit(X_train, y_train)

        # Compute training & validation predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        # Compute MSE
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_val = mean_squared_error(y_val, y_pred_val)

        print(f"Params: {params}, Train MSE: {mse_train:.4f}, Val MSE: {mse_val:.4f}")

        # If Train MSE << Val MSE, the model is overfitting!
        if mse_train < mse_val * 0.5:
            print("Overfitting detected! Reducing C and increasing epsilon...")
            params["C"] /= 2  # Reduce C
            params["epsilon"] *= 1.2  # Increase epsilon

        # Select the best model based on validation MSE
        if mse_val < best_score:
            best_score = mse_val
            best_params = params

    return best_params, best_score


def random_search_svr(X_train, y_train, X_val, y_val, param_grid, n_iter=10, loss_type=LossFunctionType.HUBER):
    """
    Performs randomized search for hyperparameter optimization.

    Parameters:
    - X_train: numpy array of shape (n_samples_train, n_features)
        The training dataset features.
    - y_train: numpy array of shape (n_samples_train,)
        The training dataset targets.
    - X_val: numpy array of shape (n_samples_val, n_features)
        The validation dataset features.
    - y_val: numpy array of shape (n_samples_val,)
        The validation dataset targets.
    - param_grid: dict
        Dictionary containing hyperparameter values to search over.
        Keys should be: "kernel_type", "C", "epsilon", "sigma", "degree", "coef".
    - n_iter: int, default=10
        The number of random parameter combinations to test.

    Returns:
    - best_params: dict
        The best hyperparameter combination found.
    - best_score: float
        The lowest Mean Squared Error (MSE) achieved.
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
            loss_function=loss_type,
            learning_rate=params.get("learning rate")
        )

        model.fit(X_train, y_train)

        # Compute training & validation predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        # Compute MSE
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_val = mean_squared_error(y_val, y_pred_val)

        print(f"Params: {params}, Train MSE: {mse_train:.4f}, Val MSE: {mse_val:.4f}")

        # If Train MSE << Val MSE, the model is overfitting!
        if mse_train < mse_val * 0.5:
            print("Overfitting detected! Reducing C and increasing epsilon...")
            params["C"] /= 2  # Reduce C
            params["epsilon"] *= 1.2  # Increase epsilon

        # Select the best model based on validation MSE
        if mse_val < best_score:
            best_score = mse_val
            best_params = params

    return best_params, best_score
