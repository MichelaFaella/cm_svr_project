import numpy as np


def huber_like_loss(y_true, y_pred, epsilon=0.1, delta=1.0):
    """
    Computes the Huber-like smoothed loss for Support Vector Regression (SVR).
    Se hai dati con rumore e vuoi un compromesso tra MAE e MSE scegli questa!!!

    Parameters:
    - y_true: Ground truth values (numpy array).
    - y_pred: Predicted values (numpy array).
    - epsilon: Insensitivity parameter for SVR.
    - delta: Smoothing threshold for the quadratic-linear transition.

    Returns:
    - The computed loss as a scalar.
    """

    # Compute the difference considering the ε-insensitive margin
    diff = np.abs(y_true - y_pred) - epsilon

    # Quadratic smoothing for small errors
    # - If diff < 0 (within the ε-margin), loss is 0 (no penalty).
    # - If 0 ≤ diff ≤ delta, apply quadratic loss: (diff^2) / (2 * delta)
    loss = np.where(diff < 0, 0, 0.5 * (diff ** 2) / delta)

    # Linear loss for large errors
    # - If diff > delta, switch to linear growth: diff - (delta / 2)
    loss = np.where(diff > delta, diff - 0.5 * delta, loss)

    # Compute the gradient
    grad = np.where(diff < 0, 0, diff / delta)  # Quadratic smoothing for small errors
    grad = np.where(diff > delta, np.sign(y_pred - y_true), grad)  # Linear smoothing for large errors

    # Return the mean loss value
    return np.mean(loss), grad


def quantile_loss(y_true, y_pred, tau=0.5):
    """
        Computes the Quantile Loss (Pinball Loss) for Support Vector Regression (SVR).
        Se hai dati distribuiti in modo asimmetrico e vuoi modellare gli intervalli di confidenza usa questa!!!

        Parameters:
        - y_true: Ground truth values (numpy array).
        - y_pred: Predicted values (numpy array).
        - tau: Quantile to optimize (default 0.5 for median).

        Returns:
        - The computed loss as a scalar.
        """

    diff = y_true - y_pred
    loss = np.mean(np.maximum(tau * diff, (tau - 1) * diff))

    # Compute the gradient
    grad = np.where(diff > 0, tau, tau - 1)

    return loss, grad


def epsilon_insensitive_loss(y_true, y_pred, epsilon=0.1):
    """
    Computes the ε-insensitive loss for Support Vector Regression (SVR).
    If you want to ignore small deviations within the ε margin and penalize only large errors, choose this!!!

    Parameters:
    - y_true: Ground truth values (numpy array).
    - y_pred: Predicted values (numpy array).
    - epsilon: Insensitivity margin for SVR.

    Returns:
    - The computed loss as a scalar.
    - The gradient of the loss.
    """

    # Compute the difference ignoring small deviations within ε
    diff = np.abs(y_true - y_pred) - epsilon

    # Apply the ε-insensitive loss function
    loss = np.where(diff > 0, diff, 0)

    # Compute the gradient
    grad = np.where(y_pred > y_true + epsilon, 1, 0)
    grad = np.where(y_pred < y_true - epsilon, -1, grad)

    return np.mean(loss), grad
