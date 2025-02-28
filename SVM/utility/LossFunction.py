import numpy as np


def huber_like_loss(y_true, y_pred, epsilon=0.1, delta=1.0):
    """
    Computes the Huber-like smoothed loss for Support Vector Regression (SVR).

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


def log_sum_exp_loss(y_true, y_pred, epsilon=0.1, beta=10):
    """
    Computes the Log-Sum-Exp (Softmax-like) loss for Support Vector Regression (SVR).

    Parameters:
    - y_true: Ground truth values (numpy array).
    - y_pred: Predicted values (numpy array).
    - epsilon: Insensitivity parameter for SVR.
    - beta: Sharpness parameter (higher values make the function behave like a hard threshold).

    Returns:
    - The computed loss as a scalar.
    """

    # Compute the difference considering the ε-insensitive margin
    diff = np.abs(y_true - y_pred) - epsilon

    # Apply the Log-Sum-Exp transformation to smooth the loss
    # - This function approximates the hinge loss in a differentiable way.
    # - When beta is large, it behaves more like the standard ε-insensitive loss.
    loss = np.mean(np.log(1 + np.exp(beta * diff)) / beta)

    # Compute the gradient
    grad = np.sign(y_pred - y_true) * (1 / (1 + np.exp(-beta * diff)))

    return loss, grad


def squared_hinge_loss(y_true, y_pred, epsilon=0.1):
    """
    Computes the squared hinge loss for Support Vector Regression (SVR).

    Parameters:
    - y_true: Ground truth values (numpy array).
    - y_pred: Predicted values (numpy array).
    - epsilon: Insensitivity parameter for SVR.

    Returns:
    - The computed loss as a scalar.
    """

    # Compute the difference considering the ε-insensitive margin
    diff = np.abs(y_true - y_pred) - epsilon

    # Apply squared hinge loss
    # - If diff < 0 (within the ε-margin), loss is 0 (no penalty).
    # - If diff > 0, loss grows quadratically (stronger penalization for larger errors).
    loss = np.mean(np.maximum(0, diff) ** 2)

    # Compute the gradient
    grad = np.where(diff > 0, np.sign(y_pred - y_true) * diff, 0)

    return loss, grad
