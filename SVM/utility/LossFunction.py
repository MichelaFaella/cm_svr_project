import numpy as np

import numpy as np

import numpy as np


def epsilon_insensitive_loss(y_true, y_pred, epsilon=0.1, mu=0.01):
    """
    Computes the smoothed ε-insensitive loss for Support Vector Regression (SVR).
    This version introduces a soft thresholding mechanism to smooth the gradient.

    Parameters:
    - y_true: Ground truth values (numpy array).
    - y_pred: Predicted values (numpy array).
    - epsilon: Insensitivity margin for SVR.
    - mu: Smoothing parameter to control gradient transition.

    Returns:
    - The computed loss as a scalar.
    - The smoothed gradient.
    """

    # Compute the absolute error minus epsilon threshold
    diff = np.abs(y_true - y_pred) - epsilon
    loss = np.where(diff > 0, diff, 0)

    # Smooth the gradient using a hyperbolic tangent function
    grad = np.tanh((y_pred - y_true) / mu)

    return np.mean(loss), grad
