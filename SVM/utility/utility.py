import numpy as np

from SVM.utility.Enum import KernelType
from SVM.utility.Kernels import compute_kernel

import numpy as np


def svr_dual_loss(alpha, K, Y, epsilon):
    """
    Computes the dual loss function for SVR optimization.

    Parameters:
    - alpha: Lagrange multipliers (vector of shape (2 * n_samples,))
    - K: Kernel matrix (n_samples x n_samples)
    - Y: Target values (n_samples,)
    - epsilon: Insensitivity parameter in SVR

    Returns:
    - The computed dual loss
    """
    n_samples = Y.shape[0]

    # Decomposing α into α⁺ and α⁻
    alpha_plus, alpha_minus = alpha[:n_samples], alpha[n_samples:]
    alpha_diff = alpha_plus - alpha_minus  # α = α⁺ - α⁻

    # Normalize kernel matrix to avoid numerical instability
    K_norm = K / np.max(K) if np.max(K) > 0 else K  # Avoid division by zero

    # Scale epsilon based on the range of Y
    epsilon_scaled = epsilon * np.max(np.abs(Y)) if np.max(np.abs(Y)) > 0 else epsilon

    # Compute the dual loss
    loss = (
            0.5 * np.dot(alpha_diff, np.dot(K_norm, alpha_diff))
            - np.sum(alpha_diff * Y)
            + epsilon_scaled * np.sum(alpha_plus + alpha_minus)
    )

    return loss



