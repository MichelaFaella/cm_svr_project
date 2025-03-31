import numpy as np

from SVM.utility.Enum import KernelType


def rbf_kernel(x1, x2, sigma):
    """
        Computes the Radial Basis Function (RBF) kernel between two vectors.

        Parameters:
        - x1: numpy array, first vector
        - x2: numpy array, second vector
        - sigma: float, scaling parameter for the RBF kernel (default = 1.0)

        Returns:
        - float, the computed RBF kernel value
    """

    diff = np.linalg.norm(x1 - x2) ** 2
    return np.exp(-diff / (2 * sigma ** 2))


def polinomial_kernel(x1, x2, degree=2, coef=1):
    """
        Computes the polynomial kernel between two vectors.

        Parameters:
        - x1: numpy array, first vector
        - x2: numpy array, second vector
        - degree: int, degree of the polynomial kernel (default = 3)
        - coefficient: float, bias term for the polynomial kernel (default = 1)

        Returns:
        - float, the computed polynomial kernel value
        """

    dot_prod = np.dot(x1, x2)
    return (dot_prod + coef) ** degree


def linear_kernel(x1, x2):
    """
    Compute the linear kernel between two vectors.

    Parameters:
    - x1: First vector (numpy array or list).
    - x2: Second vector (numpy array or list).

    Returns:
    - The result of the dot product between x1 and x2.
    """
    return np.dot(x1, x2)


def compute_kernel(X1, X2, kernel_type, sigma=1.0, degree=3, coef=1.0):
    """
    Compute the kernel matrix between datasets X1 and X2.

    Parameters:
    - X1: numpy array (n_samples_1, n_features)
    - X2: numpy array (n_samples_2, n_features)
    - kernel_type: KernelType enum (LINEAR, POLYNOMIAL, RBF)
    - sigma: float, only for RBF kernel
    - degree: int, only for polynomial kernel
    - coef: float, only for polynomial kernel

    Returns:
    - Kernel matrix K of shape (n_samples_1, n_samples_2)
    """
    if kernel_type == KernelType.LINEAR:
        return X1 @ X2.T

    elif kernel_type == KernelType.POLYNOMIAL:
        return (X1 @ X2.T + coef) ** degree

    elif kernel_type == KernelType.RBF:
        # Efficient computation of squared Euclidean distances
        X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        dist_sq = X1_sq + X2_sq - 2 * (X1 @ X2.T)

        # Apply RBF formula with sigma^2
        return np.exp(-dist_sq / (2 * sigma ** 2))

    else:
        raise ValueError(f"Invalid kernel type: {kernel_type}")
