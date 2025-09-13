import numpy as np

from SVM.utility.Enum import KernelType


def rbf_kernel(x1, x2, sigma):
    """Computes the Radial Basis Function (RBF) kernel between two vectors."""

    diff = np.linalg.norm(x1 - x2) ** 2
    return np.exp(-diff / (2 * sigma ** 2))


def polinomial_kernel(x1, x2, degree=2, coef=1):
    """Computes the polynomial kernel between two vectors."""

    dot_prod = np.dot(x1, x2)
    return (dot_prod + coef) ** degree


def linear_kernel(x1, x2):
    """Compute the linear kernel between two vectors."""
    return np.dot(x1, x2)


def compute_kernel(X1, X2, kernel_type, sigma=1.0, degree=3, coef=1.0):
    """Compute the kernel matrix between datasets X1 and X2."""
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
