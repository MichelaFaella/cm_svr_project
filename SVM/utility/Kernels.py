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


def compute_kernel(x1, x2, kernel_type, **kwargs):
    """
        Computes the kernel matrix for input matrices X1 and X2.

        Parameters:
        - X1: numpy array of shape (n_samples_1, n_features)
            The first set of input data points.
        - X2: numpy array of shape (n_samples_2, n_features)
            The second set of input data points.
        - kernel_type: KernelType:
            The type of kernel function to use.
        - kwargs: additional parameters for the kernel function
            - If kernel_type is RBF:
                - sigma: float (default=1.0), controls the width of the Gaussian function.
            - If kernel_type is POLYNOMIAL:
                - degree: int (default=3), the degree of the polynomial.
                - coef: float (default=1), the independent term in the polynomial kernel.

        Returns:
        - K: numpy array of shape (n_samples_1, n_samples_2)
            The computed kernel matrix, where K[i, j] represents the kernel value
            between the i-th sample of X1 and the j-th sample of X2.
        """

    kernel_functions = {
        KernelType.RBF: rbf_kernel,
        KernelType.POLYNOMIAL: polinomial_kernel,
    }

    if kernel_type not in kernel_functions:
        return print(f"Invalid or null kernel type: {kernel_type}")

    n1, n2 = x1.shape[0], x2.shape[0]
    K = np.zeros((n1, n2)) # Creation of an empty matrix of dimention n1 x n2

    # Extract relevant parameters dynamically from kwargs
    kernel_params = {
        KernelType.RBF: {"sigma": kwargs.get("sigma", 1.0)},
        KernelType.POLYNOMIAL: {
            "degree": kwargs.get("degree", 3),
            "coef": kwargs.get("coef", 1),
        },
    }

    selected_params = kernel_params.get(kernel_type, {})

    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel_functions[kernel_type](x1[i], x2[j], **selected_params)

    return K