import numpy as np
from SVM.utility.Enum import KernelType
from SVM.utility.Kernels import compute_kernel
from scipy.optimize import minimize
from SVM.utility.utility import svr_dual_loss

class SupportVectorRegression:
    def __init__(self, C=1.0, epsilon=0.1, kernel_type=KernelType.RBF, sigma=1.0, degree=3, coef=1):
        self.C = C  # Ensure C is at least 1.0
        self.epsilon = epsilon  # Ensure epsilon is at least 0.01
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.degree = degree
        self.coef = coef
        self.alpha = None
        self.b = None
        self.X_train = None

    def fit(self, X, Y):
        """ Fits the SVR model to the training data using the dual formulation."""
        self.X_train = X
        n_samples = X.shape[0]

        # Compute kernel matrix
        K = compute_kernel(X, X, kernel_type=self.kernel_type, sigma=self.sigma, degree=self.degree, coef=self.coef)

        # Constraints for the dual optimization
        Aeq = np.hstack([np.ones(n_samples), -np.ones(n_samples)])
        beq = np.array([0.0])
        bounds = [(0, self.C) for _ in range(2 * n_samples)]
        alpha_init = np.zeros(2 * n_samples)

        # Solve the dual optimization problem
        res = minimize(svr_dual_loss, alpha_init, args=(K, Y, self.epsilon),
                       bounds=bounds, constraints={'type': 'eq', 'fun': lambda a: np.dot(Aeq, a) - beq},
                       method='SLSQP', options={'maxiter': 500})

        if not res.success:
            raise RuntimeError("Optimization failed: " + res.message)

        # Extract Lagrange multipliers
        alpha_opt = res.x
        alpha_plus, alpha_minus = alpha_opt[:n_samples], alpha_opt[n_samples:]
        self.alpha = alpha_plus - alpha_minus

        # Compute bias term b using support vectors
        support_vector_indices = np.where((self.alpha > 1e-4) & (self.alpha < self.C))[0]
        if len(support_vector_indices) > 0:
            self.b = np.mean(Y[support_vector_indices] - np.dot(K[support_vector_indices], self.alpha))
        else:
            self.b = np.median(Y - np.dot(K, self.alpha))  # Usa la mediana come backup

    def predict(self, X_test):
        """ Predicts new values using the trained SVR model. """
        if self.alpha is None or self.b is None:
            raise ValueError("Model has not been trained yet.")

        # Compute kernel matrix for test data
        K_test = compute_kernel(X_test, self.X_train, kernel_type=self.kernel_type,
                                sigma=self.sigma, degree=self.degree, coef=self.coef)

        return np.dot(K_test, self.alpha) + self.b
