import numpy as np
from SVM.utility.Enum import KernelType
from SVM.utility.Kernels import compute_kernel
import matplotlib.pyplot as plt


class SupportVectorRegression:
    def __init__(self, C=1.0, epsilon=10, eps=0.1, kernel_type=KernelType.RBF, sigma=1.0,
                 degree=3, coef=1.0, learning_rate=0.01, momentum=0.7, tol=1e-8):
        """
                Initialize the Support Vector Regression model parameters.

                Parameters:
                - C: Regularization parameter (controls trade-off between model complexity and tolerance to violations)
                - epsilon: Width of the epsilon-insensitive zone
                - kernel_type: The type of kernel to use (Linear, Polynomial, or RBF)
                - sigma: Parameter for the RBF kernel (controls width of Gaussian)
                - degree: Degree of the polynomial kernel (if used)
                - coef: Coefficient term in the polynomial kernel
                - learning_rate: Learning rate for gradient updates
                - momentum: Momentum term for Nesterov acceleration
                - tol: Convergence tolerance for stopping criterion
                """
        self.C = C
        self.epsilon = epsilon
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.degree = degree
        self.coef = coef
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.tol = tol
        self.eps = eps
        self.beta = None
        self.b = None
        self.X_train = None
        self.mu = None

    def fit(self, X, Y, max_iter=300):
        """
        Train the SVR model using Nesterov's smoothed gradient method.
        """
        self.X_train = X
        n_samples = X.shape[0]
        self.b = 0.0

        # Compute kernel matrix
        K = compute_kernel(
            X, X,
            kernel_type=self.kernel_type,
            sigma=self.sigma,
            degree=self.degree,
            coef=self.coef
        )

        # Spectral norm (Lipschitz constant)
        spectral_norm_K = np.linalg.norm(K, ord=2)

        # Compute smoothing parameter μ
        self.mu = self.eps / (n_samples * (self.C ** 2) + spectral_norm_K)


        # Initialize variables
        self.beta = np.zeros(n_samples)
        beta_prev = np.zeros_like(self.beta)
        velocity = np.zeros_like(self.beta)

        beta_norms = []
        grad_norms = []
        training_loss = []

        # Plateau check setup
        q_mu_prev = None
        plateau_count = 0
        plateau_threshold = 10

        for iteration in range(max_iter):
            # Nesterov extrapolation
            y_t = self.beta + self.momentum * (self.beta - beta_prev)
            beta_prev = self.beta.copy()

            # Compute gradient at extrapolated point
            grad = self.compute_smooth_gradient(K, Y, y_t)

            # Gradient clipping
            max_grad_norm = 20.0
            grad_norm = np.linalg.norm(grad)
            if grad_norm > max_grad_norm:
                grad = grad * (max_grad_norm / grad_norm)
            grad_norms.append(grad_norm)

            # Update β with momentum
            velocity = self.momentum * velocity + self.learning_rate * grad
            self.beta = y_t + velocity

            # Box constraint
            self.beta = np.clip(self.beta, -self.C, self.C)

            # Equality constraint
            self.beta -= np.mean(self.beta)

            # Dual objective Q_mu
            Q_mu = np.sum(Y * self.beta) - self.epsilon * np.sum(
                self.smooth_abs(self.beta)) - 0.5 * self.beta @ K @ self.beta
            training_loss.append(Q_mu)

            # Early stopping: plateau detection
            if q_mu_prev is not None:
                if abs(Q_mu - q_mu_prev) < 1e-4:
                    plateau_count += 1
                    if plateau_count >= plateau_threshold:
                        print(f"[EARLY STOPPING] Q_mu plateaued for {plateau_threshold} iterations.")
                        break
                else:
                    plateau_count = 0
            q_mu_prev = Q_mu

            # Track beta diff
            beta_diff = np.linalg.norm(self.beta - beta_prev)
            beta_norms.append(beta_diff)

            # Bias estimation
            support_indices = np.where((np.abs(self.beta) > 1e-6) & (np.abs(self.beta) < self.C))[0]
            if len(support_indices) > 0:
                residuals = Y[support_indices] - np.dot(K[support_indices], self.beta)
                self.b = np.median(residuals)
            else:
                print("[WARNING] No support vectors in (0, C). Falling back to full dataset for bias estimation.")
                residuals = Y - np.dot(K, self.beta)
                self.b = np.median(residuals)

            # Convergence check
            if beta_diff < self.tol and grad_norm < 1.0:
                print(f"[CONVERGENCE] Iteration {iteration} | β change: {beta_diff:.2e}")
                print("Unique β values:", np.unique(np.round(self.beta, 6)))
                break

            # Logging
            if iteration % 10 == 0:
                print(f"Iter {iteration} | max(β): {np.max(np.abs(self.beta)):.6f} | "
                      f"∥grad∥: {grad_norm:.6f} | Q_mu: {Q_mu:.6f} | ∑β: {np.sum(self.beta):.2e}")

        self.training_loss = {
            "beta_norms": beta_norms,
            "grad_norms": grad_norms,
            "Q_mu": training_loss
        }

        return training_loss

    def predict(self, X_test):
        if self.beta is None or self.b is None:
            raise ValueError("Model has not been trained.")

        K_test = compute_kernel(
            X_test, self.X_train,
            kernel_type=self.kernel_type,
            sigma=self.sigma,
            degree=self.degree,
            coef=self.coef
        )

        predictions = K_test @ self.beta + self.b
        print("PRED AVG:", np.mean(predictions), "MAX:", np.max(predictions), "MIN:", np.min(predictions))
        return predictions

    def smooth_abs(self, x):
        """
        Smoothed approximation of the absolute value function (fμ(x)):

        fμ(x) = (x²)/(2μ) if |x| ≤ μ
        fμ(x) = |x| - (μ/2) if |x| > μ

        Parameters:
        - x: input array

        Returns:
        - array with smoothed absolute values
        """
        abs_x = np.abs(x)
        return np.where(
            abs_x <= self.mu,
            (x ** 2) / (2 * self.mu),
            abs_x - (self.mu / 2)
        )

    def smooth_abs_derivative(self, x):
        """
        Derivative of the smoothed absolute value function (fμ'(x)):

        fμ'(x) = x/μ if |x| ≤ μ
        fμ'(x) = sign(x) if |x| > μ

        Parameters:
        - x: input array

        Returns:
        - derivative values (array)
        """
        abs_x = np.abs(x)
        return np.where(
            abs_x <= self.mu,
            x / self.mu,
            np.sign(x)
        )

    def compute_smooth_gradient(self, K, d, beta):
        """
        Compute the gradient of the smoothed dual objective Qμ(β):

        ∇Qμ(β) = d - ε * fμ'(β) - K * β

        Parameters:
        - K: kernel matrix (precomputed)
        - d: target values vector
        - beta: current value of dual variables

        Returns:
        - gradient vector
        """

        return d - self.epsilon * self.smooth_abs_derivative(beta) - K @ beta
