import numpy as np
from SVM.utility.Enum import KernelType
from SVM.utility.Kernels import compute_kernel


class SupportVectorRegression:
    def __init__(self, C=1.0, epsilon=0.1, kernel_type=KernelType.RBF, sigma=1.0,
                 degree=3, coef=1.0, learning_rate=0.01, momentum=0.9, tol=1e-5):
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
        self.beta = None
        self.b = None
        self.X_train = None
        self.mu = None

    def fit(self, X, Y, max_iter=1000):
        """
                Train the SVR model using Nesterov's smoothed gradient method.

                Parameters:
                - X: Training input data (numpy array)
                - Y: Target values (numpy array)
                - max_iter: Maximum number of optimization iterations

                Returns:
                - training_loss: List of dual objective values (Q_mu) at each iteration
                """
        self.X_train = X
        n_samples = X.shape[0]
        self.b = 0.0

        # Compute kernel matrix based on current kernel type
        K = compute_kernel(
            X, X,
            kernel_type=self.kernel_type,
            sigma=self.sigma,
            degree=self.degree,
            coef=self.coef
        )

        # Spectral norm (Lipschitz constant of the quadratic term)
        spectral_norm_K = np.linalg.norm(K, ord=2)

        # Compute smoothing parameter μ = ε / (N*C² + ||K||)
        self.mu = self.epsilon / (n_samples * (self.C ** 2) + spectral_norm_K)

        # Initialize β and velocity
        self.beta = np.zeros(n_samples)
        velocity = np.zeros_like(self.beta)

        training_loss = []

        for iteration in range(max_iter):
            prev_beta = np.copy(self.beta)

            # Nesterov extrapolation
            y_t = self.beta + self.momentum * (self.beta - prev_beta)

            # Gradient
            grad = self.compute_smooth_gradient(K, Y, y_t)

            # Update with momentum
            velocity = self.momentum * velocity - self.learning_rate * grad
            self.beta = y_t + velocity

            # Project β to box constraints
            self.beta = np.clip(self.beta, -self.C, self.C)

            # Enforce ∑β = 0 by adjusting one variable (more robust than subtracting mean blindly)
            beta_sum = np.sum(self.beta)
            if abs(beta_sum) > 1e-8:
                idx = np.argmax(np.abs(self.beta))  # pick max component
                self.beta[idx] -= beta_sum  # correct it

            # Compute bias b using support vectors
            support_indices = np.where((np.abs(self.beta) > 1e-6) & (np.abs(self.beta) < self.C))[0]
            if len(support_indices) > 0:
                self.b = np.mean(Y[support_indices] - np.dot(K[support_indices], self.beta))
            else:
                self.b = np.mean(Y - np.dot(K, self.beta))

            # Smoothed dual objective
            Q_mu = np.sum(Y * self.beta) - self.epsilon * np.sum(
                self.smooth_abs(self.beta)) - 0.5 * self.beta @ K @ self.beta
            training_loss.append(Q_mu)

            if np.linalg.norm(self.beta - prev_beta) < self.tol:
                print(f"Converged at iteration {iteration}")
                break

            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Q_mu: {Q_mu:.6f}")

        return training_loss

    def predict(self, X_test):
        """
        Predict output values for new input samples.

        Parameters:
        - X_test: Test input data (numpy array)

        Returns:
        - Predicted target values (numpy array)
        """
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
        predictions = np.clip(predictions, -3.0, 3.0)

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
