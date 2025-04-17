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
        Train the SVR model using the original Nesterov (2005) smoothed optimization method
        with dual averaging (algorithm 3.11 in the paper).
        """

        self.X_train = X
        n_samples = X.shape[0]
        self.b = 0.0  # initialize bias term

        # Compute kernel matrix
        K = compute_kernel(
            X, X,
            kernel_type=self.kernel_type,
            sigma=self.sigma,
            degree=self.degree,
            coef=self.coef
        )

        # Estimate smoothing parameter μ by balancing the nonsmooth (|β|) and smooth (βᵗKβ) parts of the objective.
        # According to Nesterov (2005), the gradient Lipschitz constant of the smoothed function is:
        #     L_total = ε / μ + ||K||,
        # where ||K|| is the spectral norm of the kernel matrix (from the smooth quadratic term).
        # To ensure stability and efficient convergence, we solve for μ:
        #     μ = ε / (n C² + ||K||),
        # where n is the number of training samples.
        spectral_norm_K = np.linalg.norm(K, ord=2)
        self.mu = self.eps / (n_samples * self.C ** 2 + spectral_norm_K)

        # Initialize β and gradient accumulator
        self.beta = np.zeros(n_samples)
        s = np.zeros_like(self.beta)  # gradient sum accumulator

        # Logs for convergence analysis
        beta_norms = []
        grad_norms = []
        training_loss = []

        q_mu_prev = None
        plateau_count = 0
        plateau_threshold = 10

        for iteration in range(1, max_iter + 1):
            # Compute gradient at current β
            grad = self.compute_smooth_gradient(K, Y, self.beta)

            # Accumulate gradients: s_k = ∑ ∇f(β_i)
            s += grad

            # Set learning rate (can be fixed or decreasing)
            eta_k = self.learning_rate  # or eta_0 / sqrt(k)

            # Dual averaging update: project the mirror step onto feasible set Q
            self.beta = self.project_box_sum_zero(-eta_k * s, self.C)

            # Compute smoothed dual objective Q_μ(β)
            Q_mu = np.sum(Y * self.beta) - self.epsilon * np.sum(
                self.smooth_abs(self.beta)) - 0.5 * self.beta @ K @ self.beta
            training_loss.append(Q_mu)

            # Track gradient norm and β difference
            grad_norm = np.linalg.norm(grad)
            grad_norms.append(grad_norm)

            if iteration > 1:
                beta_diff = np.linalg.norm(self.beta - beta_prev)
                beta_norms.append(beta_diff)
            beta_prev = self.beta.copy()

            # Estimate bias term (b) using support vectors in (0, C)
            support_indices = np.where((np.abs(self.beta) > 1e-6) & (np.abs(self.beta) < self.C))[0]
            if len(support_indices) > 0:
                residuals = Y[support_indices] - np.dot(K[support_indices], self.beta)
                self.b = np.median(residuals)
            else:
                print("[WARNING] No support vectors in (0, C). Using full dataset for bias estimation.")
                residuals = Y - np.dot(K, self.beta)
                self.b = np.median(residuals)

            # Convergence check: early stopping on Q_mu plateau
            if q_mu_prev is not None and abs(Q_mu - q_mu_prev) < 1e-4:
                plateau_count += 1
                if plateau_count >= plateau_threshold:
                    print(f"[EARLY STOPPING] Q_mu plateaued for {plateau_threshold} iterations.")
                    break
            else:
                plateau_count = 0
            q_mu_prev = Q_mu

            # Stop if gradient is small and beta is stable
            if grad_norm < 1.0 and iteration > 1 and beta_diff < self.tol:
                print(f"[CONVERGENCE] Iteration {iteration} | β change: {beta_diff:.2e}")
                break

            # Logging
            if iteration % 10 == 0:
                print(f"Iter {iteration}| mu: {self.mu} | ∥grad∥: {grad_norm:.6f} | Q_mu: {Q_mu:.6f} | ∑β: {np.sum(self.beta):.2e}")

        # Store training history
        self.training_loss = {
            "beta_norms": beta_norms,
            "grad_norms": grad_norms,
            "Q_mu": training_loss
        }

        return training_loss

    def predict(self, X_test):
        """
        Predict output values for test samples using trained SVR model.
        """
        if self.beta is None or self.b is None:
            raise ValueError("Model has not been trained.")

        K_test = compute_kernel(X_test, self.X_train,
                                kernel_type=self.kernel_type,
                                sigma=self.sigma,
                                degree=self.degree,
                                coef=self.coef)

        predictions = K_test @ self.beta + self.b
        print("PRED AVG:", np.mean(predictions), "MAX:", np.max(predictions), "MIN:", np.min(predictions))
        return predictions

    def smooth_abs(self, x):
        """
        Smooth approximation of the absolute value function.
        """
        abs_x = np.abs(x)
        return np.where(
            abs_x <= self.mu,
            (x ** 2) / (2 * self.mu),
            abs_x - (self.mu / 2)
        )

    def smooth_abs_derivative(self, x):
        """
        Derivative of the smooth approximation of |x|.
        """
        abs_x = np.abs(x)
        return np.where(
            abs_x <= self.mu,
            x / self.mu,
            np.sign(x)
        )

    def compute_smooth_gradient(self, K, d, beta):
        """
        Compute the gradient of the smoothed dual objective Q_μ.
        """
        return -d + self.epsilon * self.smooth_abs_derivative(beta) + K @ beta

    @staticmethod
    def project_box_sum_zero(v, C):
        """
        Project vector v onto the feasible region:
        -C ≤ v_i ≤ C and ∑v_i = 0

        This enforces both box and equality constraints by iterative clipping and correction.
        """
        u = np.clip(v, -C, C)
        total = np.sum(u)
        if np.abs(total) < 1e-10:
            return u

        free = (u > -C + 1e-8) & (u < C - 1e-8)
        num_free = np.count_nonzero(free)

        if num_free == 0:
            return u - total / len(u)

        adjustment = total / num_free
        u[free] -= adjustment

        return SupportVectorRegression.project_box_sum_zero(u, C)

