import numpy as np
from SVM.utility.Enum import KernelType
from scipy.special import softplus


class SupportVectorRegression:
    """
    Support Vector Regression trained via Nesterov Smoothed Optimization
    (Nesterov, 2005) on the smoothed dual objective.
    """

    def __init__(self,
                 C: float = 1.0,
                 epsilon: float = 0.1,
                 mu: float = None,
                 kernel_type: KernelType = KernelType.RBF,
                 sigma: float = 1.0,
                 degree: int = 3,
                 coef: float = 1.0,
                 max_iter: int = 500,
                 tol: float = 1e-6):
        # regularization parameter
        self.C = C
        # width of the ε-insensitive tube
        self.epsilon = epsilon
        # smoothing parameter (if None, set to 1/n in fit)
        self.mu = mu
        # kernel selection
        self.kernel_type = kernel_type
        # RBF bandwidth
        self.sigma = sigma
        # polynomial degree
        self.degree = degree
        # polynomial coefficient
        self.coef = coef
        # maximum number of outer iterations
        self.max_iter = max_iter
        # convergence tolerance
        self.tol = tol

        # placeholders
        self.X_train = None
        self.Y_train = None
        self.beta = None
        self.b = None
        # will hold the Q_mu curve
        self.training_history = None

    def _kernel_matrix(self, X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
        """
        Compute the Gram matrix between X and Y.
        """
        if Y is None:
            Y = X

        if self.kernel_type == KernelType.LINEAR:
            return X @ Y.T
        elif self.kernel_type == KernelType.POLYNOMIAL:
            return (X @ Y.T + self.coef) ** self.degree
        elif self.kernel_type == KernelType.RBF:
            X2 = np.sum(X ** 2, axis=1)[:, None]
            Y2 = np.sum(Y ** 2, axis=1)[None, :]
            d2 = X2 + Y2 - 2 * (X @ Y.T)
            return np.exp(-d2 / (2 * self.sigma ** 2))
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")



    @staticmethod
    def _project_box_sum_zero(v: np.ndarray, C: float, tol: float = 1e-12) -> np.ndarray:
        """
        Project onto { sum(v)=0, -C <= v_i <= C } by clipping and redistributing.
        """
        u = np.clip(v, -C, C)
        s = u.sum()
        if abs(s) < tol:
            return u
        for _ in range(10):
            mask = (u > -C + tol) & (u < C - tol)
            if not np.any(mask):
                u -= s / len(u)
                break
            s = u.sum()
            delta = s / mask.sum()
            u[mask] -= delta
            if abs(u.sum()) < tol:
                break
        return u

    # helper function for smoothed ε-insensitive loss
    @staticmethod
    def smoothed_epsilon_insensitive_loss(residuals, mu, epsilon):
        abs_r = np.abs(residuals)
        x = (abs_r - epsilon) / mu
        return mu * np.sum(softplus(x))

    @staticmethod
    def _smooth_abs_derivative(beta: np.ndarray, epsilon: float, mu: float) -> np.ndarray:
        abs_beta = np.abs(beta)
        x = (abs_beta - epsilon) / mu
        # versione numericamente stabile
        sigmoid = np.where(x >= 0,
                           1.0 / (1.0 + np.exp(-x)),
                           np.exp(x) / (1.0 + np.exp(x)))
        return np.sign(beta) * sigmoid

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SupportVectorRegression":
        """
        Train by minimizing the smoothed dual via Nesterov acceleration.
        Records beta_norms, grad_norms, Q_mu and duality gap at each iteration.
        """
        # store training data
        self.X_train = X.copy()
        self.Y_train = y.copy()
        n = X.shape[0]

        # 1) build Gram matrix
        K = self._kernel_matrix(X)

        # 2) compute spectral norm (largest eigenvalue) of K
        lam_max = np.max(np.linalg.eigvalsh(K))

        # 3) set smoothing parameter μ = ε / (n*C^2 + λ_max)
        if self.mu is None:
            self.mu = self.epsilon / (n * self.C ** 2 + lam_max)

        # 4) compute Lipschitz constant L = λ_max(K) + ε/μ
        L = lam_max + self.epsilon / self.mu

        # history buffers
        Q_mu_list, grad_norms, beta_norms, primal_vals = [], [], [], []

        # init Nesterov sequences
        y_k = np.zeros(n)
        z_k = np.zeros(n)
        A_k = 0.0

        for k in range(self.max_iter):
            alpha_k = (k + 1) / 2.0
            A_k1 = A_k + alpha_k
            tau_k = alpha_k / A_k1

            # extrapolation
            x_k = tau_k * z_k + (1 - tau_k) * y_k

            # gradient of -Q_mu at x_k
            grad = (
                    -self.Y_train
                    + self._smooth_abs_derivative(x_k, self.epsilon, self.mu)
                    + K @ x_k
            )
            grad_norms.append(np.linalg.norm(grad))

            # dual objective Q_mu(x_k):
            #   y^T x_k - C * μ * sum log1p(exp((|x_k|-ε)/μ)) - 0.5 x_k^T K x_k
            log_term = np.log1p(np.exp((np.abs(x_k) - self.epsilon) / self.mu))
            Q_mu = self.Y_train @ x_k - self.C * self.mu * np.sum(log_term) - 0.5 * x_k @ (K @ x_k)
            Q_mu_list.append(Q_mu)

            # proximal-gradient step
            y_k1 = self._project_box_sum_zero(x_k - grad / L, self.C)
            beta_norms.append(np.linalg.norm(y_k1 - y_k))

            # primal objective (per monitoring)
            sv = (np.abs(y_k1) > 1e-8) & (np.abs(np.abs(y_k1) - self.C) > 1e-8)
            b_k1 = np.mean(self.Y_train[sv] - (K @ x_k)[sv]) if np.any(sv) else 0.0
            f_beta = K @ x_k + b_k1
            residuals = self.Y_train - f_beta
            primal_obj = 0.5 * x_k @ (K @ x_k) + \
                         self.C * self.smoothed_epsilon_insensitive_loss(residuals, self.mu, self.epsilon)
            primal_vals.append(primal_obj)

            print(f"[Iter {k:4d}] primal={primal_obj:.4e} | dual={Q_mu:.4e} "
                  f"| Δβ={beta_norms[-1]:.4e} | grad_norm={grad_norms[-1]:.4e}")

            # momentum update
            z_k = self._project_box_sum_zero(z_k - (alpha_k / L) * grad, self.C)

            # convergence check
            if beta_norms[-1] < self.tol and grad_norms[-1] < self.tol:
                print(f"[Converged at iter {k}] Δβ={beta_norms[-1]:.2e}, grad_norm={grad_norms[-1]:.2e}")
                y_k = y_k1
                break

            y_k, A_k = y_k1, A_k1

        # store final duals
        self.beta = y_k

        # final bias estimate
        sv = (np.abs(self.beta) > 1e-8) & (np.abs(np.abs(self.beta) - self.C) > 1e-8)
        self.b = (np.mean(self.Y_train[sv] - (K @ self.beta)[sv])
                  if np.any(sv) else np.mean(self.Y_train - K @ self.beta))

        # final duality gap: P - D
        f_beta_final = K @ self.beta + self.b
        residuals_final = self.Y_train - f_beta_final
        primal_final = 0.5 * self.beta @ (K @ self.beta) + \
                       self.C * self.smoothed_epsilon_insensitive_loss(residuals_final, self.mu, self.epsilon)
        log_term_f = np.log1p(np.exp((np.abs(self.beta) - self.epsilon) / self.mu))
        dual_final = self.Y_train @ self.beta - self.C * self.mu * np.sum(log_term_f) \
                     - 0.5 * self.beta @ (K @ self.beta)

        gap = primal_final - dual_final
        rel_gap = gap / (abs(primal_final) + self.epsilon)
        print(f"[✓] Final duality gap = {gap:.4e}, relative = {rel_gap:.4e}")

        # save history
        self.training_history = {
            'beta_norms': beta_norms,
            'grad_norms': grad_norms,
            'Q_mu': Q_mu_list,
            'primal_obj': primal_vals,
            'duality_gap': [p - d for p, d in zip(primal_vals, Q_mu_list)],
            'duality_gap_final': gap,
            'primal_final': primal_final,
            'dual_final': dual_final
        }

        # compute smoothed histories
        window = 50
        kernel = np.ones(window) / window
        for key in ('beta_norms', 'grad_norms', 'Q_mu', 'duality_gap'):
            arr = np.array(self.training_history[key])
            self.training_history[f'{key}_smooth'] = np.convolve(arr, kernel, mode='valid').tolist()
        self.training_history['iter_smooth'] = list(range(window // 2,
                                                          window // 2 + len(
                                                              self.training_history['beta_norms_smooth'])))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on new data.
        """
        K_test = self._kernel_matrix(X, self.X_train)
        return K_test @ self.beta + self.b
