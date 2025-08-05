import numpy as np
from SVM.utility.Enum import KernelType


class SupportVectorRegression:
    """
    Support Vector Regression trained via Nesterov Smoothed Optimization
    (Nesterov, 2005) on the smoothed dual objective.
    """

    def __init__(
        self,
        C: float = 1.0,
        epsilon: float = 0.1,
        mu: float = None,
        kernel_type: KernelType = KernelType.RBF,
        sigma: float = 1.0,
        degree: int = 3,
        coef: float = 1.0,
        max_iter: int = 500,
        tol: float = 1e-6,
        tol_Q: float = 1e-8,
        patience: int = 20
    ):
        # regularization parameter
        self.C = C
        # width of the ε-insensitive tube
        self.epsilon = epsilon
        # smoothing parameter (if None, set in fit)
        self.mu = mu
        # kernel configuration
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.degree = degree
        self.coef = coef
        # optimization parameters
        self.max_iter = max_iter
        self.tol = tol
        # early-stopping on dual objective
        self.tol_Q = tol_Q
        self.patience = patience

        # placeholders for training
        self.X_train = None
        self.Y_train = None
        self.beta = None
        self.b = None
        self.training_history = None

    def _kernel_matrix(self, X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
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
    def _project_box_sum_zero(v: np.ndarray, C: float, tol: float = 1e-6) -> np.ndarray:
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

    def f_mu(self, x: np.ndarray, mu: float) -> np.ndarray:
        abs_x = np.abs(x)
        val = np.empty_like(x, dtype=float)
        mask_le = abs_x <= mu
        val[mask_le] = x[mask_le]**2 / (2 * mu)
        mask_gt = abs_x > mu
        val[mask_gt] = abs_x[mask_gt] - mu/2
        return val

    def _grad_f_mu(self, x: np.ndarray, mu: float) -> np.ndarray:
        abs_x = np.abs(x)
        grad = np.empty_like(x, dtype=float)
        mask_le = abs_x <= mu
        grad[mask_le] = x[mask_le] / mu
        mask_gt = abs_x > mu
        grad[mask_gt] = np.sign(x[mask_gt])
        return grad

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SupportVectorRegression":
        """
        Train the SVR by minimizing the smoothed dual via Nesterov accelerated gradient,
        decaying μ on-the-fly each iteration.
        Stops when:
          - ΔQ_mu < ε_machine · |Q_prev|;
          - no improvement in Q_mu for 'patience' iterations;
          - Δβ and ||grad|| both < tol.
        """
        # store training data
        self.X_train = X.copy()
        self.Y_train = y.copy()
        n = X.shape[0]

        # compute Gram matrix and its largest eigenvalue
        K = self._kernel_matrix(X)
        lam_max = np.max(np.linalg.eigvalsh(K))

        # determine μ_min and μ_start
        mu_min = self.epsilon / (n * self.C ** 2 + lam_max)
        mu_start = self.mu if self.mu is not None else 100 * mu_min
        # compute exponential decay rate so μ reaches μ_min in max_iter steps
        decay = (mu_min / mu_start) ** (1 / (self.max_iter - 1))

        beta_norms, grad_norms, Q_list = [], [], []
        Q_best = -np.inf
        no_improve = 0

        # initialize dual variables for Nesterov
        y_k = np.zeros(n)
        z_k = np.zeros(n)
        A_k = 0.0

        Q_prev = None  # for machine-epsilon stopping criterion

        for k in range(self.max_iter):
            # update μ_k and corresponding Lipschitz constant L
            mu_k = max(mu_min, mu_start * (decay ** k))
            L = lam_max + self.epsilon / mu_k

            # Nesterov extrapolation step
            alpha = (k + 1) / 2
            A_next = A_k + alpha
            tau = alpha / A_next
            x_k = tau * z_k + (1 - tau) * y_k

            # compute gradient and its norm (using μ_k)
            grad = -self.Y_train + self.epsilon * self._grad_f_mu(x_k, mu_k) + K @ x_k
            grad_norm = np.linalg.norm(grad)
            grad_norms.append(grad_norm)

            # proximal step and compute Δβ
            y_next = self._project_box_sum_zero(x_k - grad / L, self.C)
            beta_delta = np.linalg.norm(y_next - y_k)
            beta_norms.append(beta_delta)

            # compute smoothed dual objective Q_mu (with μ_k)
            Q_mu = (
                    self.Y_train @ y_next
                    - self.epsilon * np.sum(self.f_mu(y_next, mu_k))
                    - 0.5 * y_next @ (K @ y_next)
            )
            Q_list.append(Q_mu)

            # print progress
            print(f"[Iter {k:4d}] μ={mu_k:.2e} | Q_mu={Q_mu:.4e} "
                  f"| Δβ={beta_delta:.4e} | grad_norm={grad_norm:.4e}")

            # 1) stop if change in Q_mu is below machine epsilon
            if Q_prev is not None and abs(Q_mu - Q_prev) < np.finfo(float).eps * abs(Q_prev):
                print(f"[Dual-converged] |ΔQ|={abs(Q_mu - Q_prev):.2e} < ε_machine")
                break
            Q_prev = Q_mu

            # 2) early-stop if no improvement in Q_mu over 'patience' iterations
            if Q_mu - Q_best > self.tol_Q:
                Q_best = Q_mu
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.patience:
                print(f"[Early-stop] no Q_mu improvement in {self.patience} iters at k={k}")
                break

            # 3) stop if both Δβ and gradient norm fall below tol
            if beta_delta < self.tol and grad_norm < self.tol:
                print(f"[Converged] at iter {k}: Δβ={beta_delta:.2e}, grad_norm={grad_norm:.2e}")
                y_k = y_next
                break

            # momentum update
            z_k = self._project_box_sum_zero(z_k - (alpha / L) * grad, self.C)
            y_k, A_k = y_next, A_next

        # store final dual variables and bias term
        self.beta = y_k
        sv = np.where((self.beta > -self.C + 1e-6) & (self.beta < self.C - 1e-6))[0]
        self.b = (
            np.mean(self.Y_train[sv] - (K @ self.beta)[sv])
            if sv.size else np.mean(self.Y_train - K @ self.beta)
        )

        # save training history for diagnostics
        self.training_history = {
            'beta_norms': beta_norms,
            'grad_norms': grad_norms,
            'Q_mu': Q_list,
        }
        # smooth the signals for plotting
        w = 50
        filt = np.ones(w) / w
        for key in ('beta_norms', 'grad_norms', 'Q_mu'):
            arr = np.array(self.training_history[key])
            self.training_history[f'{key}_smooth'] = np.convolve(arr, filt, mode='valid').tolist()
        start = w // 2
        self.training_history['iter_smooth'] = list(
            range(start, start + len(self.training_history['beta_norms_smooth']))
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        K_test = self._kernel_matrix(X, self.X_train)
        return K_test @ self.beta + self.b
