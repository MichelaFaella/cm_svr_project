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
    def _project_box_sum_zero(v: np.ndarray, C: float, tol: float = 1e-6) -> np.ndarray:
        
        """Project onto { sum(v)=0, -C <= v_i <= C } by clipping and redistributing."""
        
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

     # NEW: Function for f_mu(x) from paper (smoothing of |x|)
    def f_mu(self, x, mu):
        abs_x = np.abs(x)
        val = np.empty_like(x, dtype=float)
        
        mask_le_mu = abs_x <= mu # |x| <= mu
        val[mask_le_mu] = x[mask_le_mu]**2 / (2 * mu)
        
        mask_gt_mu = abs_x > mu # |x| > mu
        val[mask_gt_mu] = abs_x[mask_gt_mu] - mu / 2
        return val

    # NEW: Function for f_mu'(x) from paper (derivative of f_mu(x))
    def _grad_f_mu(self, x, mu):
        abs_x = np.abs(x)
        grad_val = np.empty_like(x, dtype=float)
        
        mask_le_mu = abs_x <= mu
        grad_val[mask_le_mu] = x[mask_le_mu] / mu
        
        mask_gt_mu = abs_x > mu
        grad_val[mask_gt_mu] = np.sign(x[mask_gt_mu])
        
        return grad_val

    def _smooth_abs_derivative(self, x, epsilon, mu):
        # stabile: evita overflow quando x è grande negativo
        out = np.empty_like(x)
        pos = x >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[neg])
        out[neg] = ex / (1.0 + ex)
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SupportVectorRegression":
        
        """Train the SVR by minimizing the smoothed dual via accelerated proximal‐gradient (Nesterov).
        Records beta_norms, grad_norms, Q_mu at each iteration.
       """
        # 1) store training data
        self.X_train = X.copy()
        self.Y_train = y.copy()
        n = X.shape[0]

        # 2) build Gram matrix and its spectral norm
        K = self._kernel_matrix(X)
        lam_max = np.max(np.linalg.eigvalsh(K))
        print("lam_ma: ", lam_max)


        # 3) choose a theoretically sound smoothing parameter μ
        if self.mu is None:
            # A robust choice for μ is one that balances the Lipschitz constants of the
            # quadratic term (λ_max) and the smoothed term (ε/μ).
            # Setting λ_max ≈ ε/μ gives a simple and stable choice for μ.
            # A small tolerance is added to avoid division by zero if lam_max is 0.
            self.mu = self.epsilon / (n * self.C ** 2 + lam_max)
            print(f"[SVR] μ chosen to balance Lipschitz constants: {self.mu:.3e}")

        # 4) compute the theoretically correct Lipschitz constant of ∇Q_μ
        # The correct formula is L = λ_max(K) + ε/μ
        L = lam_max + self.epsilon / self.mu
        # Note: With the choice of μ above, L simplifies to ≈ 2 * lam_max.
        print(f"[DIAGNOSTIC] n={n}, C={self.C}, eps={self.epsilon}, eps/mu={self.epsilon / self.mu}, lam_max={lam_max:.4e}, mu={self.mu:.4e}, L={L:.4e}")
        print(f"Lipschitz constant L = ||K|| + ε/μ = {L:.3e}")

        # initialize history buffers
        beta_norms, grad_norms, Q_mu_list = [], [], []

        # initialize Nesterov variables
        y_k = np.zeros(n)
        z_k = np.zeros(n)
        A_k = 0.0

        for k in range(self.max_iter):
            # extrapolation weights
            alpha_k = (k + 1) / 2.0
            A_k1 = A_k + alpha_k
            tau_k = alpha_k / A_k1

            # extrapolate
            x_k = tau_k * z_k + (1 - tau_k) * y_k

            # gradient of -Q_μ at x_k
            # CORRECTED: Using _grad_f_mu_paper and epsilon factor
            grad_fmu_term = self.epsilon * self._grad_f_mu(x_k, self.mu) 
            grad = -self.Y_train + grad_fmu_term + K @ x_k 
            grad_norms.append(np.linalg.norm(grad))

            # proximal step
            y_k1 = self._project_box_sum_zero(x_k - grad / L, self.C)
            beta_norms.append(np.linalg.norm(y_k1 - y_k))

        
            true_l1 = np.sum(np.abs(y_k1))
            smoothed_l1 = np.sum(self.f_mu(y_k1, self.mu))
            print(f"True L1: {true_l1:.4f}, Smoothed L1: {smoothed_l1:.4f}")

            print(f"Sum of beta: {np.sum(y_k1):.6e}")

            # dual_smoothed Q_μ at y_k1
            Q_mu = (
                    self.Y_train @ y_k1
                    - self.epsilon * np.sum(self.f_mu(y_k1, self.mu)) # Term from paper's dual
                    - 0.5 * y_k1 @ (K @ y_k1)
            )
            Q_mu_list.append(Q_mu)

            # logging
            print(
                f"[Iter {k:4d}]  dual={Q_mu:.4e} | "
                f"Δβ={beta_norms[-1]:.4e} | grad_norm={grad_norms[-1]:.4e} | "
            )

            # momentum update
            z_k = self._project_box_sum_zero(z_k - (alpha_k / L) * grad, self.C)

            # convergence check
            if beta_norms[-1] < self.tol and grad_norms[-1] < self.tol:
                print(f"[Converged at iter {k}] Δβ={beta_norms[-1]:.2e}, grad_norm={grad_norms[-1]:.2e}")
                y_k = y_k1
                break

            y_k, A_k = y_k1, A_k1

        # store final duals and bias
        self.beta = y_k
        sv = np.where((self.beta > - self.C + 1e-6) & (self.beta < self.C - 1e-6))[0]
        self.b = (
            np.mean(self.Y_train[sv] - (K @ self.beta)[sv])
            if np.any(sv) else np.mean(self.Y_train - K @ self.beta)
        )

        
        self.training_history = {
            'beta_norms': beta_norms,
            'grad_norms': grad_norms,
            'Q_mu': Q_mu_list,
        }

        # compute smoothed histories and iter_smooth
        window = 50
        kernel = np.ones(window) / window
        for key in ('beta_norms', 'grad_norms', 'Q_mu'):
            arr = np.array(self.training_history[key])
            self.training_history[f'{key}_smooth'] = np.convolve(arr, kernel, mode='valid').tolist()
        start = window // 2
        end = start + len(self.training_history['beta_norms_smooth'])
        self.training_history['iter_smooth'] = list(range(start, end))

        return self

    """def fit(self, X: np.ndarray, y: np.ndarray) -> "SupportVectorRegression":
        
        Train the SVR by minimizing the smoothed dual via accelerated proximal‐gradient (Nesterov),
        using continuation strategy to gradually decrease μ.
        
        self.X_train = X.copy()
        self.Y_train = y.copy()
        n = X.shape[0]
        K = self._kernel_matrix(X)
        lam_max = np.max(np.linalg.eigvalsh(K))
        self.K_train = K  # optionally cache

        # Set continuation parameters
        gamma = 0.5
        mu_start = 0.05
        mu_min = self.epsilon / (n * self.C**2 + lam_max)
        max_stages = 10
        print(f"[Continuation] Starting μ: {mu_start:.2e}, Target μ: {mu_min:.2e}, Decrease factor: {gamma}")

        # Initial β
        beta = np.zeros(n)

        # Initialize history tracking
        full_beta_norms, full_grad_norms, full_Q_mu_list = [], [], []

        for stage in range(max_stages):
            mu = mu_start * (gamma ** stage)
            if mu < mu_min:
                print(f"[Continuation] Reached μ < μ_min ({mu:.2e} < {mu_min:.2e}). Stopping.")
                break

            print(f"\n[Stage {stage + 1}] Optimizing with μ = {mu:.3e}")
            L = lam_max + self.epsilon / mu
            print(f"[Stage {stage + 1}] Lipschitz L = {L:.3e}")

            # Reuse previous β as warm-start
            y_k = beta.copy()
            z_k = beta.copy()
            A_k = 0.0

            beta_norms, grad_norms, Q_mu_list = [], [], []

            for k in range(self.max_iter):
                alpha_k = (k + 1) / 2.0
                A_k1 = A_k + alpha_k
                tau_k = alpha_k / A_k1
                x_k = tau_k * z_k + (1 - tau_k) * y_k

                grad = -self.Y_train + self.epsilon * self._grad_f_mu(x_k, mu) + K @ x_k
                grad_norms.append(np.linalg.norm(grad))

                y_k1 = self._project_box_sum_zero(x_k - grad / L, self.C)
                beta_norms.append(np.linalg.norm(y_k1 - y_k))

                Q_mu = (
                    self.Y_train @ y_k1
                    - self.epsilon * np.sum(self.f_mu(y_k1, mu))
                    - 0.5 * y_k1 @ (K @ y_k1)
                )
                Q_mu_list.append(Q_mu)

                true_l1 = np.sum(np.abs(y_k1))
                smoothed_l1 = np.sum(self.f_mu(y_k1, mu))
                print(f"[Iter {k:4d}] dual={Q_mu:.4e} | Δβ={beta_norms[-1]:.4e} | grad_norm={grad_norms[-1]:.4e} | True L1={true_l1:.4f} | Smoothed L1={smoothed_l1:.4f}")

                z_k = self._project_box_sum_zero(z_k - (alpha_k / L) * grad, self.C)

                if beta_norms[-1] < self.tol and grad_norms[-1] < self.tol:
                    print(f"[Converged at iter {k}] Δβ={beta_norms[-1]:.2e}, grad_norm={grad_norms[-1]:.2e}")
                    y_k = y_k1
                    break

                y_k, A_k = y_k1, A_k1

            # Store for next continuation stage
            beta = y_k

            # Aggregate all history across continuation stages
            full_beta_norms.extend(beta_norms)
            full_grad_norms.extend(grad_norms)
            full_Q_mu_list.extend(Q_mu_list)

        # Final model
        self.beta = beta
        sv = np.where((self.beta > -self.C + 1e-6) & (self.beta < self.C - 1e-6))[0]
        self.b = (
            np.mean(self.Y_train[sv] - (K @ self.beta)[sv])
            if np.any(sv) else np.mean(self.Y_train - K @ self.beta)
        )

        self.training_history = {
            'beta_norms': full_beta_norms,
            'grad_norms': full_grad_norms,
            'Q_mu': full_Q_mu_list,
        }

        # Smoothed metrics
        window = 50
        kernel = np.ones(window) / window
        for key in ('beta_norms', 'grad_norms', 'Q_mu'):
            arr = np.array(self.training_history[key])
            self.training_history[f'{key}_smooth'] = np.convolve(arr, kernel, mode='valid').tolist()
        start = window // 2
        end = start + len(self.training_history['beta_norms_smooth'])
        self.training_history['iter_smooth'] = list(range(start, end))

        return self"""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on new data.
        """
        K_test = self._kernel_matrix(X, self.X_train)
        return K_test @ self.beta + self.b
