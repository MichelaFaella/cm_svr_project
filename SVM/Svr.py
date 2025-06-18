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

    """@staticmethod
    def _project_box_sum_zero(v: np.ndarray, C: float) -> np.ndarray:
        
        Projects v onto { u | -C <= u_i <= C, sum(u_i) = 0 }.
        Uses an iterative algorithm to find the correct shift.
        
        n = len(v)
        v_sorted = np.sort(v)
        
        # Binary search for lambda
        low = v_sorted[0] - C
        high = v_sorted[-1] + C
        
        # Function to check the sum given lambda
        def check_sum(lambda_val):
            return np.sum(np.clip(v - lambda_val, -C, C))

        # Perform bisection
        for _ in range(100): # A sufficient number of iterations for bisection
            mid = (low + high) / 2
            current_sum = check_sum(mid)
            if abs(current_sum) < 1e-12: # Tolerance for sum close to zero
                return np.clip(v - mid, -C, C)
            elif current_sum > 0:
                low = mid
            else:
                high = mid
        
        # Fallback if bisection doesn't converge perfectly (should not happen with enough iter)
        return np.clip(v - mid, -C, C)"""
    
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
        """
        Train the SVR by minimizing the smoothed dual via accelerated proximal‐gradient (Nesterov).
        Records beta_norms, grad_norms, Q_mu, primal_obj and duality_gap at each iteration.
        """
        # 1) store training data
        self.X_train = X.copy()
        self.Y_train = y.copy()
        n = X.shape[0]

        # 2) build Gram matrix and its spectral norm
        K = self._kernel_matrix(X)
        lam_max = np.max(np.linalg.eigvalsh(K))

        # 3) choose smoothing parameter μ if not provided
        if self.mu is None:
            mu_bound = self.epsilon / (n * self.C ** 2)
            mu_kernel = self.C / lam_max
            # compromise: 1/μ = 1/mu_bound + 1/mu_kernel
            self.mu = 1.0 / (n * self.C ** 2 / self.epsilon + lam_max / self.C)
            print(f"[SVR] mu_bound={mu_bound:.3e}, mu_kernel={mu_kernel:.3e}, μ used={self.mu:.3e}")

        # 4) compute Lipschitz constant of ∇Q_μ = λ_max(K) + C/μ
        L = lam_max + self.C / self.mu
        print(f"sono L {L}")

        # initialize history buffers
        beta_norms, grad_norms, Q_mu_list, primal_vals, duality_gap = [], [], [], [], []

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

            # dual_smoothed Q_μ at y_k1
            # CORRECTED: Using f_mu_paper and epsilon factor
            Q_mu = (
                    self.Y_train @ y_k1
                    - self.epsilon * np.sum(self.f_mu(y_k1, self.mu)) # Term from paper's dual
                    - 0.5 * y_k1 @ (K @ y_k1)
            )
            Q_mu_list.append(Q_mu)

            # primal_smoothed P_μ at y_k1
            sv = (np.abs(y_k1) > 1e-8) & (np.abs(np.abs(y_k1) - self.C) > 1e-8)
            b_k1 = np.mean(self.Y_train[sv] - (K @ y_k1)[sv]) if np.any(sv) else 0.0
            f_beta = K @ y_k1 + b_k1
            residuals = self.Y_train - f_beta
            primal_obj = (
                    0.5 * y_k1 @ (K @ y_k1)
                    + self.C * self.smoothed_epsilon_insensitive_loss(residuals, self.mu, self.epsilon)
            )
            primal_vals.append(primal_obj)

            # record smoothed duality gap P_μ - Q_μ
            duality_gap.append(primal_obj - Q_mu)

            # logging
            print(
                f"[Iter {k:4d}] primal={primal_obj:.4e} | dual={Q_mu:.4e} | "
                f"Δβ={beta_norms[-1]:.4e} | grad_norm={grad_norms[-1]:.4e} | "
                f"gap_smoothed={duality_gap[-1]:.4e}"
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
        """self.beta = y_k
        sv = (np.abs(self.beta) > 1e-8) & (np.abs(np.abs(self.beta) - self.C) > 1e-8)
        self.b = (
            np.mean(self.Y_train[sv] - (K @ self.beta)[sv])
            if np.any(sv) else np.mean(self.Y_train - K @ self.beta)
        )"""

         # --- REPLACEMENT CODE ---
        # Store final dual variables from the last iteration of the optimizer
        self.beta = y_k
        K_beta_final = K @ self.beta # Pre-calculate for efficiency

        # Define a tolerance to identify support vectors away from the boundaries (0 and C)
        tol_sv = 1e-6 

        # Create a boolean mask to find all "unbounded" support vectors.
        # These are the most reliable points for calculating the bias 'b'.
        unbounded_sv_mask = (np.abs(self.beta) > tol_sv) & (np.abs(self.beta) < self.C - tol_sv)

        # Check if any unbounded support vectors were found
        if np.any(unbounded_sv_mask):
            # Extract the relevant values for the unbounded support vectors
            y_sv = self.Y_train[unbounded_sv_mask]
            K_beta_sv = K_beta_final[unbounded_sv_mask]
            beta_sv = self.beta[unbounded_sv_mask]
            
            # Calculate 'b' for each unbounded SV using the KKT condition:
            # b = y_i - (Kβ)_i - ε * sign(β_i)
            b_values = y_sv - K_beta_sv - self.epsilon * np.sign(beta_sv)
            
            # The final bias is the average of these calculated values, which is more stable
            self.b = np.mean(b_values)
            print(f"[Info] Final bias b calculated from {np.sum(unbounded_sv_mask)} unbounded support vectors.")
        else:
            # This is a fallback case for when no support vectors are found in the 'margin'.
            # This can happen if C is very small or for certain simple datasets.
            # The heuristic is less accurate but better than nothing.
            print("[Warning] No unbounded SVs found. Bias 'b' calculated with a fallback heuristic.")
            self.b = np.mean(self.Y_train - K_beta_final)
        # --- END REPLACEMENT CODE ---

        # final smoothed gap
        f_beta_final = K @ self.beta + self.b
        residuals_final = self.Y_train - f_beta_final
        primal_final = (
                0.5 * self.beta @ (K @ self.beta)
                + self.C * self.smoothed_epsilon_insensitive_loss(residuals_final, self.mu, self.epsilon)
        )
        # CORRECTED: Final dual calculation should also use f_mu_paper for consistency
        dual_final = (
                self.Y_train @ self.beta
                - self.epsilon * np.sum(self.f_mu(self.beta, self.mu))
                - 0.5 * self.beta @ (K @ self.beta)
        )
        gap_final = primal_final - dual_final
        rel_gap = gap_final / (abs(primal_final) + self.epsilon)
        print(f"[✓] Final smoothed gap = {gap_final:.4e}, relative = {rel_gap:.4e}")

        # save training history
        self.training_history = {
            'beta_norms': beta_norms,
            'grad_norms': grad_norms,
            'Q_mu': Q_mu_list,
            'primal_obj': primal_vals,
            'duality_gap': duality_gap,
            'duality_gap_final': gap_final,
            'primal_final': primal_final,
            'dual_final': dual_final,
        }

        # compute smoothed histories and iter_smooth
        window = 50
        kernel = np.ones(window) / window
        for key in ('beta_norms', 'grad_norms', 'Q_mu', 'primal_obj', 'duality_gap'):
            arr = np.array(self.training_history[key])
            self.training_history[f'{key}_smooth'] = np.convolve(arr, kernel, mode='valid').tolist()
        start = window // 2
        end = start + len(self.training_history['beta_norms_smooth'])
        self.training_history['iter_smooth'] = list(range(start, end))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on new data.
        """
        K_test = self._kernel_matrix(X, self.X_train)
        return K_test @ self.beta + self.b
