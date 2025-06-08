import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import re
import time
import cvxpy as cp

from SVM.utility.Enum import KernelType
from SVM.utility.Kernels import compute_kernel
from SVM.utility.utility import preprocessData, denormalize_price, customRegressionReport, denormalize
from SVM.utility.Solver import SolverOutputCapture

# ------------------------- DATA LOADING -------------------------
print("Loading dataset...")
dataset = "dataset_diamonds/diamonds_cleaned.csv"
data = pd.read_csv(dataset, sep=',')
data_sampled = data.sample(n=2000, random_state=64).reset_index(drop=True)

# ------------------------- PREPROCESSING -------------------------
X_train, y_train, X_val, y_val, X_test, y_test, y_mean, y_std, mean, std = preprocessData(data_sampled)
y_train = y_train.flatten()
y_val = y_val.flatten()
y_test = y_test.flatten()

# ------------------------- MERGE TRAIN + VAL -------------------------
X_train_final = np.vstack((X_train, X_val))
y_train_final = np.concatenate((y_train, y_val))

# ------------------------- HYPERPARAMETERS -------------------------
C = 0.1
epsilon = 0.8
sigma = 1.0
kernel_type = KernelType.RBF
degree = 1
coef = 2


# ------------------------- SOLVE SVR DUALE -------------------------
def solve_svr_dual(X, y, epsilon, C, sigma, kernel_type, degree, coef, max_iter=1000, tol=1e-6):
    N = X.shape[0]
    K = compute_kernel(X, X, kernel_type, sigma, degree, coef)
    beta = cp.Variable(N)

    constraints = [
        beta <= C,
        beta >= -C,
        cp.sum(beta) == 0,
    ]

    # Minimizzazione equivalente alla precedente massimizzazione
    linear = y @ beta - epsilon * cp.norm1(beta)
    quadratic = 0.5 * cp.quad_form(beta, cp.psd_wrap(K))
    objective = cp.Minimize(quadratic - linear)

    prob = cp.Problem(objective, constraints)

    beta_val_prev = np.zeros(N)
    beta_history = []
    grad_norms = []
    update_norms = []
    Q_vals = []

    for it in range(max_iter):
        if beta.value is not None:
            beta.value = beta_val_prev

        # --- TIMER START ---
        t0 = time.perf_counter()
        with SolverOutputCapture() as capture:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=True)
        wall_time = time.perf_counter() - t0
        print(f"[Iter {it}] Solve time: {wall_time:.3f} seconds")
        # --- TIMER END ---

        beta_val = beta.value

        # Compute dual objective value Q(β)
        Q = y @ beta_val - epsilon * np.linalg.norm(beta_val, 1) - 0.5 * beta_val @ (K @ beta_val)
        Q_vals.append(Q)

        # Compute gradient ∇Q(β) = y - epsilon * sign(β) - Kβ
        # Handle zero β entries carefully for sign:
        sign_beta = np.sign(beta_val)
        grad = y - epsilon * sign_beta - K @ beta_val
        grad_norm = np.linalg.norm(grad)
        grad_norms.append(grad_norm)

        # Compute update norm ∥β^(t) - β^(t-1)∥
        update_norm = np.linalg.norm(beta_val - beta_val_prev)
        update_norms.append(update_norm)

        beta_history.append(beta_val.copy())

        print(f"[Iter {it}] Q={Q:.4e} | grad_norm={grad_norm:.4e} | Δβ={update_norm:.4e}")

        if update_norm < tol:
            print(update_norm)
            print(f"Converged at iteration {it} with Δβ={update_norm:.2e}")
            break

        beta_val_prev = beta_val

    support_indices = np.where((np.abs(beta_val) > 1e-5) & (np.abs(beta_val) < C - 1e-5))[0]
    b = np.mean([y[i] - np.sum(beta_val * K[i, :]) - epsilon * np.sign(beta_val[i]) for i in support_indices]) if len(
        support_indices) > 0 else 0

    history = {
        'beta_history': beta_history,
        'grad_norms': grad_norms,
        'update_norms': update_norms,
        'Q_vals': Q_vals,
    }

    return beta_val, b, capture.getvalue(), history


def predict_svr(X_train, X_test, beta, b, kernel_type, sigma, degree, coef):
    K_test = compute_kernel(X_test, X_train, kernel_type, sigma, degree, coef)
    return K_test @ beta + b


# ------------------------- TRAINING + PREDICTION -------------------------
print("\n---------------- TRAINING ----------------")
start_time_final = time.perf_counter()
beta, b, solver_output, history = solve_svr_dual(X_train_final, y_train_final, epsilon, C, sigma, kernel_type, degree,
                                                 coef)
end_time_final = time.perf_counter()
training_time_final = end_time_final - start_time_final
print(f"Training time CVXPY: {training_time_final:.4f} seconds")

print("\n---------------- TEST PREDICTION ----------------")
Y_pred_test = predict_svr(X_train_final, X_test, beta, b, kernel_type, sigma, degree, coef)

y_test_denorm = denormalize_price(y_test, y_mean, y_std)
Y_pred_test_denorm = denormalize_price(Y_pred_test, y_mean, y_std)
X_test_denorm = denormalize(X_test, mean, std)

# ------------------------- EPSILON TUBE PLOT -------------------------
print("\n---------------- PLOTTING SVR CURVE ON SIGNIFICANT FEATURE ----------------")
timestamp = time.strftime("%Y%m%d-%H%M%S")
os.makedirs("plots/cvxpy", exist_ok=True)

feature_name = "carat"
feature_idx = list(data_sampled.columns).index(feature_name)
sorted_idx = np.argsort(X_test_denorm[:, feature_idx])
X_sorted = X_test_denorm[sorted_idx]
y_sorted = y_test_denorm[sorted_idx]
Y_pred_sorted = Y_pred_test_denorm[sorted_idx]
epsilon_denorm = denormalize_price(np.array([epsilon]), y_mean, y_std)[0]

plt.figure(figsize=(10, 6))
plt.scatter(X_sorted[:, feature_idx], y_sorted, alpha=0.4, color="orange", label="True Data (Test)")
plt.plot(X_sorted[:, feature_idx], Y_pred_sorted, color='blue', label="SVR Prediction")
plt.plot(X_sorted[:, feature_idx], Y_pred_sorted + epsilon_denorm, 'r--', label="+ε Tube")
plt.plot(X_sorted[:, feature_idx], Y_pred_sorted - epsilon_denorm, 'r--', label="-ε Tube")
plt.fill_between(X_sorted[:, feature_idx],
                 Y_pred_sorted - epsilon_denorm,
                 Y_pred_sorted + epsilon_denorm,
                 color='red', alpha=0.1)
plt.xlabel(f"{feature_name} (denormalized)")
plt.ylabel("Predicted Price")
plt.title(f"CVXPY SVR + Epsilon Tube on Test Set [{feature_name}]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"plots/cvxpy/epsilon_tube_test_{timestamp}.png")
plt.show()

# ------------------------- CONVERGENCE PLOTS -------------------------
print("\n---------------- PLOTTING CONVERGENCE ----------------")

plt.plot(history['Q_vals'])
plt.title('Dual Objective Q(β) vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('Q(β)')
plt.grid(True)
plt.show()

plt.plot(history['grad_norms'])
plt.title('Gradient Norm vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('||∇Q(β)||')
plt.grid(True)
plt.show()

plt.plot(history['update_norms'])
plt.title('Update Norm ||β(t) - β(t-1)|| vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('Update Norm')
plt.grid(True)
plt.show()

# ------------------------- METRICS -------------------------
print("\n---------------- TEST METRICS ----------------")

customRegressionReport(y_test_denorm, Y_pred_test_denorm, name="CVXPy SVR (OSQP)")

print("max β:", np.max(beta))
print("min β:", np.min(beta))
print("support vectors:", np.sum((np.abs(beta) > 1e-3) & (np.abs(beta) < C)))
print("bias b:", b)

print("\nAll plots and reports saved in the 'plots/cvxpy' directory.")
