import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import os
import re
import time
from SVM.utility.Enum import KernelType
from SVM.utility.Kernels import compute_kernel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from SVM.utility.utility import preprocessData, denormalize_price, customRegressionReport, denormalize  # Your preprocessing function
from SVM.utility.Solver import SolverOutputCapture  

# Load dataset
dataset = "dataset_diamonds/diamonds_cleaned.csv"
data = pd.read_csv(dataset, sep=',', header=0)

# Sample 300 instances
data_sampled = data.sample(n=3000, random_state=42).reset_index(drop=True)

# Preprocess the dataset: apply normalization and split into train, validation, and test sets
X_train, y_train, X_val, y_val, X_test, y_test, y_mean, y_std, mean, std = preprocessData(data_sampled)

# Merge training and validation sets for training
X_train = np.vstack((X_train, X_val))
y_train = np.concatenate((y_train.flatten(), y_val.flatten()))

epsilon = 0.8  # Epsilon-insensitive margin for SVR loss function
C = 1.0  # Regularization parameter
sigma = 1.0  # Parameter for RBF kernel
kernel_type = KernelType.RBF # Kernel type (RBF, Linear, Polynomial)
degree = 3 # Degree of polynomial kernel (if used)
coef = 1.0 # Coefficient term in polynomial kernel

def solve_svr_dual(X, y, epsilon=0.1, C=1.0, sigma=1.0, kernel_type=KernelType.RBF, degree=3, coef=1.0):
    N = X.shape[0]

    # Define the kernel function (linear kernel)
    K = compute_kernel(
            X, X,
            kernel_type = kernel_type,
            sigma = sigma,
            degree = degree,
            coef = coef
        )
    
    # Define the dual variable (beta)
    beta = cp.Variable(N) 

    # Define the objective function
    linear_term = y @ beta - epsilon * cp.norm1(beta)
    quadratic_term = 0.5 * cp.quad_form(beta, cp.psd_wrap(K))

    # Objective: sum(d_i * beta_i) - epsilon * sum(|beta_i|) - 0.5 * sum(beta_i * K_ij * beta_j)
    objective = cp.Maximize(linear_term - quadratic_term)

    # Constraints
    constraints = [
        beta <= C,
        beta >= -C,
        cp.sum(beta) == 0,  # Sum of dual variables must equal zero (KKT condition)
    ]

    # Define the optimization problem
    prob = cp.Problem(objective, constraints)
    
    # Solve the optimization problem
    with SolverOutputCapture() as capture:
        prob.solve(solver=cp.ECOS,  # also try cp.CLARABEL or cp.SCS
                    verbose=True,)

    # Retrieve dual variables
    beta_val = beta.value

    # Compute bias term (b) using KKT conditions
    support_indices = np.where((np.abs(beta_val) > 1e-5) & (np.abs(beta_val) < C - 1e-5))[0]
    if len(support_indices) == 0:
        print("Warning: No support vectors found for bias estimation.")
        b = 0
    else:
        b_candidates = []
        for i in support_indices:
            b_i = y[i] - np.sum(beta_val * K[i, :]) - epsilon * np.sign(beta_val[i])
            b_candidates.append(b_i)
        b = np.mean(b_candidates)

    # For linear kernel, w is explicitly available
    w = beta_val @ X if kernel_type == KernelType.LINEAR else None

    solver_output = capture.getvalue()
    solver_stats = prob.solver_stats

    print(f"Bias (b) estimated from support vectors: {b}")
    if w is not None:
        print(f"Linear weights (w): {w}")

    return w, b, solver_output, solver_stats, beta_val, prob.value

def predict_svr(X_train, X_test, beta_val, b, kernel_type, sigma, degree, coef):
    K_test = compute_kernel(X_test, X_train, kernel_type, sigma, degree, coef)
    return K_test @ beta_val + b

#-------------- MODEL PREDICTION AND DENORMALIZATION -----------------

w_value, b_value, solver_output, solver_stats, beta, prob = solve_svr_dual(X_train, y_train, epsilon=epsilon, C=C, sigma=sigma, kernel_type=kernel_type)

# Compute predicted values for the test set
y_pred_test = predict_svr(X_train, X_test, beta, b_value, kernel_type, sigma, degree, coef)

# Denormalize predictions
y_pred_test_denorm = denormalize_price(y_pred_test, y_mean, y_std)
y_test_denorm = denormalize_price(y_test, y_mean, y_std)
X_train_denorm = denormalize(X_train, mean, std)
X_test_denorm = denormalize(X_test, mean, std)

# ---------------- FEATURE SELECTION FOR VISUALIZATION -----------------

# Select meaningful feature to visualize (e.g., "carat")
feature_name = "carat"
feature_idx = list(data_sampled.columns).index(feature_name) # Get index of the feature

# Sort test data for smooth curve plotting
sorted_idx = np.argsort(X_test_denorm[:, feature_idx])
X_test_sorted = X_test_denorm[sorted_idx]
y_test_sorted = y_test_denorm[sorted_idx]
Y_pred_test_sorted = y_pred_test_denorm[sorted_idx]

# Compute upper and lower bounds of the epsilon tube
epsilon_denorm = epsilon * float(y_std.iloc[0]) if isinstance(y_std, pd.Series) else epsilon * y_std
upper_bound = y_pred_test_denorm + epsilon_denorm
lower_bound = y_pred_test_denorm - epsilon_denorm

# ------------------------- SOLVER CONVERGENCE ANALYSIS -------------------------
print("\n---------------- SOLVER CONVERGENCE ANALYSIS ----------------")

# Regex to extract iteration, primal residual, dual residual, objective, convergence information
pattern = re.compile(r"\s*(\d+)\|\s*([\d.e+-]+)\s+([\d.e+-]+)\s+[\d.e+-]+\s+([\d.e+-]+)")

# List to store extracted values
iterations = []
primal_residuals = []
dual_residuals = []
objective_values = []

# Iterate through the solver output and extract relevant information
for match in pattern.finditer(solver_output):
    iteration = int(match.group(1))             # Iteration number
    primal_residual = float(match.group(2))     # Primal feasibility residual
    dual_residual = float(match.group(3))       # Dual feasibility residual
    objective_value = float(match.group(4))     # Objective function value

    # Append extracted values to the lists
    iterations.append(iteration)
    primal_residuals.append(primal_residual)
    dual_residuals.append(dual_residual)
    objective_values.append(objective_value)

# -------------- PLOTTING CONVERGENCE ANALYSIS -----------------
print("\n---------------- PLOTTING CONVERGENCE ANALYSIS ----------------")

os.makedirs("plots/cvxpy", exist_ok=True)

# Plot convergence of the objective function over iterations
plt.figure(figsize=(8, 5))
plt.plot(objective_values, label="Objective Function", marker='o', linestyle='-')
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.title("Convergence of CVXPY Solver (SCS)")
plt.legend()
plt.grid(True)
timestamp = time.strftime("%Y%m%d-%H%M%S")
plt.savefig(f"plots/cvxpy/svr_cvxpy_obj_{timestamp}.png")
plt.show()

# Plot primal & dual residuals to analyze feasibility
plt.figure(figsize=(8, 5))
plt.plot(iterations, primal_residuals, label="Primal Residual", marker='o', linestyle='-')
plt.plot(iterations, dual_residuals, label="Dual Residual", marker='s', linestyle='--')
plt.xlabel("Iteration")
plt.ylabel("Residual Value")
plt.title("Convergence of Primal & Dual Residuals")
plt.yscale('log')  # Log scale for better visualization
plt.legend()
plt.grid(True)
timestamp = time.strftime("%Y%m%d-%H%M%S")
plt.savefig(f"plots/cvxpy/svr_cvxpy_prim_dual_{timestamp}.png")
plt.show()


# ------------------------- EPSILON TUBE PLOTS (REALISTIC CURVE) -------------------------
print("\n---------------- PLOTTING SVR CURVE ON SIGNIFICANT FEATURE ----------------")

# Scatter plot for test set predictions vs true values
plt.figure(figsize=(10, 6))
plt.scatter(X_test_sorted[:, feature_idx], y_test_sorted, alpha=0.4, color="gray", label="True Data (Validation)")
plt.plot(X_test_sorted[:, feature_idx], Y_pred_test_sorted, color='blue', linewidth=2, label="SVR Prediction")
plt.plot(X_test_sorted[:, feature_idx], Y_pred_test_sorted + epsilon_denorm, 'r--', label="+\u03b5 Tube")
plt.plot(X_test_sorted[:, feature_idx], Y_pred_test_sorted - epsilon_denorm, 'r--', label="-\u03b5 Tube")
plt.fill_between(X_test_sorted[:, feature_idx],
                 Y_pred_test_sorted - epsilon_denorm,
                 Y_pred_test_sorted + epsilon_denorm,
                 color='red', alpha=0.1)
plt.xlabel(f"{feature_name} (denormalized)")
plt.ylabel("Predicted Price")
plt.title(f"SVR + Epsilon Tube on Validation Set [{feature_name}]")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot with a timestamp
timestamp = time.strftime("%Y%m%d-%H%M%S")
plt.savefig(f"plots/cvxpy/svr_cvxpy_{timestamp}.png")
plt.show()

# ------------------------- CUSTOM REGRESSION REPORT -------------------------
print("\n---------------- CUSTOM REGRESSION REPORT ----------------")
customRegressionReport(y_test_denorm, y_pred_test_denorm, name="CVXPy SVR")
