import numpy as np
import pandas as pd
from docplex.mp.model import Model
import matplotlib.pyplot as plt
import time
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from SVM.utility.preprocess import preprocessData, denormalize_price, customRegressionReport  # Your preprocessing function

print("Loading dataset...")
dataset = "dataset_diamonds/diamonds_cleaned.csv"
data = pd.read_csv(dataset, sep=',', header=0)

# Randomly select 3000 samples
data_sampled = data.sample(n=280, random_state=42)  # Set random_state for reproducibility

# Reset index
data_sampled = data_sampled.reset_index(drop=True)

# Print first few rows
print(data_sampled.head())

# Preprocessing (Z-score normalization + train/val/test split)
X_train, y_train, X_val, y_val, X_test, y_test, y_mean, y_std = preprocessData(data_sampled)

# Merge training and validation sets
X_train = np.vstack((X_train, X_val))
y_train = np.concatenate((y_train.flatten(), y_val.flatten()))

y_train = y_train.flatten()
y_test = y_test.flatten()

# Define linear kernel
def linear_kernel(X1, X2):
    return np.dot(X1, X2.T)

#Define the SVR problem with CPLEX
def svr_cplex(X_train, y_train, C, epsilon, linear_kernel):
    n = len(y_train)  # Number of training samples

    # Compute kernel matrix K
    K = linear_kernel(X_train, X_train)

    # Create CPLEX model
    mdl = Model(name="SVR_Dual")

    # Define the dual variables beta with bounds
    beta = [mdl.continuous_var(lb=-C, ub=C, name=f'beta_{i}') for i in range(n)]

    # Introduce auxiliary variables s_i for |beta_i| (This transformation allows the optimization solver 
    # to handle the non-differentiable part of the objective (the absolute value) in a tractable manner.)
    s = [mdl.continuous_var(lb=0, name=f's_{i}') for i in range(n)]

    # Objective function (maximize dual formulation)
    objective = mdl.sum(y_train[i] * beta[i] for i in range(n)) - \
                epsilon * mdl.sum(mdl.abs(s[i]) for i in range(n)) - \
                0.5 * mdl.sum(beta[i] * beta[j] * K[i, j] for i in range(n) for j in range(n))
    mdl.maximize(objective)

    # Constraint: |beta_i| <= s_i
    for i in range(n):
        mdl.add_constraint(beta[i] <= s[i])
        mdl.add_constraint(beta[i] >= -s[i])

    # Sum of beta_i must be 0
    mdl.add_constraint(mdl.sum(beta[i] for i in range(n)) == 0)

    # Solve with CPLEX
    solution = mdl.solve()

    # Extract optimal beta
    if solution:
        beta_values = np.array([beta[i].solution_value for i in range(n)])
        print("Solution found!")
        return beta_values, K
    else:
        print("No solution found.")
        return None, K

C = 1.0  # Regularization
epsilon = 0.1  # Epsilon-insensitive margin

print("Training SVR model with CPLEX...")
beta_values, K = svr_cplex(X_train, y_train, C=C, epsilon=epsilon, linear_kernel=linear_kernel)

# Compute weight vector w
w = np.sum(beta_values[:, np.newaxis] * X_train, axis=0)

# Compute bias b using support vectors
support_vector_indices = np.where((beta_values > 0) & (beta_values < C))[0]

b_values = []
for i in support_vector_indices:
    b_i = y_train[i] - np.sum(beta_values * K[i, :])
    b_values.append(b_i)

b = np.mean(b_values) if b_values else 0  # Avoid NaN in case no support vectors are found

print(f"Computed w: {w}")
print(f"Computed b: {b}")

# -------------------------
# Make Predictions on Test Set
# -------------------------
y_pred_test = np.dot(X_test, w) + b  # Compute predictions

# Denormalize predictions
y_pred_test_denorm = denormalize_price(y_pred_test, y_mean, y_std)
y_test_denorm =  denormalize_price(y_test, y_mean, y_std)

# Sort validation data for a smooth plot
sorted_idx = np.argsort(X_test[:, 0])
X_sorted = X_test[sorted_idx]
Y_pred_sorted = y_pred_test_denorm[sorted_idx]
y_sorted = y_test_denorm[sorted_idx]

# Compute upper and lower bounds of the epsilon tube
upper_bound = y_pred_test_denorm + epsilon * float(y_std)  # Convert y_std to scalar if necessary
lower_bound = y_pred_test_denorm - epsilon * float(y_std)  # Same for the lower bound

# ---------------------------------
# Plot Results: Optimized SVR Model & Training Loss
# ---------------------------------
plt.figure(figsize=(12, 5))

# Subplot 1: SVR Predictions
plt.subplot(1, 2, 1)
plt.scatter(X_sorted[:, 0], y_sorted, color='darkorange', label='Validation Data')
plt.plot(X_sorted[:, 0], Y_pred_sorted, color='navy', lw=2, label='SVR Model')
plt.fill_between(X_sorted[:, 0], lower_bound, upper_bound, color='lightblue', alpha=0.3, label='Epsilon Tube')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Support Vector Regression with Epsilon Tube')
plt.legend()
plt.grid(True)

# Subplot 2: Training Loss Curve
plt.subplot(1, 2, 2)
#plt.plot(range(len(training_loss)), training_loss, color='green', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)

# Ensure the "plots" directory exists
os.makedirs("plots", exist_ok=True)

timestamp = time.strftime("%Y%m%d-%H%M%S")
plt.savefig(f"plots/svr_cplex_{timestamp}.png")
print("Plot saved")

plt.show()

# -------------------------
# Model Evaluation
# -------------------------
customRegressionReport(y_test_denorm, y_pred_test_denorm, name="Test")










"""
# Compute kernel matrix K
K = linear_kernel(X_train, X_train)

# SVR hyperparameters
C = 1.0  # Regularization
epsilon = 0.1  # Epsilon-insensitive margin
n = len(y_train)  # Number of training samples

# Create CPLEX model
mdl = Model(name="SVR_Dual")

# Dual variables α and α*
alpha = [mdl.continuous_var(lb=0, ub=C, name=f'alpha_{i}') for i in range(n)]
alpha_star = [mdl.continuous_var(lb=0, ub=C, name=f'alpha_star_{i}') for i in range(n)]

# Objective function (maximize dual formulation)
objective = mdl.sum(y_train[i] * (alpha[i] - alpha_star[i]) for i in range(n)) - \
            0.5 * mdl.sum((alpha[i] - alpha_star[i]) * (alpha[j] - alpha_star[j]) * K[i, j]
                          for i in range(n) for j in range(n))

mdl.maximize(objective)

# Constraint: sum of (alpha - alpha*) must be 0
mdl.add_constraint(mdl.sum(alpha[i] - alpha_star[i] for i in range(n)) == 0)

# Solve with CPLEX
solution = mdl.solve()

# Extract optimal α and α*
if solution:
    alpha_values = np.array([alpha[i].solution_value for i in range(n)])
    alpha_star_values = np.array([alpha_star[i].solution_value for i in range(n)])
    print("Solution found!")
else:
    print("No solution found.")
    exit()

# Compute weight vector w
w = np.sum((alpha_values - alpha_star_values)[:, np.newaxis] * X_train, axis=0)

# Compute bias b using support vectors
support_vector_indices = np.where((alpha_values > 0) & (alpha_values < C) |
                                  (alpha_star_values > 0) & (alpha_star_values < C))[0]

b_values = []
for i in support_vector_indices:
    b_i = y_train[i] - np.sum((alpha_values - alpha_star_values) * K[i, :])
    b_values.append(b_i)
b = np.mean(b_values)  # Take the mean over all support vectors

print(f"Computed w: {w}")
print(f"Computed b: {b}")
"""