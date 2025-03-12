import numpy as np
import pandas as pd
from docplex.mp.model import Model
import matplotlib.pyplot as plt
import time
import os
from SVM.utility.preprocess import preprocessData, denormalize_zscore  # Your preprocessing function

# Load dataset
dataset_red = "dataset_wine/winequality-red.csv"
data_red = pd.read_csv(dataset_red, sep=';', header=0)

# Preprocess the data (same as your original code)
train_set, X_train, y_train, X_val, y_val, X_test, y_test = preprocessData(data_red)

# Merge training and validation sets
X_train = np.vstack((X_train, X_val))
y_train = np.concatenate((y_train.flatten(), y_val.flatten()))

y_train = y_train.flatten()
y_test = y_test.flatten()

# Define linear kernel
def linear_kernel(X1, X2):
    return np.dot(X1, X2.T)

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

# -------------------------
# Make Predictions on Test Set
# -------------------------
y_pred_test = np.dot(X_test, w) + b  # Compute predictions

# Denormalize predictions
X_test_denorm = denormalize_zscore(X_test, train_set)
y_pred_test_denorm = denormalize_zscore(y_pred_test, train_set)
y_test_denorm = denormalize_zscore(y_test, train_set)

# Sort validation data for a smooth plot
sorted_idx = np.argsort(X_val[:, 0])
X_sorted = X_test_denorm[sorted_idx]
Y_pred_sorted = y_pred_test_denorm[sorted_idx]
y_sorted = y_test_denorm[sorted_idx]

# Compute upper and lower bounds of the epsilon tube
upper_bound = y_pred_test_denorm + epsilon
lower_bound = y_pred_test_denorm - epsilon

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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test_denorm, y_pred_test_denorm)
mae = mean_absolute_error(y_test_denorm, y_pred_test_denorm)
r2 = r2_score(y_test_denorm, y_pred_test_denorm)

print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R² Score: {r2:.4f}")
