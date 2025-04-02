import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from SVM.utility.preprocess import preprocessData, denormalize_price, customRegressionReport  # Your preprocessing function

# Load dataset
dataset = "dataset_diamonds/diamonds_cleaned.csv"
data = pd.read_csv(dataset, sep=',', header=0)

# Sample 300 instances
data_sampled = data.sample(n=1000, random_state=42).reset_index(drop=True)

# Preprocess
X_train, y_train, X_val, y_val, X_test, y_test, y_mean, y_std = preprocessData(data_sampled)
X_train = np.vstack((X_train, X_val))
y_train = np.concatenate((y_train.flatten(), y_val.flatten()))

n, d = X_train.shape  # Number of samples & features

C = 3.0  # Regularization
epsilon = 1.0  # Epsilon-insensitive margin

# **Define CVXPY variables**
w = cp.Variable(d)      # Weight vector
b = cp.Variable()       # Bias term
xi = cp.Variable(n)     # Slack variables for +epsilon
xi_star = cp.Variable(n) # Slack variables for -epsilon

# **SVR Objective Function (Minimize Regularized Loss)**
objective = cp.Minimize(
    0.5 * cp.norm(w, 2) ** 2 + C * cp.sum(xi + xi_star)
)

# **Constraints**
constraints = [
    y_train - (X_train @ w + b) <= epsilon + xi,  # Upper epsilon margin
    (X_train @ w + b) - y_train <= epsilon + xi_star,  # Lower epsilon margin
    xi >= 0,  # Non-negative slack variables
    xi_star >= 0
]

# **Solve the optimization problem**
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.SCS)

# Extract solution
w_value = w.value
b_value = b.value

print(f"Optimized w: {w_value}")
print(f"Optimized b: {b_value}")

# Make Predictions
y_pred_test = X_test @ w_value + b_value

# Denormalize predictions
y_pred_test_denorm = denormalize_price(y_pred_test, y_mean, y_std)
y_test_denorm = denormalize_price(y_test, y_mean, y_std)

# Sort test data for smooth plotting
sorted_idx = np.argsort(X_test[:, 0])
X_sorted = X_test[sorted_idx]
Y_pred_sorted = y_pred_test_denorm[sorted_idx]
y_sorted = y_test_denorm[sorted_idx]

# Compute upper and lower bounds of the epsilon tube
epsilon_denorm = epsilon * float(y_std.iloc[0]) if isinstance(y_std, pd.Series) else epsilon * y_std
upper_bound = y_pred_test_denorm + epsilon_denorm
lower_bound = y_pred_test_denorm - epsilon_denorm

# **Plot Results**
plt.figure(figsize=(12, 5))

# Scatter plot for test data
plt.scatter(X_sorted[:, 0], y_sorted, color='darkorange', label='Test Data')

# Plot the SVR model prediction
plt.plot(X_sorted[:, 0], Y_pred_sorted, color='navy', lw=2, label='SVR Model')

# Plot the epsilon tube (shading between upper and lower bounds)
plt.fill_between(X_sorted[:, 0], lower_bound, upper_bound, color='lightblue', alpha=0.3, label='Epsilon Tube')

# Labeling the axes and adding title
plt.xlabel('Feature (normalized)')
plt.ylabel('Predicted Price')
plt.title('Support Vector Regression with CVXPY and Epsilon Tube')
plt.legend()
plt.grid(True)

# Save the plot
os.makedirs("plots", exist_ok=True)
timestamp = time.strftime("%Y%m%d-%H%M%S")
plt.savefig(f"plots/svr_cvxpy_{timestamp}.png")
plt.show()

customRegressionReport(y_test_denorm, y_pred_test_denorm, name="Test")
