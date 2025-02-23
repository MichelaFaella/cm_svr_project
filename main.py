import numpy as np
import matplotlib.pyplot as plt
import time
import os

from SVM.Svr import SupportVectorRegression
from SVM.utility.Enum import KernelType
from sklearn.model_selection import train_test_split
from SVM.utility.Search import random_search_svr, grid_search_svr
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
X = np.linspace(-3, 3, 60).reshape(-1, 1)  # 60 samples
Y = np.sin(X) + 0.2 * X**2 + 0.1 * np.random.randn(*X.shape)  # More complex function

# Standardize data
scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_x.fit_transform(X)
Y_scaled = scaler_y.fit_transform(Y.reshape(-1, 1)).flatten()

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# Create test points for predictions
X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
X_test_scaled = scaler_x.transform(X_test)

# -------------------------
# Step 1: Randomized Search to Estimate Hyperparameter Ranges
# -------------------------
print("\nPerforming Randomized Search to estimate hyperparameter ranges...")

param_grid_random = {
    "kernel_type": [KernelType.RBF, KernelType.POLYNOMIAL],
    "C": np.logspace(0, 3, 10).tolist(),  # Expanded range
    "epsilon": np.linspace(0.001, 0.5, 8).tolist(),
    "sigma": np.linspace(0.1, 3.0, 10).tolist(),  # Broader range for RBF
    "degree": [2, 3, 4, 5],  # Avoiding overly complex polynomials
    "coef": np.linspace(0, 2, 4).tolist()  # Less granularity for coef
}

best_random_params, best_random_score = random_search_svr(X_train, y_train, X_val, y_val, param_grid_random, n_iter=12)
print(f"\n Best result from Random Search\n")
print(f"Parameters: {best_random_params}, MSE: {best_random_score}")

# -------------------------
# Step 2: Grid Search for Fine-Tuning
# -------------------------
print("\n Performing Grid Search to fine-tune best hyperparameters...")

param_grid_grid = {
    "kernel_type": [best_random_params["kernel_type"]],
    "C": np.linspace(best_random_params["C"] / 3, best_random_params["C"] / 1.5, 5).tolist(),
    "epsilon": np.linspace(best_random_params["epsilon"] * 1.5, best_random_params["epsilon"] * 3, 5).tolist(),
    "sigma":  np.linspace(best_random_params["sigma"], best_random_params["sigma"] * 2, 5).tolist(),
    "degree": [best_random_params["degree"]] if best_random_params["kernel_type"] == KernelType.POLYNOMIAL else [None],
    "coef": [best_random_params["coef"]] if best_random_params["kernel_type"] == KernelType.POLYNOMIAL else [None]
}

best_grid_params, best_grid_score = grid_search_svr(X_train, y_train, X_val, y_val, param_grid_grid)
print(f"\n Best result from Grid Search\n")
print(f"Parameters: {best_grid_params}, MSE: {best_grid_score}")

# -------------------------
# Step 3: Train Final Model
# -------------------------
print("\n Training Final SVR Model...")

if best_grid_params["kernel_type"] == KernelType.RBF:
    svr_final = SupportVectorRegression(
        C=best_grid_params["C"],
        epsilon=best_grid_params["epsilon"],
        kernel_type=KernelType.RBF,
        sigma=best_grid_params["sigma"]
    )
elif best_grid_params["kernel_type"] == KernelType.POLYNOMIAL:
    svr_final = SupportVectorRegression(
        C=best_grid_params["C"],
        epsilon=best_grid_params["epsilon"],
        kernel_type=KernelType.POLYNOMIAL,
        degree=best_grid_params["degree"],
        coef=best_grid_params["coef"]
    )

# Train the model
svr_final.fit(X_train, y_train)

# Check if alpha changes
print("Final alpha values:", svr_final.alpha[:5])  # Check first 5 values
print("Norm of alpha:", np.linalg.norm(svr_final.alpha))

# Predict and inverse transform the result
Y_pred_final_scaled = svr_final.predict(X_test_scaled)
Y_pred_final = scaler_y.inverse_transform(Y_pred_final_scaled.reshape(-1, 1)).flatten()

# ---------------------------------
# Plot Results: Optimized SVR Model
# ---------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='red', label='Training Data')
plt.plot(X_test, Y_pred_final, color='blue', linestyle='dashed', linewidth=2, label='Optimized SVR Model')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Optimized Support Vector Regression')
plt.legend()
plt.grid(True)

# Ensure the "plots" directory exists
os.makedirs("plots", exist_ok=True)

timestamp = time.strftime("%Y%m%d-%H%M%S")
plt.savefig(f"plots/svr_output_{timestamp}.png")
print("Plot saved")
