import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import time

from SVM.Svr import SupportVectorRegression
from SVM.utility.Enum import KernelType
from SVM.utility.Search import random_search_svr
from sklearn.metrics import r2_score
from SVM.utility.preprocess import preprocessData, customRegressionReport, denormalize_price

# Use TkAgg backend for interactive plots
matplotlib.use('TkAgg')

# ------------------------- DATA LOADING -------------------------
print("Loading dataset...")
dataset = "dataset_diamonds/diamonds_cleaned.csv"
data = pd.read_csv(dataset, sep=',', header=0)
data_sampled = data.sample(n=3000, random_state=42).reset_index(drop=True)

# ------------------------- PREPROCESSING -------------------------
X_train, y_train, X_val, y_val, X_test, y_test, y_mean, y_std = preprocessData(data_sampled)
y_train = y_train.flatten()
y_val = y_val.flatten()
y_test = y_test.flatten()

# ------------------------- HYPERPARAMETER SEARCH -------------------------
param_grid_random = {
    "kernel_type": [KernelType.RBF],
    "C": [10, 30, 50],
    "epsilon": [10, 20, 30, 40, 50],
    "sigma": [0.01, 0.03, 0.05, 0.075, 0.1],
    "degree": [3],
    "coef": [0.0],
    "learning_rate": [0.001, 0.003, 0.005],
    "momentum": [0.7, 0.8, 0.9]
}
best_params, best_score = random_search_svr(X_train, y_train, X_val, y_val, param_grid_random, n_iter=10)

print(f"Best params: {best_params} with {best_score}")

# ------------------------- FINAL MODEL TRAINING -------------------------
svr_final = SupportVectorRegression(
    C=best_params["C"],
    epsilon=best_params["epsilon"],
    kernel_type=best_params["kernel_type"],
    sigma=best_params["sigma"],
    degree=best_params["degree"],
    coef=best_params["coef"],
    learning_rate=best_params["learning_rate"],
    momentum=best_params["momentum"]
)
svr_final.fit(X_train, y_train)

# ------------------------- VALIDATION PREDICTION -------------------------
print("\n---------------- VALIDATION PHASE ----------------")
Y_pred_val = svr_final.predict(X_val)
y_val_denorm = denormalize_price(y_val, y_mean, y_std)
Y_pred_val_denorm = denormalize_price(Y_pred_val, y_mean, y_std)
customRegressionReport(y_val_denorm, Y_pred_val_denorm, name="Test")

# ------------------------- FINAL TRAINING ON TRAIN+VAL -------------------------
print("\n---------------- TRAINING ON TRAIN + VALIDATION ----------------")
X_train_final = np.vstack((X_train, X_val))
y_train_final = np.hstack((y_train, y_val))
svr_final.fit(X_train_final, y_train_final)

# ------------------------- TEST PREDICTION -------------------------
print("\n---------------- TEST PHASE ----------------")
Y_pred_test = svr_final.predict(X_test)
y_test_denorm = denormalize_price(y_test, y_mean, y_std)
Y_pred_test_denorm = denormalize_price(Y_pred_test, y_mean, y_std)

# ------------------------- EPSILON TUBE PLOTS -------------------------
feature_idx = 0
feature_name = data_sampled.columns[feature_idx]
x_range = np.linspace(X_val[:, feature_idx].min(), X_val[:, feature_idx].max(), 300)
X_grid = np.tile(np.mean(X_val, axis=0), (300, 1))
X_grid[:, feature_idx] = x_range
y_pred_grid = svr_final.predict(X_grid)
y_pred_grid_denorm = denormalize_price(y_pred_grid, y_mean, y_std)
epsilon_denorm = float(best_params["epsilon"] * (y_std.iloc[0] if isinstance(y_std, pd.Series) else y_std))

os.makedirs("plots", exist_ok=True)

# Validation set epsilon tube plot
plt.figure(figsize=(10, 6))
plt.scatter(X_val[:, feature_idx], y_val_denorm, alpha=0.4, color="gray", label="True Data (Validation)")
plt.plot(x_range, y_pred_grid_denorm, color='blue', linewidth=2, label="SVR Prediction")
plt.plot(x_range, y_pred_grid_denorm + epsilon_denorm, 'r--', label="+ε Tube")
plt.plot(x_range, y_pred_grid_denorm - epsilon_denorm, 'r--', label="-ε Tube")
plt.fill_between(x_range,
                 y_pred_grid_denorm - epsilon_denorm,
                 y_pred_grid_denorm + epsilon_denorm,
                 color='red', alpha=0.1)
plt.xlabel(f"{feature_name} (normalized)")
plt.ylabel("Predicted Price")
plt.title(f"SVR + Epsilon Tube on Validation Set [{feature_name}]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/svr_epsilon_tube_validation.png")
plt.show()

# Test set epsilon tube plot
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, feature_idx], y_test_denorm, alpha=0.4, color="orange", label="True Data (Test)")
plt.plot(x_range, y_pred_grid_denorm, color='blue', linewidth=2, label="SVR Prediction")
plt.plot(x_range, y_pred_grid_denorm + epsilon_denorm, 'r--', label="+ε Tube")
plt.plot(x_range, y_pred_grid_denorm - epsilon_denorm, 'r--', label="-ε Tube")
plt.fill_between(x_range,
                 y_pred_grid_denorm - epsilon_denorm,
                 y_pred_grid_denorm + epsilon_denorm,
                 color='red', alpha=0.1)
plt.xlabel(f"{feature_name} (normalized)")
plt.ylabel("Predicted Price")
plt.title(f"SVR + Epsilon Tube on Test Set [{feature_name}]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/svr_epsilon_tube_test.png")
plt.show()

# ------------------------- METRICS -------------------------
print("\n---------------- VALIDATION METRICS ----------------")
customRegressionReport(y_val_denorm, Y_pred_val_denorm, name="Validation")

print("\n---------------- TEST METRICS ----------------")
customRegressionReport(y_test_denorm, Y_pred_test_denorm, name="Test")

print("\nAll plots and reports saved in the 'plots' directory.")
