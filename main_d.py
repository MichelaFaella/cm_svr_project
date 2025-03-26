import numpy as np
import matplotlib
import pandas as pd
from sklearn.metrics import r2_score
from SVM.utility.preprocess import preprocessData_d, denormalize_zscore, customRegressionReport, denormalize_price

# Use a backend suitable for showing plots
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import time
import os

from SVM.Svr import SupportVectorRegression
from SVM.utility.Enum import KernelType
from SVM.utility.Search import random_search_svr

# ------------------------- DATA LOADING -------------------------
print("Loading dataset...")
dataset = "dataset_diamonds/diamonds_cleaned.csv"
data = pd.read_csv(dataset, sep=',', header=0)

# Randomly select 3000 samples
data_sampled = data.sample(n=3000, random_state=42)  # Set random_state for reproducibility

# Reset index
data_sampled = data_sampled.reset_index(drop=True)

# Print first few rows
print(data_sampled.head())

# Preprocessing (Z-score normalization + train/val/test split)
X_train, y_train, X_val, y_val, X_test, y_test, y_mean, y_std = preprocessData_d(data_sampled)

print(f"Min/Max of normalized train_X: {X_train.min()}, {X_train.max()}")
print(f"Min/Max of normalized train_Y: {y_train.min()}, {y_train.max()}")

# Flatten targets for compatibility
y_train = y_train.flatten()
y_val = y_val.flatten()
y_test = y_test.flatten()

# ------------------------- RANDOM HYPERPARAMETER SEARCH -------------------------
print("Starting random hyperparameter search for SVR...")
param_grid_random = {
    "kernel_type": [KernelType.RBF, KernelType.LINEAR],
    "C": [1, 5],  # Extended range: low to moderate values for flexibility
    "epsilon": [0.1, 0.2, 0.3],  # Wide tube to tolerate noise
    "sigma": [1.0, 1.5],  # Smooth kernels with various widths
    "degree": [3],  # Fixed for polynomial (not used here)
    "coef": [0.0],  # Irrelevant for RBF, kept for completeness
    "learning_rate": [0.0005],  # Small learning rates for stable convergence
    "momentum": [0.9]  # Different momentum levels to help optimization
}

best_params, best_score = random_search_svr(X_train, y_train, X_val, y_val, param_grid_random, n_iter=10)
print(f"\nBest hyperparameters found: {best_params}")
print(f"Validation MSE of best config: {best_score}")

# ------------------------- TRAIN FINAL MODEL -------------------------
print("\nTraining final SVR model with best parameters...")
svr_final = SupportVectorRegression(
    C=best_params["C"],
    epsilon=best_params["epsilon"],
    kernel_type=best_params["kernel_type"],
    sigma=best_params.get("sigma", 1.0),
    degree=best_params.get("degree", 3),
    coef=best_params.get("coef", 1),
    learning_rate=best_params["learning_rate"],
    momentum=best_params["momentum"]
)

training_loss = svr_final.fit(X_train, y_train)

# ------------------------- VALIDATION PREDICTION & VISUALIZATION -------------------------
print("\nEvaluating on validation set...")
Y_pred_val = svr_final.predict(X_val)

print(f"Raw Y_pred_val (normalized): {Y_pred_val[:10]}")  # Print first 10 predictions
print(f"Min/Max of Raw Y_pred_val (normalized): {Y_pred_val.min()}, {Y_pred_val.max()}")

print(f"y_mean (original): {y_mean}, y_std (original): {y_std}")

# Denormalize predictions and ground truth
y_val_denorm = denormalize_price(y_val, y_mean, y_std)
Y_pred_val_denorm = denormalize_price(Y_pred_val, y_mean, y_std)

print(f"Min/Max of y_val_denorm: {y_val_denorm.min()}, {y_val_denorm.max()}")
print(f"Min/Max of Y_pred_val_denorm: {Y_pred_val_denorm.min()}, {Y_pred_val_denorm.max()}")

r2_val = r2_score(y_val_denorm, Y_pred_val_denorm)
print(f"Validation R² score: {r2_val:.4f}")

# Create output directory
os.makedirs("plots", exist_ok=True)

# Scatter plot: True vs Predicted (Validation)
plt.figure(figsize=(6, 6))
plt.scatter(y_val_denorm, Y_pred_val_denorm, alpha=0.5, color="blue")
plt.scatter(y_val_denorm, y_val_denorm, alpha=0.5, color="green", label="True Values")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title(f"Validation Set: True vs Predicted (R² = {r2_val:.2f})")
plt.grid(True)
plt.savefig(f"plots/svr_validation_scatter_{time.strftime('%Y%m%d-%H%M%S')}.png")
plt.show()

# Line plot on validation set (with epsilon tube)
sorted_idx = np.argsort(X_val[:, 0])
plt.figure(figsize=(10, 5))
plt.scatter(X_val[sorted_idx, 0], y_val_denorm[sorted_idx], color='orange', label='True Price')
plt.plot(X_val[sorted_idx, 0], Y_pred_val_denorm[sorted_idx], color='grey', lw=2, label='SVR Prediction')
plt.fill_between(X_val[sorted_idx, 0],
                 Y_pred_val_denorm[sorted_idx] - best_params["epsilon"],
                 Y_pred_val_denorm[sorted_idx] + best_params["epsilon"],
                 color='lightblue', alpha=0.3, label='Epsilon Tube')
plt.xlabel("Feature (sorted by first feature)")
plt.ylabel("Diamond Price (denormalized)")
plt.legend()
plt.grid(True)
plt.savefig(f"plots/svr_validation_line_{time.strftime('%Y%m%d-%H%M%S')}.png")
plt.show()

# Detailed regression report on validation set
customRegressionReport(y_val_denorm, Y_pred_val_denorm, name="Validation")

# ------------------------- TEST PHASE -------------------------
print("\nTraining final model on full training set (train + val) and evaluating on test set...")
X_train_final = np.vstack((X_train, X_val))
y_train_final = np.hstack((y_train, y_val))
training_loss = svr_final.fit(X_train_final, y_train_final)

Y_pred_test = svr_final.predict(X_test)
y_test_denorm = denormalize_price(y_test, y_mean, y_std)
Y_pred_test_denorm = denormalize_price(Y_pred_test, y_mean, y_std)

r2_test = r2_score(y_test_denorm, Y_pred_test_denorm)
print(f"Test R² score: {r2_test:.4f}")

# Scatter plot: Test set
plt.figure(figsize=(6, 6))
plt.scatter(y_test_denorm, Y_pred_test_denorm, alpha=0.5, color="purple")
plt.scatter(y_val_denorm, y_val_denorm, alpha=0.5, color="green", label="True Values")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title(f"Test Set: True vs Predicted (R² = {r2_test:.2f})")
plt.grid(True)
plt.savefig(f"plots/svr_test_scatter_{time.strftime('%Y%m%d-%H%M%S')}.png")
plt.show()

# Line plot on test set with epsilon tube
sorted_idx = np.argsort(X_test[:, 0])
plt.figure(figsize=(10, 5))
plt.scatter(X_test[sorted_idx, 0], y_test_denorm[sorted_idx], color='orange', label='True Price')
plt.plot(X_test[sorted_idx, 0], Y_pred_test_denorm[sorted_idx], color='navy', lw=2, label='SVR Prediction')
plt.fill_between(X_test[sorted_idx, 0],
                 Y_pred_test_denorm[sorted_idx] - best_params["epsilon"],
                 Y_pred_test_denorm[sorted_idx] + best_params["epsilon"],
                 color='lightblue', alpha=0.3, label='Epsilon Tube')
plt.xlabel("Feature (sorted by first feature)")
plt.ylabel("Diamonds price (denormalized)")
plt.legend()
plt.grid(True)
plt.savefig(f"plots/svr_test_line_{time.strftime('%Y%m%d-%H%M%S')}.png")
plt.show()

# Detailed regression report on test set
customRegressionReport(y_test_denorm, Y_pred_test_denorm, name="Test")

print("\nAll plots and reports saved in the 'plots' directory.")
