import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

from SVM.Svr import SupportVectorRegression
from SVM.utility.Enum import KernelType
from SVM.utility.Search import grid_search_svr, random_search_svr
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
X_train, y_train, X_val, y_val, X_test, y_test, y_mean, y_std, mean, std = preprocessData(data_sampled)
y_train = y_train.flatten()
y_val = y_val.flatten()
y_test = y_test.flatten()

print("y_mean:", y_mean)
print("y_std:", y_std)
print("Shape y_mean:", y_mean.shape)
print("Shape y_train:", y_train.shape)
print("First 5 normalized y_train:", y_train[:5])

# ------------------------- HYPERPARAMETER SEARCH -------------------------
param_grid_random = {
    "kernel_type": [
        KernelType.RBF
    ],

    # C controls regularization (applies to all kernels)
    'C': [0.3, 0.5, 0.7],

    # Epsilon-insensitive zone width
    'epsilon': [0.3, 0.35, 0.4, 0.45],

    # Sigma is used only for RBF
    'sigma': [0.3, 0.4, 0.5, 0.6],

    # Polynomial kernel parameters (ignored elsewhere)
    "degree": [2],
    "coef": [0.0],

    # Optimizer parameters
    'learning_rate': [0.03, 0.04, 0.05, 0.06],
    "momentum": [0.8, 0.9, 0.95]
}
# Best params: {'kernel_type': <KernelType.RBF: 'radial basis function'>, 'C': 0.5, 'epsilon': 0.4, 'sigma': 0.5,
# 'degree': 2, 'coef': 0.0, 'learning_rate': 0.1, 'momentum': 0.95} with 6.307816496453521

# best_params, best_score = grid_search_svr(X_train, y_train, X_val, y_val, param_grid_random)
best_params, best_score = random_search_svr(X_train, y_train, X_val, y_val, param_grid_random, n_iter=100)
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

# ------------------------- CONVERGENCE PLOTS -------------------------
print("\n---------------- PLOTTING CONVERGENCE ----------------")
# Only now we plot convergence
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(svr_final.training_loss["beta_norms"], label="|β - β_prev|")
plt.xlabel("Iteration")
plt.ylabel("Beta Update Norm")
plt.title("Convergence Speed")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(svr_final.training_loss["grad_norms"], label="||∇Q_mu||")
plt.xlabel("Iteration")
plt.ylabel("Gradient Norm")
plt.title("Gradient Magnitude")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(svr_final.training_loss["Q_mu"], label="Q_mu")
plt.xlabel("Iteration")
plt.ylabel("Dual Objective (Q_mu)")
plt.title("Final Loss Convergence")
plt.legend()

plt.show()

# ------------------------- VALIDATION PREDICTION -------------------------
print("\n---------------- VALIDATION PHASE ----------------")
Y_pred_val = svr_final.predict(X_val)
y_val_denorm = denormalize_price(y_val, y_mean, y_std)
Y_pred_val_denorm = denormalize_price(Y_pred_val, y_mean, y_std)

X_val_denorm = denormalize_price(X_val, mean, std)
X_test_denorm = denormalize_price(X_test, mean, std)

print("First 5 denormalized predictions:", Y_pred_val_denorm[:5])
print("Denorm mean:", np.mean(Y_pred_val_denorm))
print("Denorm std:", np.std(Y_pred_val_denorm))

# ------------------------- FINAL TRAINING ON TRAIN+VAL -------------------------
print("\n---------------- TRAINING ON TRAIN + VALIDATION ----------------")
X_train_final = np.vstack((X_train, X_val))
y_train_final = np.hstack((y_train, y_val))
svr_final.fit(X_train_final, y_train_final)
print("max(β) VAL:", np.max(np.abs(svr_final.beta)))
print("bias b VAL:", svr_final.b)

# ------------------------- TEST PREDICTION -------------------------
print("\n---------------- TEST PHASE ----------------")
Y_pred_test = svr_final.predict(X_test)
y_test_denorm = denormalize_price(y_test, y_mean, y_std)
Y_pred_test_denorm = denormalize_price(Y_pred_test, y_mean, y_std)
X_train_final_denorm = denormalize_price(X_train_final, mean, std)

# ------------------------- EPSILON TUBE PLOTS (REALISTIC CURVE) -------------------------
print("\n---------------- PLOTTING SVR CURVE ON SIGNIFICANT FEATURE ----------------")
timestamp = time.strftime("%Y%m%d-%H%M%S")

# Select meaningful feature to visualize (e.g., "carat")
feature_name = "carat"
feature_idx = list(data_sampled.columns).index(feature_name)

# Sort validation data by the selected feature
sorted_idx_val = np.argsort(X_val_denorm[:, feature_idx])
X_val_sorted = X_val_denorm[sorted_idx_val]
y_val_sorted = y_val_denorm[sorted_idx_val]
Y_pred_val_sorted = Y_pred_val_denorm[sorted_idx_val]

# Sort test data by the selected feature
sorted_idx_test = np.argsort(X_test_denorm[:, feature_idx])
X_test_sorted = X_test_denorm[sorted_idx_test]
y_test_sorted = y_test_denorm[sorted_idx_test]
Y_pred_test_sorted = svr_final.predict(X_test_sorted)
Y_pred_test_denorm_sorted = denormalize_price(Y_pred_test_sorted, y_mean, y_std)

# Get denormalized epsilon from final model
epsilon_denorm = denormalize_price(np.array([svr_final.epsilon]), y_mean, y_std)[0]

os.makedirs("plots/validation", exist_ok=True)
os.makedirs("plots/test", exist_ok=True)

# Plot: Validation set
plt.figure(figsize=(10, 6))
plt.scatter(X_val_sorted[:, feature_idx], y_val_sorted, alpha=0.4, color="gray", label="True Data (Validation)")
plt.plot(X_val_sorted[:, feature_idx], Y_pred_val_sorted, color='blue', linewidth=2, label="SVR Prediction")
plt.plot(X_val_sorted[:, feature_idx], Y_pred_val_sorted + epsilon_denorm, 'r--', label="+\u03b5 Tube")
plt.plot(X_val_sorted[:, feature_idx], Y_pred_val_sorted - epsilon_denorm, 'r--', label="-\u03b5 Tube")
plt.fill_between(X_val_sorted[:, feature_idx],
                 Y_pred_val_sorted - epsilon_denorm,
                 Y_pred_val_sorted + epsilon_denorm,
                 color='red', alpha=0.1)
plt.xlabel(f"{feature_name} (denormalized)")
plt.ylabel("Predicted Price")
plt.title(f"SVR + Epsilon Tube on Validation Set [{feature_name}]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"plots/validation/svr_epsilon_tube_validation{timestamp}.png")
plt.show()

# Plot: Test set
plt.figure(figsize=(10, 6))
plt.scatter(X_test_sorted[:, feature_idx], y_test_sorted, alpha=0.4, color="orange", label="True Data (Test)")
plt.plot(X_test_sorted[:, feature_idx], Y_pred_test_denorm_sorted, color='blue', linewidth=2, label="SVR Prediction")
plt.plot(X_test_sorted[:, feature_idx], Y_pred_test_denorm_sorted + epsilon_denorm, 'r--', label="+\u03b5 Tube")
plt.plot(X_test_sorted[:, feature_idx], Y_pred_test_denorm_sorted - epsilon_denorm, 'r--', label="-\u03b5 Tube")
plt.fill_between(X_test_sorted[:, feature_idx],
                 Y_pred_test_denorm_sorted - epsilon_denorm,
                 Y_pred_test_denorm_sorted + epsilon_denorm,
                 color='red', alpha=0.1)
plt.xlabel(f"{feature_name} (denormalized)")
plt.ylabel("Predicted Price")
plt.title(f"SVR + Epsilon Tube on Test Set [{feature_name}]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"plots/test/svr_epsilon_tube_test{timestamp}.png")
plt.show()

# ------------------------- METRICS -------------------------
print("\n---------------- VALIDATION METRICS ----------------")
customRegressionReport(y_val, Y_pred_val, name="Validation")

print("\n---------------- TEST METRICS ----------------")
customRegressionReport(y_test, Y_pred_test, name="Test")

print("\nAll plots and reports saved in the 'plots' directory.")
