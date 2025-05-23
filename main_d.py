import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

from SVM.Svr import SupportVectorRegression
from SVM.utility.Enum import KernelType
from SVM.utility.Search import grid_search_svr
from SVM.utility.utility import preprocessData, customRegressionReport, denormalize_price, denormalize, \
    plot_convergence_curves

# Use TkAgg backend for interactive plots
matplotlib.use('TkAgg')

# ------------------------- DATA LOADING -------------------------
print("Loading dataset...")
dataset = "dataset_diamonds/diamonds_cleaned.csv"
data = pd.read_csv(dataset, sep=',', header=0)
data_sampled = data.sample(n=2000, random_state=64).reset_index(drop=True)

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
"""
param_grid_random = {
    # proviamo sia RBF, sia POLY e LINEAR
    'kernel_type': [KernelType.RBF, KernelType.POLYNOMIAL, KernelType.LINEAR],

    # trade‐off complessità vs. errore
    'C':       [0.1, 1.0, 10.0, 100.0],

    # larghezza della zona ε‐insensitive
    'epsilon': [0.01, 0.1, 1.0],

    # per RBF: scala del kernel
    'sigma':   [0.1, 0.5, 1.0, 2.0],

    # per POLY: grado e coefficiente
    'degree': [2, 3, 4],
    'coef':   [0.0, 1.0, 10.0],

    # controllo della convergence
    'max_iter': [500, 1000],
    'tol':      [1e-3, 1e-6],
}"""

param_grid_random = {
    # proviamo sia RBF, sia POLY e LINEAR
    'kernel_type': [KernelType.RBF],

    # trade‐off complessità vs. errore
    'C': [1],

    # larghezza della zona ε‐insensitive
    'epsilon': [1.8],

    # per RBF: scala del kernel
    'sigma': [0.5],

    # per POLY: grado e coefficiente
    'degree': [2],
    'coef': [0.0],
}

best_params, best_score = grid_search_svr(X_train, y_train, X_val, y_val, param_grid_random)
print(f"Best params: {best_params} with {best_score}")

# ------------------------- FINAL MODEL TRAINING -------------------------
svr_final = SupportVectorRegression(
    C=best_params["C"],
    epsilon=best_params["epsilon"],
    kernel_type=best_params["kernel_type"],
    sigma=best_params["sigma"],
    degree=best_params["degree"],
    coef=best_params["coef"],
)
svr_final.fit(X_train, y_train)

# ------------------------- CONVERGENCE PLOTS -------------------------
print("\n---------------- PLOTTING CONVERGENCE ----------------")
plot_convergence_curves(svr_final.training_history, title_prefix=f"SVR_Diamonds-{best_params["kernel_type"]}")

# ------------------------- VALIDATION PREDICTION -------------------------
print("\n---------------- VALIDATION PHASE ----------------")
Y_pred_val = svr_final.predict(X_val)
y_val_denorm = denormalize_price(y_val, y_mean, y_std)
Y_pred_val_denorm = denormalize_price(Y_pred_val, y_mean, y_std)

X_val_denorm = denormalize(X_val, mean, std)
X_test_denorm = denormalize(X_test, mean, std)

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
X_train_final_denorm = denormalize(X_train_final, mean, std)

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

# Sort test data by the selected feature (for plotting)
sorted_idx_test = np.argsort(X_test_denorm[:, feature_idx])
X_test_sorted_denorm = X_test_denorm[sorted_idx_test]  # for x-axis
X_test_sorted_norm = X_test[sorted_idx_test]  # to predict
y_test_sorted = y_test_denorm[sorted_idx_test]

# Predict on normalized features
Y_pred_test_sorted = svr_final.predict(X_test_sorted_norm)
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

# Plot: Test set (CORRETTO)
plt.figure(figsize=(10, 6))
plt.scatter(X_test_sorted_denorm[:, feature_idx], y_test_sorted, alpha=0.4, color="orange", label="True Data (Test)")
plt.plot(X_test_sorted_denorm[:, feature_idx], Y_pred_test_denorm_sorted, color='blue', linewidth=2,
         label="SVR Prediction")
plt.plot(X_test_sorted_denorm[:, feature_idx], Y_pred_test_denorm_sorted + epsilon_denorm, 'r--', label="+\u03b5 Tube")
plt.plot(X_test_sorted_denorm[:, feature_idx], Y_pred_test_denorm_sorted - epsilon_denorm, 'r--', label="-\u03b5 Tube")
plt.fill_between(X_test_sorted_denorm[:, feature_idx],
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

print("max β:", np.max(svr_final.beta))
print("min β:", np.min(svr_final.beta))
print("support vector attivi:", np.sum((np.abs(svr_final.beta) > 1e-3) & (np.abs(svr_final.beta) < svr_final.C)))
print("bias b:", svr_final.b)

print("\nAll plots and reports saved in the 'plots' directory.")
