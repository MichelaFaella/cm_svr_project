import numpy as np
import matplotlib
import pandas as pd
from SVM.utility.preprocess import preprocessData, denormalize_zscore, customRegressionReport

matplotlib.use('Agg')  # OPPURE 'Qt5Agg' 'TkAgg'
import matplotlib.pyplot as plt

import time
import os

from SVM.Svr import SupportVectorRegression
from SVM.utility.Enum import KernelType, LossFunctionType
from SVM.utility.Search import random_search_svr, grid_search_svr

# Download data
dataset_red = "dataset_wine/winequality-red.csv"
dataset_white = "dataset_wine/winequality-white.csv"

# Read the training dataset
data_red = pd.read_csv(dataset_red, sep=';', header=0)
data_white = pd.read_csv(dataset_white, sep=';', header=0)

train_set, X_train, y_train, X_val, y_val, X_test, y_test = preprocessData(data_red)

print("train_X: ", X_train.shape, "\ntrain_Y: ", y_train.shape)
print("validation_X: ", X_val.shape, "\nvalidation_Y: ", y_val.shape)
print("test_X: ", X_test.shape, "\ntest_Y: ", y_test.shape)

y_train = y_train.flatten()
y_val = y_val.flatten()
y_test = y_test.flatten()

# -------------------------
# Step 1: Randomized Search to Estimate Hyperparameter Ranges
# -------------------------
print("\nPerforming Randomized Search to estimate hyperparameter ranges...")

loss_type = LossFunctionType.EPSILON_INSENSITIVE

param_grid_random = {
    "kernel_type": [KernelType.RBF, KernelType.POLYNOMIAL, KernelType.LINEAR],  # Entrambi i kernel
    "C": np.logspace(-2, 3, 5).tolist(),  # Da 0.01 a 1000 per testare flessibilità
    "epsilon": np.linspace(0.01, 0.5, 5).tolist(),  # Evita valori troppo piccoli o grandi
    "sigma": np.linspace(0.1, 5, 2).tolist(),  # Ampio range per RBF
    "degree": [2, 3, 4],  # Evitiamo polinomi troppo complessi
    "coef": np.linspace(0, 2.0, 5).tolist(),  # Coefficiente di bias per il kernel polinomiale
    "learning_rate": np.logspace(-4, -1, 5).tolist()
}

best_random_params, best_random_score = random_search_svr(X_train, y_train, X_val, y_val,
                                                          param_grid_random, n_iter=10,
                                                          loss_type=loss_type)

print(f"\n Best result from Random Search\n")
print(f"Parameters: {best_random_params}, MSE: {best_random_score}")

# -------------------------
# Step 2: Grid Search for Fine-Tuning
# -------------------------
"""print("\n Performing Grid Search to fine-tune best hyperparameters...")

param_grid_fine = {
    "kernel_type": [best_random_params["kernel_type"]],  # Prendiamo il miglior kernel trovato
    "C": np.linspace(best_random_params["C"] * 0.9, best_random_params["C"] * 1.1, 5).tolist(),
    "epsilon": np.linspace(best_random_params["epsilon"] * 0.9, best_random_params["epsilon"] * 1.1, 5).tolist(),
    "sigma": np.linspace(best_random_params["sigma"] * 0.9, best_random_params["sigma"] * 1.1, 5).tolist() if
    best_random_params["kernel_type"] == KernelType.RBF else [None],  # Affiniamo sigma solo per RBF
    "degree": [best_random_params["degree"]] if best_random_params["kernel_type"] == KernelType.POLYNOMIAL else [None],
    "coef": np.linspace(best_random_params["coef"] * 0.9, best_random_params["coef"] * 1.1, 5).tolist() if
    best_random_params["kernel_type"] == KernelType.POLYNOMIAL else [None],  # Affiniamo coef solo per polinomiale
    "learning_rate": np.linspace(best_random_params["learning_rate"] * 0.9, best_random_params["learning_rate"] * 1.1,
                                 5).tolist()
}

best_grid_params, best_grid_score = grid_search_svr(
    X_train, y_train, X_val, y_val, param_grid_fine, loss_type=loss_type
)

print(f"\n Best result from Grid Search\n")
print(f"Parameters: {best_grid_params}, MSE: {best_grid_score}")"""

# -------------------------
# Step 3: Train Final Model
# -------------------------
print("\n Training Final SVR Model...")

if best_random_params["kernel_type"] == KernelType.RBF:
    svr_final = SupportVectorRegression(
        C=best_random_params["C"],
        epsilon=best_random_params["epsilon"],
        kernel_type=KernelType.RBF,
        sigma=best_random_params["sigma"],
        loss_function=loss_type,
        learning_rate=best_random_params["learning_rate"]
    )
elif best_random_params["kernel_type"] == KernelType.POLYNOMIAL:
    svr_final = SupportVectorRegression(
        C=best_random_params["C"],
        epsilon=best_random_params["epsilon"],
        kernel_type=KernelType.POLYNOMIAL,
        degree=best_random_params["degree"],
        coef=best_random_params["coef"],
        loss_function=loss_type,
        learning_rate=best_random_params["learning_rate"]
    )
elif best_random_params["kernel_type"] == KernelType.LINEAR:
    svr_final = SupportVectorRegression(
        C=best_random_params["C"],
        epsilon=best_random_params["epsilon"],
        kernel_type=KernelType.LINEAR,
        degree=best_random_params["degree"],
        coef=best_random_params["coef"],
        loss_function=loss_type,
        learning_rate=best_random_params["learning_rate"]
    )

# Train the model and track loss over iterations
training_loss = svr_final.fit(X_train, y_train)

# Check if alpha changes
print("Final alpha values:", svr_final.alpha[:5])  # Check first 5 values
print("Norm of alpha:", np.linalg.norm(svr_final.alpha))

# Predict and inverse transform the result
Y_pred_val = svr_final.predict(X_val)

# Sort validation data for a smooth plot
sorted_idx = np.argsort(X_val[:, 0])
X_val_sorted = X_val[sorted_idx]
Y_pred_val_sorted = Y_pred_val[sorted_idx]
y_val_sorted = y_val[sorted_idx]

# Compute upper and lower bounds of the epsilon tube
epsilon = best_random_params["epsilon"]
upper_bound = Y_pred_val + epsilon
lower_bound = Y_pred_val - epsilon

# ---------------------------------
# Plot Results: Optimized SVR Model & Training Loss
# ---------------------------------
plt.figure(figsize=(12, 5))

# Subplot 1: SVR Predictions
plt.subplot(1, 2, 1)
plt.scatter(X_val_sorted[:, 0], y_val_sorted, color='darkorange', label='Validation Data')
plt.plot(X_val_sorted[:, 0], Y_pred_val_sorted, color='grey', lw=2, label='SVR Model')
plt.fill_between(X_val_sorted[:, 0], lower_bound, upper_bound, color='lightblue', alpha=0.3, label='Epsilon Tube')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Support Vector Regression with Epsilon Tube')
plt.legend()
plt.grid(True)

# Subplot 2: Training Loss Curve
plt.subplot(1, 2, 2)
plt.plot(range(len(training_loss)), training_loss, color='green', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)

# Ensure the "plots" directory exists
os.makedirs("plots", exist_ok=True)

timestamp = time.strftime("%Y%m%d-%H%M%S")
plt.savefig(f"plots/svr_{loss_type}_{timestamp}_val.png")
print("Plot saved")

plt.show()

print("Validation costum report: ")
customRegressionReport(y_val, Y_pred_val, name="validation")

print("Validation costum report denormalized: ")
y_val_denorm = denormalize_zscore(y_val, train_set)
Y_pred_val_denorm = denormalize_zscore(Y_pred_val, train_set)
customRegressionReport(y_val_denorm, Y_pred_val_denorm, name="validation_denorm")

# ----------------------------------------------------------------
# Prediction on Test set
# ----------------------------------------------------------------

X_train_final = np.vstack((X_train, X_val))
y_train_final = np.vstack((y_train.reshape(-1, 1), y_val.reshape(-1, 1))).flatten()

# Train the model and track loss over iterations
training_loss = svr_final.fit(X_train_final, y_train_final)

# Check if alpha changes
print("Final alpha values:", svr_final.alpha[:5])  # Check first 5 values
print("Norm of alpha:", np.linalg.norm(svr_final.alpha))

# Predict and inverse transform the result
Y_pred_final = svr_final.predict(X_test)

# Sort validation data for a smooth plot
sorted_idx = np.argsort(X_test[:, 0])
X_test_sorted = X_test[sorted_idx]
Y_pred_test_sorted = Y_pred_final[sorted_idx]
y_test_sorted = y_test[sorted_idx]

# Compute upper and lower bounds of the epsilon tube
epsilon = best_random_params["epsilon"]
upper_bound_test = Y_pred_final + epsilon
lower_bound_test = Y_pred_final - epsilon

# ---------------------------------
# Plot Results: Optimized SVR Model & Training Loss
# ---------------------------------
plt.figure(figsize=(12, 5))

# Subplot 1: SVR Predictions
plt.subplot(1, 2, 1)
plt.scatter(X_test_sorted[:, 0], y_test_sorted, color='darkorange', label='Validation Data')
plt.plot(X_test_sorted[:, 0], Y_pred_test_sorted, color='navy', lw=2, label='SVR Model')
plt.fill_between(X_test_sorted[:, 0], lower_bound_test, upper_bound_test, color='lightblue', alpha=0.3, label='Epsilon Tube')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Optimized Support Vector Regression')
plt.legend()
plt.grid(True)

# Subplot 2: Training Loss Curve
plt.subplot(1, 2, 2)
plt.plot(range(len(training_loss)), training_loss, color='green', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)

# Ensure the "plots" directory exists
os.makedirs("plots", exist_ok=True)

timestamp = time.strftime("%Y%m%d-%H%M%S")
plt.savefig(f"plots/svr_{loss_type}_{timestamp}_test.png")
print("Plot saved")

plt.show()

# Denormalize predictions and training set
X_train_final_denorm = denormalize_zscore(X_train_final, train_set)
X_test_denorm = denormalize_zscore(X_test, train_set)
y_train_denorm = denormalize_zscore(y_train_final, train_set)
y_test_denorm = denormalize_zscore(y_test, train_set).flatten()

Y_pred_final_denorm = denormalize_zscore(Y_pred_final, train_set).flatten()

# Sort validation data for a smooth plot
sorted_idx = np.argsort(X_test_denorm[:, 0])
X_test_sorted_denorm = X_test_denorm[sorted_idx]
Y_pred_test_sorted_denorm = Y_pred_final_denorm[sorted_idx]
y_test_sorted_denorm = y_test_denorm[sorted_idx]

# Compute upper and lower bounds of the epsilon tube
upper_bound_test_denorm = Y_pred_final_denorm + epsilon
lower_bound_test_denorm = Y_pred_final_denorm - epsilon

# ---------------------------------
# Plot Denormalized Results: Optimized SVR Model & Training Loss
# ---------------------------------

plt.figure(figsize=(12, 5))

# Subplot 1: SVR Predictions
plt.subplot(1, 2, 1)
plt.scatter(X_test_sorted_denorm[:, 0], y_test_sorted_denorm, color='darkorange', label='Validation Data')
plt.plot(X_test_sorted_denorm[:, 0], Y_pred_test_sorted_denorm, color='navy', lw=2, label='SVR Model')
plt.fill_between(X_test_sorted_denorm[:, 0], lower_bound_test_denorm, upper_bound_test_denorm, color='lightblue', alpha=0.3, label='Epsilon Tube')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Optimized Support Vector Regression')
plt.legend()
plt.grid(True)

# Subplot 2: Training Loss Curve
plt.subplot(1, 2, 2)
plt.plot(range(len(training_loss)), training_loss, color='green', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)

# Ensure the "plots" directory exists
os.makedirs("plots", exist_ok=True)

timestamp = time.strftime("%Y%m%d-%H%M%S")
plt.savefig(f"plots/svr_{loss_type}_{timestamp}_test.png")
print("Plot saved")

plt.show()

print("Test costum report")
customRegressionReport(y_test_denorm, Y_pred_final_denorm, name="test")
