import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time


# Split data into training, validation and test sets.
def splitData(
        data,
):
    # Convert to NumPy array
    data_array = np.array(data)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Shuffle the data
    shuffled_indices = np.random.permutation(len(data_array))

    # Calculate the split index
    train_index = int(len(data_array) * 0.60)  # 60% for training
    validation_index = int(len(data_array) * 0.80)  # 20% for validation, 80% cumulative

    # Split the data
    train_indices = shuffled_indices[:train_index]
    validation_indices = shuffled_indices[train_index:validation_index]
    test_indices = shuffled_indices[validation_index:]

    split_train_set = data_array[train_indices]
    split_validation_set = data_array[validation_indices]
    split_test_set = data_array[test_indices]

    # Convert to DataFrames
    split_train_set_df = pd.DataFrame(split_train_set, columns=data.columns)
    split_validation_set_df = pd.DataFrame(split_validation_set, columns=data.columns)
    split_test_set_df = pd.DataFrame(split_test_set, columns=data.columns)

    return split_train_set_df, split_validation_set_df, split_test_set_df


# function to perform Zscore normalization
def zscore_normalization(data, means=None, stds=None):
    # Create a copy of the data to avoid modifying the original DataFrame
    data_normalized = data.copy()

    # Select only numeric columns
    numeric_data = data_normalized.select_dtypes(include=["float64", "int64"])
    columns_to_normalize = numeric_data.columns

    # Calculate means and standard deviations if not provided
    if means is None or stds is None:
        means = numeric_data.mean(axis=0)
        stds = numeric_data.std(axis=0)

    # Avoid division by zero for constant columns
    stds_replaced = stds.replace(0, 1)

    # Apply Z-score normalization
    data_normalized[columns_to_normalize] = (numeric_data - means) / stds_replaced

    return data_normalized, means, stds


# denormalize target feature
def denormalize_price(predictions, y_mean, y_std):
    y_mean = y_mean.values[0] if isinstance(y_mean, pd.Series) else y_mean
    y_std = y_std.values[0] if isinstance(y_std, pd.Series) else y_std
    return predictions * y_std + y_mean


# denormalize features
def denormalize(predictions, mean, std):
    mean = mean.values[0] if isinstance(mean, pd.Series) else mean
    std = std.values[0] if isinstance(std, pd.Series) else std
    return predictions * std + mean


# remove outliers
def remove_outliers(df, method="iqr"):
    df_cleaned = df.copy()
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns

    if method == "zscore":
        from scipy.stats import zscore
        z_scores = np.abs(zscore(df_cleaned[numeric_cols]))
        mask = (z_scores < 3).all(axis=1)
        return df_cleaned[mask]

    elif method == "iqr":
        Q1 = df_cleaned[numeric_cols].quantile(0.25)
        Q3 = df_cleaned[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((df_cleaned[numeric_cols] < (Q1 - 1.5 * IQR)) |
                 (df_cleaned[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        return df_cleaned[mask]

    else:
        raise ValueError("Metodo non valido. Usa 'zscore' o 'iqr'.")


# preprocessing of data
def preprocessData(data, outlier_method="iqr"):
    # Remove outliers
    data = remove_outliers(data, method=outlier_method)

    # Split the data
    split_train_set, split_validation_set, split_test_set = splitData(data)

    # Extract target variable
    train_Y = split_train_set["price"]
    validation_Y = split_validation_set["price"]
    test_Y = split_test_set["price"]

    # Drop target column from features
    train_X = split_train_set.drop(["price"], axis=1)
    validation_X = split_validation_set.drop(["price"], axis=1)
    test_X = split_test_set.drop(["price"], axis=1)

    # Normalize features using MinMaxScaler
    scaler_X = MinMaxScaler()
    train_X = scaler_X.fit_transform(train_X)
    validation_X = scaler_X.transform(validation_X)
    test_X = scaler_X.transform(test_X)

    # Normalize target using Z-score (unchanged)
    train_Y, y_mean, y_std = zscore_normalization(pd.DataFrame(train_Y))
    validation_Y, _, _ = zscore_normalization(pd.DataFrame(validation_Y), means=y_mean, stds=y_std)
    test_Y, _, _ = zscore_normalization(pd.DataFrame(test_Y), means=y_mean, stds=y_std)

    return (
        np.array(train_X), np.array(train_Y).reshape(-1, 1),
        np.array(validation_X), np.array(validation_Y).reshape(-1, 1),
        np.array(test_X), np.array(test_Y).reshape(-1, 1),
        y_mean, y_std,
        pd.Series(scaler_X.data_min_), pd.Series(scaler_X.data_max_ - scaler_X.data_min_)
    )


# custom function to give a full report for regression
# takes the true values of the target , the predicted values, and the target column name
# it gives the MAE, MSE, RMSE and a scatter plot for the true vs predicted values
def customRegressionReport(trueValues, predictedValues, labels=None, name="val"):
    # Compute metrics
    mse = np.mean((predictedValues - trueValues) ** 2)
    mee = np.mean(np.abs(predictedValues - trueValues))  # Corretto!
    rmse = np.sqrt(mse)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Euclidean Error (MEE): {mee:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # Scatter plot true vs predicted
    plt.figure(figsize=(6, 6))
    min_val = min(trueValues.min(), predictedValues.min())
    max_val = max(trueValues.max(), predictedValues.max())

    plt.scatter(trueValues, predictedValues, alpha=0.5, color="blue", label="Predicted Values")
    plt.scatter(trueValues, trueValues, alpha=0.5, color="green", label="True Values")

    # Adding labels to points if provided
    if labels is not None:
        for i, label in enumerate(labels):
            plt.annotate(label, (trueValues[i], predictedValues[i]), fontsize=8, alpha=0.7)

    plt.plot(
        [min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal Fit (y=x)"
    )  # Line y = x

    plt.xlabel("True Values Quality")
    plt.ylabel("Predicted Values Quality")
    plt.title(f"True vs Predicted Quality ({name})")
    plt.legend()
    plt.grid(True)
    plt.legend()

    os.makedirs("plots/scatter", exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"plots/scatter/svr_customRegression_{name}_{timestamp}.png")
    plt.show()


# function to plot convergence curves
def plot_convergence_curves(hist, title_prefix="SVR", tol=1e-6):
    import os, time
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs("plots/convergence", exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    fname = f"plots/convergence/{title_prefix}_convergence_full_{timestamp}.png"

    it_s = np.array(hist['iter_smooth'])
    beta_s = np.array(hist['beta_norms_smooth'])
    grad_s = np.array(hist['grad_norms_smooth'])
    Q_s = np.array(hist['Q_mu_smooth'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    fig.suptitle(f"{title_prefix} Convergenza", fontsize=16)

    # 1) Δβ
    ax = axes[0]
    ax.plot(it_s, beta_s, color='tab:blue', label="‖β - β_prev‖ (smoothed)")
    ax.set_title("Convergence Speed")
    ax.set_xlabel("Iterazione")
    ax.set_ylabel("Update Norm")
    ax.grid(True)
    ax.legend()

    # 2) ∥∇Q_μ∥
    ax = axes[1]
    ax.plot(it_s, grad_s, color='tab:orange', label="‖∇Q_μ‖ (smoothed)")
    ax.set_title("Gradient Magnitude")
    ax.set_xlabel("Iterazione")
    ax.set_ylabel("Gradient Norm")
    ax.grid(True)
    ax.legend()

    # 3) Dual objective Q_μ
    ax = axes[2]
    ax.plot(it_s, Q_s, color='tab:green', label="Q_μ(β) (smoothed)")
    ax.set_title("Objective Convergence")
    ax.set_xlabel("Iterazione")
    ax.set_ylabel("Q_μ")
    ax.grid(True)
    ax.legend()

    fig.savefig(fname)
    print(f"[✓] Saved full convergence plot to {fname}")
    plt.show()


# function to plot duality gap
def plot_duality_gap(history, save_dir='plots/convergence', kernel_name="RBF"):
    import os, time
    import numpy as np
    import matplotlib.pyplot as plt

    # Extract the dual objective sequence Q_mu from history
    Q_list = np.array(history['Q_mu'])
    # Iterations 1…K as floats for negative powers
    iterations = np.arange(1, len(Q_list) + 1, dtype=float)
    # Estimate Q* as the maximum observed value
    Q_star = Q_list.max()
    gap = Q_star - Q_list

    # Reference line O(1/k^2)
    reference = gap[0] * iterations ** (-2)

    plt.figure(figsize=(6, 4))
    # Plot both curves on log–log axes for direct comparison
    plt.loglog(iterations, gap, 'o-', label='Duality gap Δₖ', markersize=4)
    plt.loglog(iterations, reference, '--', label='Reference $O(1/k^2)$')

    plt.xlabel('Iteration $k$')
    plt.ylabel('Duality gap Δₖ')
    plt.title(f'Duality Gap Convergence — {kernel_name} Kernel')
    plt.grid(True, which='both', linestyle=':')
    plt.legend()

    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(
        save_dir,
        f"duality_gap_{kernel_name}_{time.strftime('%Y%m%d-%H%M%S')}.png"
    )
    plt.savefig(filename)
    print(f"[✓] Saved duality gap plot to {filename}")
    plt.show()
