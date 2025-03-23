import os

from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn import metrics
import time


def splitData(
        data,
):
    """
    Split data into training, validation and test sets.

    Parameters:
    - data: Input DataFrame with features and targets.

    Returns:
    - split_train_set: Training set.
    - split_validation_set: Validation set.
    - split_test_set: Test set.
    """

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


# function to split data to features and target
# pass the name of the target columns
def splitToFeaturesAndTarget(data, target_column="quality"):
    print("data: ", data)
    X = data.drop(target_column, axis=1).values.tolist()
    Y = data[target_column].values.tolist()
    return X, Y


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


def denormalize_zscore(predictions, data):
    """
    Denormalizes the predicted values back to the original scale using Z-score normalization.

    Parameters:
    - predictions: The normalized predicted values.
    - means: Means used for normalization.
    - stds: Standard deviations used for normalization.
    """

    # Initialize a copy of the predictions array
    denorm_predictions = predictions.copy()

    # Select the columns of interest for denormalization
    target_data = data['quality']

    mean = target_data.mean()
    std = target_data.std()
    denorm_predictions = predictions * std + mean

    return denorm_predictions


## function to perform MinMax normalization
def min_max_normalization(data, min_vals=None, max_vals=None):
    data_normalized = data.copy()
    # Select numeric columns only for normalization
    numeric_data = data_normalized.select_dtypes(include=["float64", "int64"])

    if min_vals is None or max_vals is None:
        # Calculate min and max from the training data
        min_vals = numeric_data.min()
        max_vals = numeric_data.max()

    # Normalize each numeric column separately using the min-max formula
    normalized_data = (numeric_data - min_vals) / (max_vals - min_vals)

    # Rejoin with non-numeric columns if needed
    non_numeric_data = data.select_dtypes(exclude=["float64", "int64"])
    final_data = pd.concat([non_numeric_data, normalized_data], axis=1)

    return final_data, min_vals, max_vals


## function to denormalize the predictions values for the ML-CUP24-TS.csv file
def min_max_denormalization(predictions, data, target_columns):
    """
    Denormalizes the predicted values back to the original scale.

    Parameters:
    - predictions: The normalized predicted values.
    - data: The original data (used to get min/max values for denormalization).
    - target_columns: List of target columns (e.g., ['TARGET_x', 'TARGET_y', 'TARGET_z']).
    """

    # Initialize a copy of the predictions array
    denorm_predictions = predictions.copy()

    # Select the columns of interest for denormalization
    target_data = data[target_columns]

    # Denormalize the predictions for each target column
    for idx, target_column in enumerate(target_columns):
        min_value = target_data[target_column].min()
        max_value = target_data[target_column].max()
        denorm_predictions[:, idx] = (
                predictions[:, idx] * (max_value - min_value) + min_value
        )

    return denorm_predictions


# Perform Preprocessing on the data
# 2. applying normalization
# 3. split the data to training and validation
# 4. split training data to X and Y
# 5. split validation data to X and Y
def preprocessData(
        data
):
    # split the data to training and validation
    split_train_set, split_validation_set, split_test_set = (
        splitData(data)
    )

    # Normalize the training set and get its means and stds
    train_set, train_means, train_stds = zscore_normalization(split_train_set)
    # Normalize the validation set using the training set's means and stds
    split_validation_set, _, _ = zscore_normalization(
        split_validation_set, means=train_means, stds=train_stds
    )
    split_test_set, _, _ = zscore_normalization(
        split_test_set, means=train_means, stds=train_stds
    )
    # split the training set to features and target
    train_X, train_Y = splitToFeaturesAndTarget(train_set)
    # split the validation set to features and target
    validation_X, validation_Y = splitToFeaturesAndTarget(split_validation_set)
    # split the test set to features and target
    test_X, test_Y = splitToFeaturesAndTarget(split_test_set)

    return (
        train_set,
        np.array(train_X),
        np.array(train_Y).reshape(-1, 1),
        np.array(validation_X),
        np.array(validation_Y).reshape(-1, 1),
        np.array(test_X),
        np.array(test_Y).reshape(-1, 1)
    )


# custom function to give a full report for regression
# takes the true values of the target , the predicted values, and the target column name
# it gives the MAE, MSE, RMSE and a scatter plot for the true vs predicted values
def customRegressionReport(trueValues, predictedValues, name="val"):
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
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal Fit (y=x)")
    plt.xlabel("True Values Quality")
    plt.ylabel("Predicted Values Quality")
    plt.title(f"True vs Predicted Quality ({name})")
    plt.grid(True)
    plt.legend()

    os.makedirs("plots", exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"plots/svr_customRegression_{name}_{timestamp}.png")
    plt.show()
