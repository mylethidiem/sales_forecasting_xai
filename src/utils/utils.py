import pickle

import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def fill_misisng_values(df):
    """Fill NaN values in the 'sales' column with the mean of non-NaN values"""
    df_filled = df.copy()
    df_filled["sales"] = df_filled["sales"].fillna(df_filled["sales"].mean())
    return df_filled


def correct_outliers(df, factor=3):
    """Identify and correct outliers in the 'sales' column by reducing them to the mean"""
    df_corrected = df.copy()

    # Identify outliers using z-score
    z_scores = (df_corrected["sales"] - df_corrected["sales"].mean()) / df_corrected[
        "sales"
    ].std()
    outlier_indices = np.abs(z_scores) > factor  # Adjust the threshold as needed
    # Correct outliers by reducing them to the mean
    df_corrected.loc[outlier_indices, "sales"] = df_corrected["sales"].mean()

    return df_corrected


def get_sample_stores(df: pd.DataFrame, store_id: int = 1) -> pd.DataFrame:
    """Get the sample stores with store_id"""
    grouped = df.groupby("store_id")
    sample_store = grouped.get_group((store_id))
    return sample_store


def save_data(df, file_path, file_format="feather"):
    """
    Save a DataFrame to a specified file format.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be saved.
    - file_path (str): The path where the file will be saved.
    - file_format (str): The format in which to save the file. Supported formats: 'feather', 'csv'.
                        Default is 'feather'.
    Example:
    ```python
    # Assuming df is the DataFrame you want to save
    save_data(df, 'output_data.feather', file_format='feather')
    ```

    Note:
    - Make sure to have the required libraries (pandas and feather-format) installed.
    """
    if file_format.lower() == "feather":
        # Save to Feather format
        df.to_feather(file_path)
        print(f"DataFrame saved to {file_path} in Feather format.")
    elif file_format.lower() == "csv":
        # Save to CSV format
        df.to_csv(file_path, index=False)
        print(f"DataFrame saved to {file_path} in CSV format.")
    else:
        print(
            f"Error: Unsupported file format '{file_format}'. Supported formats: 'feather', 'csv'."
        )


def flatten_prophet_predictions(predictions_dict):
    all_dfs = []

    for store_item, df in predictions_dict.items():
        df = df.copy()
        df["store_item"] = store_item
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def load_model(file_path):
    """
    Load a machine learning model from a file.

    Parameters:
    - file_path: The file path from where the model will be loaded.

    Returns:
    - The loaded model.
    """
    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)
            print(f"Sklearn model loaded from {file_path}")

    except (pickle.UnpicklingError, FileNotFoundError):
        # If loading as scikit-learn model fails or the file is not found,
        # assume it is a LightGBM model (scikit-learn API)
        model = lgbm.Booster(model_file=file_path)
        print(f"LightGBM (scikit-learn API) model loaded from {file_path}")

    return model


# Function to calculate WAPE (Weighted Absolute Percentage Error)
def weighted_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Weighted Absolute Percentage Error

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        WAPE value (percentage)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100 * np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
