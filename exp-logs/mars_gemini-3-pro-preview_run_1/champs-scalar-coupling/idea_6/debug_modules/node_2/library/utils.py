import os
import numpy as np
import pandas as pd
import xgboost as xgb
from library import config


def calculate_log_mae(df_true, y_pred):
    """
    Calculates the Log Mean Absolute Error metric for the competition.

    The metric is defined as the Log of the Mean Absolute Error, calculated for
    each scalar coupling type, and then averaged across types.

    Args:
        df_true (pd.DataFrame): DataFrame containing 'type' and 'scalar_coupling_constant'.
                                Must be aligned with y_pred.
        y_pred (np.ndarray or pd.Series): Predicted scalar coupling constants.

    Returns:
        float: The final averaged Log MAE metric.
        dict: A dictionary containing the Log MAE for each coupling type.
    """
    # Work on a copy to avoid side effects
    df = df_true[["type", "scalar_coupling_constant"]].copy()
    df["prediction"] = y_pred

    # Calculate Absolute Error
    df["abs_error"] = np.abs(df["scalar_coupling_constant"] - df["prediction"])

    # Calculate Mean Absolute Error (MAE) per type
    mae_per_type = df.groupby("type")["abs_error"].mean()

    # Calculate Log of MAE (Natural Logarithm)
    # Note: Competition metric is mean(log(MAE_t)) over types t
    log_mae_per_type = np.log(mae_per_type)

    # Average across types to get the single score
    final_metric = log_mae_per_type.mean()

    return final_metric, log_mae_per_type.to_dict()


def save_model(model, type_name):
    """
    Saves a trained XGBoost model to a JSON file.

    Args:
        model (xgb.XGBRegressor): The trained XGBoost model.
        type_name (str): The coupling type (e.g., '1JHC'), used for the filename.
    """
    # Define directory and ensure it exists
    model_dir = os.path.join(config.WORKING_DIR, "xgb_models")
    os.makedirs(model_dir, exist_ok=True)

    # Define file path
    file_path = os.path.join(model_dir, f"{type_name}.json")

    # Save using XGBoost's JSON format (safer and more portable than pickle)
    model.save_model(file_path)
    print(f"Model for {type_name} saved to {file_path}")


def load_model(type_name):
    """
    Loads an XGBoost model from a JSON file.

    Args:
        type_name (str): The coupling type (e.g., '1JHC').

    Returns:
        xgb.XGBRegressor: The loaded XGBoost model.
    """
    model_dir = os.path.join(config.WORKING_DIR, "xgb_models")
    file_path = os.path.join(model_dir, f"{type_name}.json")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found at: {file_path}")

    # Initialize a new regressor and load the artifact
    model = xgb.XGBRegressor()
    model.load_model(file_path)

    return model


def format_submission(test_ids, predictions):
    """
    Formats the predictions into the required submission DataFrame and saves it.

    Args:
        test_ids (pd.Series or list): The 'id' column from the test set.
        predictions (pd.Series or list): The predicted 'scalar_coupling_constant' values.
    """
    # Create the submission DataFrame
    submission_df = pd.DataFrame(
        {"id": test_ids, "scalar_coupling_constant": predictions}
    )

    # Ensure the output directory exists
    submission_dir = os.path.dirname(config.SUBMISSION_PATH)
    os.makedirs(submission_dir, exist_ok=True)

    # Save to CSV
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")
