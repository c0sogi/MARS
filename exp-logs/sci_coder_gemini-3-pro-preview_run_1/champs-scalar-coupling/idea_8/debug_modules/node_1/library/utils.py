import pandas as pd
import numpy as np
import os
from library.config import SUBMISSION_PATH


def reduce_mem_usage(df, verbose=True):
    """
    Iterates through all the columns of a dataframe and modifies the data type
    to reduce memory usage.
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            f"Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )
    return df


def calculate_log_mae(
    df, pred_col="prediction", target_col="scalar_coupling_constant", type_col="type"
):
    """
    Calculates the Log Mean Absolute Error metric for the competition.

    Metric definition: Log of the Mean Absolute Error, calculated for each
    scalar coupling type, and then averaged across types.

    Args:
        df (pd.DataFrame): DataFrame containing true values, predictions, and coupling types.
        pred_col (str): Column name for predictions.
        target_col (str): Column name for ground truth.
        type_col (str): Column name for coupling types.

    Returns:
        float: The final LogMAE score.
    """
    # Create a copy to avoid modifying the original dataframe
    eval_df = df[[type_col, target_col]].copy()
    eval_df[pred_col] = df[pred_col]

    # Calculate Absolute Error
    eval_df["abs_error"] = (eval_df[target_col] - eval_df[pred_col]).abs()

    # Group by type and calculate MAE for each type
    mae_per_type = eval_df.groupby(type_col)["abs_error"].mean()

    # Calculate Natural Log of MAE
    # Note: competition metric uses natural log
    log_mae_per_type = np.log(mae_per_type)

    # Average across types to get final score
    final_score = log_mae_per_type.mean()

    print("Validation Metric Breakdown (Log MAE per type):")
    # Print full precision as requested
    for t, score in log_mae_per_type.items():
        print(f"{t}: {score}")

    print(f"Final Log MAE Score: {final_score}")

    return final_score


def save_submission(ids, predictions, output_path=SUBMISSION_PATH):
    """
    Formats and saves the submission file.

    Args:
        ids (array-like): IDs of the pairs.
        predictions (array-like): Predicted scalar coupling constants.
        output_path (str): Path to save the CSV.
    """
    submission = pd.DataFrame({"id": ids, "scalar_coupling_constant": predictions})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
