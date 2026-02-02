import numpy as np
import pandas as pd
import os
import random
import torch
from library.config import RANDOM_STATE


def set_seed(seed=RANDOM_STATE):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # Torch seeding
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # Torch not installed or not being used


def reduce_mem_usage(df, verbose=True):
    """
    Iterates through all the columns of a dataframe and modifies the data type
    to reduce memory usage.

    Args:
        df (pd.DataFrame): The dataframe to optimize.
        verbose (bool): Whether to print the memory reduction statistics.

    Returns:
        pd.DataFrame: The optimized dataframe.
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type.name != "category":
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
                # Use float32 instead of float16 to maintain precision for physics calculations
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
            f"Memory usage reduced to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )

    return df


def calculate_competition_metric(
    df,
    prediction_col="prediction",
    target_col="scalar_coupling_constant",
    type_col="type",
):
    """
    Calculates the competition metric: Log of the Mean Absolute Error,
    calculated for each scalar coupling type, and then averaged across types.

    Metric = 1/T * Sum( log( MAE_t ) )

    Args:
        df (pd.DataFrame): DataFrame containing predictions, targets, and types.
        prediction_col (str): Name of the prediction column.
        target_col (str): Name of the target column.
        type_col (str): Name of the coupling type column.

    Returns:
        float: The calculated score.
    """
    # Ensure columns exist
    required_cols = [prediction_col, target_col, type_col]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns for metric calculation: {missing}")

    # Calculate Absolute Error
    df = df.copy()
    df["abs_error"] = (df[target_col] - df[prediction_col]).abs()

    # Group by type and calculate MAE
    mae_per_type = df.groupby(type_col)["abs_error"].mean()

    # Calculate Log MAE
    # We use natural log (np.log) as is standard unless base 10 is specified.
    # Usually in this specific context, it's log base e.
    log_mae_per_type = np.log(mae_per_type)

    # Print breakdown per type (useful for the Stratified Ensemble approach)
    print("Metric Breakdown by Type:")
    print(f"{'Type':<10} | {'MAE':<15} | {'Log(MAE)':<15}")
    print("-" * 46)
    for t, mae in mae_per_type.items():
        log_mae = log_mae_per_type[t]
        print(f"{t:<10} | {mae:.8f}       | {log_mae:.8f}")
    print("-" * 46)

    # Final Score: Mean of Log MAEs
    final_score = log_mae_per_type.mean()

    return final_score
