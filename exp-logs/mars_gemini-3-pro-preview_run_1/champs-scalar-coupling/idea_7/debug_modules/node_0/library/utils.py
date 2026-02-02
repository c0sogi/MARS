import time
import numpy as np
import pandas as pd
from contextlib import contextmanager
from library.config import COUPLING_TYPES


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
                # We avoid float16 for numerical stability in scientific computing
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
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def calculate_log_mae(y_true, y_pred, types, verbose=False):
    """
    Calculates the competition metric: Log of the Mean Absolute Error,
    calculated for each scalar coupling type, and then averaged across types.

    Args:
        y_true (np.array or pd.Series): True target values.
        y_pred (np.array or pd.Series): Predicted values.
        types (np.array or pd.Series): Coupling types (e.g., '1JHC').
        verbose (bool): If True, prints the breakdown per type.

    Returns:
        float: The final Log MAE score (averaged across types).
        pd.DataFrame: A DataFrame containing MAE and Log MAE for each type.
    """
    # Create a temporary DataFrame for efficient grouping
    df_metrics = pd.DataFrame({"type": types, "y_true": y_true, "y_pred": y_pred})

    # Calculate Absolute Error
    df_metrics["abs_error"] = (df_metrics["y_true"] - df_metrics["y_pred"]).abs()

    # Group by type and calculate MAE
    mae_per_type = df_metrics.groupby("type")["abs_error"].mean()

    # Calculate Log MAE (natural log)
    # Note: We assume MAE > 0. In perfect prediction scenarios, this would be -inf.
    # However, for this regression task, exact 0.0 error is practically impossible.
    log_mae_per_type = np.log(mae_per_type)

    # Calculate the final score (mean of the log MAEs)
    final_score = log_mae_per_type.mean()

    # Create breakdown DataFrame
    breakdown = pd.DataFrame(
        {
            "mae": mae_per_type,
            "log_mae": log_mae_per_type,
            "count": df_metrics.groupby("type")["abs_error"].count(),
        }
    )

    if verbose:
        print("\nMetric Breakdown by Type:")
        print(breakdown)
        print(f"\nFinal Log MAE Score: {final_score}")

    return final_score, breakdown


@contextmanager
def timer(title):
    """
    Context manager to log the execution time of a block of code.
    """
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def print_full_precision_metrics(metrics_dict, prefix=""):
    """
    Prints metrics with full precision as requested.

    Args:
        metrics_dict (dict): Dictionary of metric names and values.
        prefix (str): Optional prefix for the print statement.
    """
    for key, value in metrics_dict.items():
        print(f"{prefix}{key}: {value}")
