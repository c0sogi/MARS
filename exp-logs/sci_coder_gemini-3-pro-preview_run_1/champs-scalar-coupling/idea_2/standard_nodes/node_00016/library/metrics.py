import numpy as np
import pandas as pd


def calculate_log_mae(y_true, y_pred, types):
    """
    Calculates the Log of the Mean Absolute Error for each scalar coupling type,
    and then averages across types.

    This is the primary evaluation metric for the task.
    Formula: Mean( Log( Mean( |y_true - y_pred| ) for each type ) )

    Args:
        y_true (array-like): Ground truth target values (scalar coupling constants).
        y_pred (array-like): Predicted target values.
        types (array-like): The coupling type for each sample (e.g., '1JHC', '2JHH').

    Returns:
        float: The final score (Log Mean Absolute Error).
    """
    # Ensure inputs are numpy arrays for consistency
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    types = np.array(types)

    # Check for shape consistency
    if len(y_true) != len(y_pred) or len(y_true) != len(types):
        raise ValueError(
            f"Shape mismatch: y_true ({len(y_true)}), y_pred ({len(y_pred)}), types ({len(types)})"
        )

    # Create a DataFrame to leverage efficient groupby operations
    df = pd.DataFrame({"type": types, "y_true": y_true, "y_pred": y_pred})

    # Calculate Absolute Error for every sample
    df["abs_error"] = np.abs(df["y_true"] - df["y_pred"])

    # Calculate Mean Absolute Error (MAE) for each coupling type
    mae_per_type = df.groupby("type")["abs_error"].mean()

    # Calculate the natural logarithm of the MAE for each type
    # Note: We assume MAE > 0. In regression tasks with floats, exact 0 error is rare.
    log_mae_per_type = np.log(mae_per_type)

    # The final metric is the average of these log-MAE scores across all types present
    final_score = log_mae_per_type.mean()

    return final_score


def get_log_mae_by_type(y_true, y_pred, types):
    """
    Computes the Log MAE for each coupling type individually.
    Useful for detailed error analysis and logging.

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Predicted target values.
        types (array-like): The coupling type for each sample.

    Returns:
        dict: A dictionary mapping coupling type (str) to its Log MAE score (float).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    types = np.array(types)

    df = pd.DataFrame({"type": types, "y_true": y_true, "y_pred": y_pred})

    df["abs_error"] = np.abs(df["y_true"] - df["y_pred"])
    mae_per_type = df.groupby("type")["abs_error"].mean()
    log_mae_per_type = np.log(mae_per_type)

    return log_mae_per_type.to_dict()
