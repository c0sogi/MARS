import numpy as np
import pandas as pd
from library.config import Config


def calculate_log_mae(
    df: pd.DataFrame,
    pred_col: str,
    target_col: str = Config.TARGET_COL,
    type_col: str = Config.TYPE_COL,
    verbose: bool = True,
) -> float:
    """
    Calculates the Log of the Mean Absolute Error for each scalar coupling type,
    and then averages the results across types.

    Metric = mean( log( mean( |y_true - y_pred| ) ) ) over all types.

    Args:
        df (pd.DataFrame): DataFrame containing truth, predictions, and types.
        pred_col (str): Name of the column containing predicted values.
        target_col (str): Name of the column containing true values.
        type_col (str): Name of the column containing coupling types.
        verbose (bool): If True, prints the metric for each type and the overall score.

    Returns:
        float: The final Log MAE score.
    """
    # specific columns extraction to avoid modifying original df
    eval_df = df[[type_col, target_col]].copy()
    eval_df["pred"] = df[pred_col]

    # Calculate Absolute Error
    eval_df["abs_error"] = (eval_df[target_col] - eval_df["pred"]).abs()

    # Calculate MAE for each type
    mae_per_type = eval_df.groupby(type_col)["abs_error"].mean()

    # Calculate Log(MAE)
    # Using natural logarithm (ln)
    log_mae_per_type = np.log(mae_per_type)

    # Average across types
    final_score = log_mae_per_type.mean()

    if verbose:
        print("Validation Metric Breakdown (Log MAE per type):")
        print(log_mae_per_type)
        print("-" * 30)
        print("Final Log MAE Score:")
        print(final_score)
        print("-" * 30)

    return final_score


def save_submission(
    ids: np.ndarray,
    preds: np.ndarray,
    output_path: str = Config.SUBMISSION_PATH,
    id_col: str = Config.ID_COL,
    target_col: str = Config.TARGET_COL,
) -> None:
    """
    Generates and saves the submission file in the required format.

    Args:
        ids (np.ndarray): Array of sample IDs.
        preds (np.ndarray): Array of predicted scalar coupling constants.
        output_path (str): Path to save the CSV file.
        id_col (str): Name of the ID column.
        target_col (str): Name of the target column.
    """
    submission_df = pd.DataFrame({id_col: ids, target_col: preds})

    # Ensure output directory exists (handled by Config usually, but safe to check if path is custom)
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
