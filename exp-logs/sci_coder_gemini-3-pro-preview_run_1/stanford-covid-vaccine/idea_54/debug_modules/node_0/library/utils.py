import os
import numpy as np
import pandas as pd
from library.config import Config, seed_everything


def mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is calculated by:
    1. Computing the RMSE for each of the 3 scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C)
       across all samples and scored sequence positions.
    2. Taking the arithmetic mean of these column-wise RMSEs.

    Args:
        y_true (np.ndarray): Ground truth values of shape (N_samples, seq_scored, 3).
        y_pred (np.ndarray): Predicted values of shape (N_samples, seq_scored, 3).

    Returns:
        float: The MCRMSE score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate Mean Squared Error for each column
    # Average over axis 0 (samples) and axis 1 (sequence positions)
    # Result shape: (3,)
    mse_per_col = np.mean((y_true - y_pred) ** 2, axis=(0, 1))

    # Calculate RMSE for each column
    rmse_per_col = np.sqrt(mse_per_col)

    # Return the mean of the column RMSEs
    return np.mean(rmse_per_col)


def format_submission(test_ids, predictions, save_path=Config.SUBMISSION_PATH):
    """
    Formats predictions into the required submission CSV format.

    Maps the model's 3-channel output to the 5 required columns:
    - reactivity -> reactivity
    - deg_Mg_pH10 -> deg_Mg_pH10
    - (zero) -> deg_pH10
    - deg_Mg_50C -> deg_Mg_50C
    - (zero) -> deg_50C

    Args:
        test_ids (list or np.ndarray): List of sample IDs corresponding to the predictions.
        predictions (np.ndarray): Array of predictions with shape (N, seq_length, 3).
                                  The 3 channels correspond to [reactivity, deg_Mg_pH10, deg_Mg_50C].
        save_path (str): Path to save the generated CSV file.
    """
    submission_data = []

    # Ensure predictions match the number of IDs
    if len(test_ids) != len(predictions):
        raise ValueError(
            f"Length mismatch: {len(test_ids)} IDs vs {len(predictions)} predictions."
        )

    for i, sample_id in enumerate(test_ids):
        # Get prediction for this sample: shape (107, 3)
        pred = predictions[i]

        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"

            # Extract predicted values from the model output
            # Model output order: reactivity, deg_Mg_pH10, deg_Mg_50C
            val_reactivity = pred[seqpos, 0]
            val_deg_Mg_pH10 = pred[seqpos, 1]
            val_deg_Mg_50C = pred[seqpos, 2]

            # Fill unscored columns with 0.0 as per task requirements
            val_deg_pH10 = 0.0
            val_deg_50C = 0.0

            # Append row in the correct order specified by Config.ALL_PRED_COLS:
            # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
            submission_data.append(
                [
                    row_id,
                    val_reactivity,
                    val_deg_Mg_pH10,
                    val_deg_pH10,
                    val_deg_Mg_50C,
                    val_deg_50C,
                ]
            )

    # Create DataFrame
    # Columns: id_seqpos + the 5 target columns
    columns = ["id_seqpos"] + Config.ALL_PRED_COLS
    sub_df = pd.DataFrame(submission_data, columns=columns)

    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sub_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
