import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse_loss(y_pred, y_true):
    """
    Calculates the MCRMSE (Mean Columnwise Root Mean Squared Error) for validation.

    This metric computes the RMSE for each of the 3 scored columns separately
    and then takes the average of those RMSEs.

    Args:
        y_pred: Predicted values, shape (Batch, Seq_Len, 3)
        y_true: Ground truth values, shape (Batch, Seq_Len, 3)

    Returns:
        float: The MCRMSE score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Slice to the scored length (first 68 positions)
    # Config.PRED_LEN should be 68
    pred_scored = y_pred[:, : Config.PRED_LEN, :]
    true_scored = y_true[:, : Config.PRED_LEN, :]

    rmses = []
    num_targets = true_scored.shape[2]

    # Calculate RMSE for each target column individually
    for i in range(num_targets):
        # Flatten batch and sequence dimensions for the specific column
        p_col = pred_scored[:, :, i].flatten()
        t_col = true_scored[:, :, i].flatten()

        # Calculate MSE
        mse = np.mean((p_col - t_col) ** 2)

        # Calculate RMSE
        rmse = np.sqrt(mse)
        rmses.append(rmse)

    # Return the mean of the RMSEs
    return np.mean(rmses)


def format_submission(test_ids, predictions, save_dir=Config.SUBMISSION_DIR):
    """
    Formats predictions into the competition CSV format and saves the file.

    Args:
        test_ids: List or array of test sample IDs.
        predictions: Numpy array or Tensor of shape (N_samples, 107, 3).
                     Columns correspond to [reactivity, deg_Mg_pH10, deg_Mg_50C].
        save_dir: Directory path to save the submission file.
    """
    # Ensure predictions are numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    os.makedirs(save_dir, exist_ok=True)

    submission_rows = []

    # Iterate through each sample
    for i, sample_id in enumerate(test_ids):
        sample_preds = predictions[i]  # Shape (107, 3)

        # Iterate through all sequence positions (0 to 106)
        for pos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{pos}"

            # Extract predicted values for the scored columns
            reactivity = sample_preds[pos, 0]
            deg_Mg_pH10 = sample_preds[pos, 1]
            deg_Mg_50C = sample_preds[pos, 2]

            # Fill unscored columns with 0.0
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_rows.append(
                [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            )

    columns = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]

    sub_df = pd.DataFrame(submission_rows, columns=columns)
    save_path = os.path.join(save_dir, Config.SUBMISSION_FILE)
    sub_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
