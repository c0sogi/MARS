import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    Metric Definition:
    MCRMSE = (1/Nt) * sum_j( sqrt( (1/n) * sum_i( (y_ij - y_hat_ij)^2 ) ) )

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values of shape (N_samples, seq_scored, n_targets).
        y_pred (np.ndarray or torch.Tensor): Predicted values of shape (N_samples, seq_scored, n_targets).

    Returns:
        float: The calculated MCRMSE score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate squared errors
    squared_errors = (y_true - y_pred) ** 2

    # Calculate MSE for each target column (averaging over samples and sequence positions)
    # Axis 0 is samples, Axis 1 is sequence positions. We average over both.
    mse_per_col = np.mean(squared_errors, axis=(0, 1))

    # Calculate RMSE for each target column
    rmse_per_col = np.sqrt(mse_per_col)

    # Calculate the mean of the RMSEs across columns
    score = np.mean(rmse_per_col)

    return float(score)


def format_submission(test_ids, predictions, save_path=Config.SUBMISSION_PATH):
    """
    Formats the predictions into the required submission CSV format.

    The model predicts 3 targets: [reactivity, deg_Mg_pH10, deg_Mg_50C].
    The submission requires 5 targets: [reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C].
    Unscored targets (deg_pH10, deg_50C) and positions > 68 are filled with 0.0.

    Args:
        test_ids (list): List of sample IDs (strings).
        predictions (np.ndarray or torch.Tensor): Model predictions of shape (N_samples, 68, 3).
        save_path (str): Path to save the generated CSV file.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    # Prepare lists for DataFrame construction
    ids_seqpos = []
    reactivity = []
    deg_Mg_pH10 = []
    deg_pH10 = []
    deg_Mg_50C = []
    deg_50C = []

    # Iterate through each sample
    for i, sample_id in enumerate(test_ids):
        sample_preds = predictions[i]  # Shape: (68, 3)

        # Iterate through the full sequence length (107)
        for seqpos in range(Config.SEQ_LEN):
            # ID format: id_sequenceposition
            ids_seqpos.append(f"{sample_id}_{seqpos}")

            if seqpos < Config.PRED_LEN:
                # Within the scored region (0-67), use model predictions
                # Model output mapping: 0 -> reactivity, 1 -> deg_Mg_pH10, 2 -> deg_Mg_50C
                reactivity.append(sample_preds[seqpos, 0])
                deg_Mg_pH10.append(sample_preds[seqpos, 1])
                deg_Mg_50C.append(sample_preds[seqpos, 2])
            else:
                # Outside scored region, fill with 0.0
                reactivity.append(0.0)
                deg_Mg_pH10.append(0.0)
                deg_Mg_50C.append(0.0)

            # Unscored columns are always 0.0
            deg_pH10.append(0.0)
            deg_50C.append(0.0)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {
            "id_seqpos": ids_seqpos,
            "reactivity": reactivity,
            "deg_Mg_pH10": deg_Mg_pH10,
            "deg_pH10": deg_pH10,
            "deg_Mg_50C": deg_Mg_50C,
            "deg_50C": deg_50C,
        }
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
