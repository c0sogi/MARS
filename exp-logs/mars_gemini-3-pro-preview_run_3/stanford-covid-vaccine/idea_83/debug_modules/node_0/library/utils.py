import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def MCRMSE(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) for the
    competition metric.

    Only specific columns and the first 68 sequence positions are scored.

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth values.
                                             Shape: (Batch, Seq_Len, 5)
        y_pred (torch.Tensor or np.ndarray): Predicted values.
                                             Shape: (Batch, Seq_Len, 5)

    Returns:
        float: The calculated MCRMSE score.
    """
    # Ensure inputs are tensors
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    # Move to CPU for calculation if necessary
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()

    # Get configuration
    seq_scored = Config.SEQ_SCORED
    all_targets = Config.get_target_columns()
    scored_cols = Config.get_scored_columns()

    # Identify indices of columns to score
    # all_targets: ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
    # scored_cols: ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']
    scored_indices = [i for i, col in enumerate(all_targets) if col in scored_cols]

    # Slice data:
    # 1. Take only the first 'seq_scored' positions (0 to 67)
    # 2. Select only the scored columns
    y_true_sliced = y_true[:, :seq_scored, scored_indices]
    y_pred_sliced = y_pred[:, :seq_scored, scored_indices]

    # Calculate MSE per column
    # Flatten batch and sequence dimensions: (Batch * Seq_Scored, Num_Scored_Cols)
    # Note: We compute RMSE per column, then average the RMSEs.
    mse = torch.mean((y_true_sliced - y_pred_sliced) ** 2, dim=(0, 1))
    rmse = torch.sqrt(mse)

    # Calculate Mean of RMSEs
    mcrmse = torch.mean(rmse)

    return mcrmse.item()


def create_submission_file(ids, preds, output_path=Config.SUBMISSION_PATH):
    """
    Formats predictions into the competition submission format and saves to CSV.

    Args:
        ids (list): List of sample IDs (strings).
        preds (np.ndarray): Prediction tensor of shape (N_samples, 107, 5).
        output_path (str): Path to save the submission CSV.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    target_cols = Config.get_target_columns()
    seq_len = Config.SEQ_LEN  # Should be 107

    # Prepare lists for DataFrame construction
    id_seqpos_list = []
    preds_flat = []

    # Iterate through samples
    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # Shape: (107, 5)

        # Validate shape
        if sample_preds.shape != (seq_len, len(target_cols)):
            raise ValueError(
                f"Prediction shape mismatch for sample {sample_id}. "
                f"Expected ({seq_len}, {len(target_cols)}), got {sample_preds.shape}"
            )

        for j in range(seq_len):
            # Construct ID_seqpos
            id_seqpos_list.append(f"{sample_id}_{j}")
            preds_flat.append(sample_preds[j])

    # Create DataFrame
    preds_flat = np.array(preds_flat)
    submission_df = pd.DataFrame(preds_flat, columns=target_cols)
    submission_df.insert(0, "id_seqpos", id_seqpos_list)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    # print(f"Submission file saved to {output_path}")
