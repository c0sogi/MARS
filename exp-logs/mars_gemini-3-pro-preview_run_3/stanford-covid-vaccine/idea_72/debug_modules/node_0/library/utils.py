import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error Loss.

    Calculates the MCRMSE across all 5 target columns as specified for the training objective.
    Formula: (1/Nt) * sum_{j=1}^{Nt} sqrt( (1/n) * sum_{i=1}^{n} (y_ij - yhat_ij)^2 )
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions of shape (Batch, Seq_Len_Out, Num_Targets).
                    Typically (Batch, 107, 5).
            targets: Ground truth of shape (Batch, Seq_Len_Tgt, Num_Targets).
                     Typically (Batch, 68, 5).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Slice inputs to match the target sequence length (usually 68)
        seq_len_tgt = targets.size(1)
        inputs_sliced = inputs[:, :seq_len_tgt, :]

        # Calculate MSE per column (averaging over batch and sequence dimensions)
        # Shape: (Num_Targets,)
        mse = torch.mean((inputs_sliced - targets) ** 2, dim=(0, 1))

        # Calculate RMSE per column
        rmse = torch.sqrt(mse)

        # Average RMSE across all target columns
        loss = torch.mean(rmse)

        return loss


def compute_val_metric(preds, targets):
    """
    Computes the MCRMSE metric for validation according to competition scoring rules.

    Logic:
    1. Slices predictions to the first 68 positions (Config.PRED_LEN).
    2. Filters columns to only the 3 scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C.
    3. Computes RMSE for each column over the provided dataset (global aggregation).
    4. Returns the mean of these RMSEs.

    Args:
        preds: Predictions (N, 107, 5) as torch.Tensor or np.ndarray.
        targets: Ground truth (N, 68, 5) as torch.Tensor or np.ndarray.

    Returns:
        float: The MCRMSE score.
    """
    # Ensure inputs are torch tensors
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    # Move to CPU for calculation
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    # 1. Slice predictions to scored sequence length (68)
    # We use the target length as the ground truth reference, which should match Config.PRED_LEN
    seq_len_scored = targets.shape[1]
    preds = preds[:, :seq_len_scored, :]

    # 2. Select Scored Columns
    # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Config.SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Identify indices of the scored columns
    col_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    preds_scored = preds[:, :, col_indices]
    targets_scored = targets[:, :, col_indices]

    # 3. Compute RMSE per column
    # Flatten batch and sequence dimensions to compute global RMSE per column
    # Shape becomes (N * 68, 3)
    preds_flat = preds_scored.reshape(-1, len(col_indices))
    targets_flat = targets_scored.reshape(-1, len(col_indices))

    mse = torch.mean((preds_flat - targets_flat) ** 2, dim=0)
    rmse = torch.sqrt(mse)

    # 4. Mean of RMSEs
    mcrmse = torch.mean(rmse)

    return mcrmse.item()
