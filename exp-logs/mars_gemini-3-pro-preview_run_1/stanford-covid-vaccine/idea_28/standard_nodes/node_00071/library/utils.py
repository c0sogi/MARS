import os
import random
import numpy as np
import torch
from library.config import SEED


def seed_everything(seed: int = SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to the SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_mcrmse(preds, targets):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    Formula:
    MCRMSE = (1/Nt) * sum_j( sqrt( (1/n) * sum_i( (y_ij - y_hat_ij)^2 ) ) )

    Where:
        Nt is the number of target columns.
        n is the total number of scored positions (samples * sequence_length).
        j iterates over columns.
        i iterates over positions.

    Args:
        preds (torch.Tensor or np.ndarray): Predicted values.
                                            Expected shape: (Batch, Seq_Len, Num_Targets) or (N, Num_Targets).
        targets (torch.Tensor or np.ndarray): Ground truth values.
                                              Expected shape: (Batch, Seq_Len, Num_Targets) or (N, Num_Targets).

    Returns:
        torch.Tensor: The scalar MCRMSE value.
    """
    # Convert numpy arrays to tensors if necessary
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)

    # Ensure inputs are on the same device
    if preds.device != targets.device:
        targets = targets.to(preds.device)

    # Flatten the batch and sequence dimensions, keeping the target dimension (last dim) intact.
    # This handles both (Batch, Seq, Channels) and (Batch*Seq, Channels) shapes.
    num_targets = preds.shape[-1]
    preds_flat = preds.view(-1, num_targets)
    targets_flat = targets.view(-1, num_targets)

    # 1. Calculate MSE for each column (averaging over the flattened sample dimension)
    # shape: (Num_Targets,)
    mse = torch.mean((preds_flat - targets_flat) ** 2, dim=0)

    # 2. Calculate RMSE for each column
    rmse = torch.sqrt(mse)

    # 3. Calculate the mean of the RMSEs across columns
    mcrmse = torch.mean(rmse)

    return mcrmse
