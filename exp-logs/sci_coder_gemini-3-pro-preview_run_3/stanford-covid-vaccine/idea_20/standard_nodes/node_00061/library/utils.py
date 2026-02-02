import os
import random
import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Configures CuDNN for deterministic execution.

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


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error Loss.
    Computes the mean of the RMSEs for each target column.

    Formula:
    MCRMSE = (1/Nt) * sum_j( sqrt( (1/n) * sum_i( (y_ij - y_hat_ij)^2 ) ) )
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds: Predictions tensor of shape (Batch, Seq_Len, Num_Targets)
                   or (Batch*Seq_Len, Num_Targets).
            targets: Ground truth tensor of shape (Batch, Seq_Len, Num_Targets)
                     or (Batch*Seq_Len, Num_Targets).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Flatten to (N, Num_Targets) if input is 3D
        if preds.dim() == 3:
            preds = preds.view(-1, preds.shape[-1])
        if targets.dim() == 3:
            targets = targets.view(-1, targets.shape[-1])

        # Compute MSE per column: (1/n) * sum_i( (y_ij - y_hat_ij)^2 )
        mse = torch.mean((preds - targets) ** 2, dim=0)

        # Compute RMSE per column
        rmse = torch.sqrt(mse)

        # Compute Mean of RMSEs across columns
        loss = torch.mean(rmse)

        return loss


def global_mcrmse(preds_list, targets_list):
    """
    Computes MCRMSE over the entire dataset by concatenating batches first.
    This ensures the metric is not biased by batch averaging.

    Args:
        preds_list: List of numpy arrays or torch tensors containing predictions.
        targets_list: List of numpy arrays or torch tensors containing ground truth.

    Returns:
        float: The global MCRMSE score.
    """
    # Convert tensors to numpy arrays if necessary
    p_list = [
        p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p
        for p in preds_list
    ]
    t_list = [
        t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t
        for t in targets_list
    ]

    # Concatenate all batches along the first dimension (samples)
    preds = np.concatenate(p_list, axis=0)
    targets = np.concatenate(t_list, axis=0)

    # Flatten if 3D (Batch, Seq, Channels) -> (Batch*Seq, Channels)
    # This treats every position in every sequence as an independent sample for the statistic
    if preds.ndim == 3:
        preds = preds.reshape(-1, preds.shape[-1])
    if targets.ndim == 3:
        targets = targets.reshape(-1, targets.shape[-1])

    # Calculate MCRMSE
    # 1. MSE per column
    mse = np.mean((preds - targets) ** 2, axis=0)

    # 2. RMSE per column
    rmse = np.sqrt(mse)

    # 3. Mean RMSE across columns
    score = np.mean(rmse)

    return float(score)
