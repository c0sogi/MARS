import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic operations for exact reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MCRMSE(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) metric.

    This metric calculates the root mean squared error for each target column separately,
    and then takes the average across columns.

    Modes:
        - 'train': Computes MCRMSE across all 5 target columns.
        - 'val': Computes MCRMSE strictly on the 3 scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C),
                 slicing predictions to the scored sequence length (68).
    """

    def __init__(self):
        super().__init__()
        self.config = Config()
        self.seq_scored = self.config.pred_len
        self.scored_indices = self.config.scored_indices
        self.device = self.config.device

    def forward(self, preds, trues, mode="train"):
        """
        Calculates the MCRMSE.

        Args:
            preds (torch.Tensor): Predicted values. Shape (B, Seq_Len, 5) or (B, Seq_Scored, 5).
            trues (torch.Tensor): Ground truth values. Shape (B, Seq_Scored, 5).
            mode (str): Evaluation mode. 'train' uses all columns, 'val' uses only scored columns.
                        Defaults to 'train'.

        Returns:
            torch.Tensor: The calculated MCRMSE scalar.
        """
        # Ensure inputs are tensors
        if not torch.is_tensor(preds):
            preds = torch.tensor(preds, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(trues):
            trues = torch.tensor(trues, dtype=torch.float32, device=self.device)

        # Ensure inputs are on the correct device
        if preds.device != trues.device:
            trues = trues.to(preds.device)

        # Slice predictions to the scored sequence length (68)
        # Targets are usually provided only for the first 68 bases.
        if preds.shape[1] > self.seq_scored:
            preds = preds[:, : self.seq_scored, :]

        # Sanity check for shapes
        if preds.shape != trues.shape:
            raise ValueError(
                f"Shape mismatch: Preds {preds.shape} vs Trues {trues.shape}"
            )

        # Calculate Squared Error
        mse = (preds - trues) ** 2

        # Calculate MSE per column (averaging over batch/sample and sequence length)
        # dim=0 is batch, dim=1 is sequence length
        mse_per_col = torch.mean(mse, dim=(0, 1))

        # Calculate RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col)

        if mode == "val":
            # Filter for the scored columns only: reactivity, deg_Mg_pH10, deg_Mg_50C
            # scored_indices is [0, 1, 3]
            rmse_per_col = rmse_per_col[self.scored_indices]

        # Average the RMSEs across the selected columns
        mcrmse = torch.mean(rmse_per_col)

        return mcrmse
