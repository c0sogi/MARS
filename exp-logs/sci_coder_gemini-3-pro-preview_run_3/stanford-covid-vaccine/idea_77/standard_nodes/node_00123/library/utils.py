import os
import random
import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers to ensure reproducibility
    across python, numpy, and torch (CPU and CUDA).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MCRMSE(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) metric.

    This module computes the MCRMSE loss, which is the mean of the RMSE values
    calculated for each target column. It supports slicing the input sequences
    to a specific length (e.g., 68) and filtering specific target columns
    (e.g., for validation scoring).
    """

    def __init__(self, pred_len=68, scored_indices=None):
        """
        Args:
            pred_len (int): The length of the sequence to score. Predictions and targets
                            will be sliced to this length along the sequence dimension.
                            Default is 68.
            scored_indices (list of int, optional): A list of column indices to include
                                                    in the final metric calculation. If None,
                                                    all columns are used.
        """
        super().__init__()
        self.pred_len = pred_len
        self.scored_indices = scored_indices

    def forward(self, preds, targets):
        """
        Computes the MCRMSE between predictions and targets.

        Args:
            preds (torch.Tensor): Predicted values of shape (Batch, Seq_Len, Targets).
            targets (torch.Tensor): Ground truth values of shape (Batch, Seq_Len, Targets).

        Returns:
            torch.Tensor: A scalar tensor representing the MCRMSE.
        """
        # Slice predictions and targets to the scored length
        # We check shape to ensure we don't slice if data is already pre-sliced or shorter
        p_slice = (
            preds[:, : self.pred_len, :] if preds.shape[1] >= self.pred_len else preds
        )
        t_slice = (
            targets[:, : self.pred_len, :]
            if targets.shape[1] >= self.pred_len
            else targets
        )

        # Calculate Mean Squared Error (MSE) for each column
        # Averaging over Batch (dim 0) and Sequence (dim 1)
        mse = torch.mean((p_slice - t_slice) ** 2, dim=(0, 1))

        # Calculate Root Mean Squared Error (RMSE) for each column
        rmse = torch.sqrt(mse)

        # Filter specific columns if indices are provided
        if self.scored_indices is not None:
            rmse = rmse[self.scored_indices]

        # Return the mean of the column-wise RMSEs
        return torch.mean(rmse)
