import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

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
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    This metric computes the RMSE for each target column separately and then
    returns the average of these RMSEs. It accounts for the fact that only
    the first `num_scored` positions of the RNA sequence are experimentally
    validated and scored.
    """

    def __init__(self, num_scored=None):
        """
        Initialize the MCRMSELoss.

        Args:
            num_scored (int, optional): The number of positions at the start of the
                                        sequence to include in the loss calculation.
                                        Defaults to Config.pred_len (68).
        """
        super().__init__()
        if num_scored is None:
            try:
                config = Config()
                self.num_scored = config.pred_len
            except Exception:
                # Fallback if Config is not accessible or initialized
                self.num_scored = 68
        else:
            self.num_scored = num_scored

    def forward(self, preds, targets):
        """
        Calculate the MCRMSE loss.

        Args:
            preds (torch.Tensor): Predictions of shape (Batch, Seq_Len, Num_Targets).
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len, Num_Targets).

        Returns:
            torch.Tensor: The scalar MCRMSE loss.
        """
        # Ensure inputs are floating point
        preds = preds.float()
        targets = targets.float()

        # Slice the tensors to include only the scored positions
        # Shape becomes: (Batch, num_scored, Num_Targets)
        preds_scored = preds[:, : self.num_scored, :]
        targets_scored = targets[:, : self.num_scored, :]

        # Calculate Squared Error: (y - y_hat)^2
        squared_error = (preds_scored - targets_scored) ** 2

        # Calculate Mean Squared Error (MSE) for each column
        # We average over the Batch (dim 0) and Sequence (dim 1) dimensions
        # resulting in a tensor of shape (Num_Targets,)
        mse_per_column = torch.mean(squared_error, dim=(0, 1))

        # Calculate Root Mean Squared Error (RMSE) for each column
        rmse_per_column = torch.sqrt(mse_per_column)

        # Calculate the Mean of the column RMSEs (MCRMSE)
        mcrmse = torch.mean(rmse_per_column)

        return mcrmse
