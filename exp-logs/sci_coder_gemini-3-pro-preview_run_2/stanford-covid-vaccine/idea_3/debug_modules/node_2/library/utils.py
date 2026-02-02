import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
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


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error Loss.
    Calculates the average RMSE across target columns, considering only the
    scored sequence positions.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        self.seq_scored = Config.PRED_LEN

    def forward(self, preds, targets):
        """
        Args:
            preds: Tensor of shape (Batch, Seq_Len, Num_Targets)
            targets: Tensor of shape (Batch, Seq_Len, Num_Targets)
        Returns:
            mcrmse: Scalar tensor representing the loss.
        """
        # Slice to consider only the scored positions (first 68)
        preds_scored = preds[:, : self.seq_scored, :]
        targets_scored = targets[:, : self.seq_scored, :]

        # Calculate MSE for each column (averaging over batch and sequence length)
        # dim=(0, 1) aggregates over Batch and Seq_Len dimensions
        mse = torch.mean((preds_scored - targets_scored) ** 2, dim=(0, 1))

        # Calculate RMSE for each column
        rmse = torch.sqrt(mse)

        # Calculate Mean of RMSEs across columns
        mcrmse = torch.mean(rmse)

        return mcrmse


def compute_mcrmse_numpy(preds, targets, scored_indices=None):
    """
    Calculates MCRMSE using Numpy arrays, useful for validation scoring.

    Args:
        preds: Numpy array (Batch, Seq_Len, Num_Targets)
        targets: Numpy array (Batch, Seq_Len, Num_Targets)
        scored_indices: List of integers representing column indices to score.
                        If None, scores all columns.
    """
    seq_scored = Config.PRED_LEN

    # Slice sequence length
    p = preds[:, :seq_scored, :]
    t = targets[:, :seq_scored, :]

    # Filter columns if indices provided
    if scored_indices is not None:
        p = p[:, :, scored_indices]
        t = t[:, :, scored_indices]

    # Calculate MSE per column
    # Average over batch (axis 0) and sequence (axis 1)
    mse = np.mean((p - t) ** 2, axis=(0, 1))

    # RMSE per column
    rmse = np.sqrt(mse)

    # Mean of RMSEs
    return np.mean(rmse)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pth",
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min} --> {val_loss}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
