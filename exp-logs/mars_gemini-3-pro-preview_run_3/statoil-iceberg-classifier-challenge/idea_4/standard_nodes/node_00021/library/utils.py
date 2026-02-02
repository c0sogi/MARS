import os
import sys
import torch
import numpy as np
from sklearn.metrics import log_loss
from library.config import seed_everything


def set_seed(seed):
    """
    Sets the random seed for reproducibility using the function from config.

    Args:
        seed (int): The seed value to use.
    """
    seed_everything(seed)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the model checkpoint when validation loss decreases.
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
        self.val_loss_min = np.Inf
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
            # Print full precision as required
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min} --> {val_loss}).  Saving model ..."
            )

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def calculate_metric(y_true, y_pred):
    """
    Calculates the Log Loss metric.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted probabilities.

    Returns:
        float: Log loss value.
    """
    # sklearn log_loss handles the log(0) case internally with an epsilon
    return log_loss(y_true, y_pred, labels=[0, 1])


def save_checkpoint(state, is_best, filepath="checkpoint.pth"):
    """
    Saves a checkpoint of the model state.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filepath (str): Path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)

    if is_best:
        dirname = os.path.dirname(filepath)
        # Assuming standard naming convention, we save a copy as model_best.pth
        best_filepath = os.path.join(dirname, "model_best.pth")
        torch.save(state, best_filepath)
