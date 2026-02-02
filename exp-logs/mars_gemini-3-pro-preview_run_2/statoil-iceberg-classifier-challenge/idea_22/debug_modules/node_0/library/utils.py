import os
import copy
import torch
import numpy as np
from library.config import Config


def seed_everything(seed=None):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to the Config class to ensure consistency.

    Args:
        seed (int, optional): The seed value to set. If None, uses Config.SEED.
    """
    Config.set_seed(seed)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model state to memory (via deepcopy) and disk.
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
            trace_func (function): Trace print function.
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
        self.best_model_state = None

    def __call__(self, val_loss, model):
        """
        Call method to be invoked at the end of each epoch.

        Args:
            val_loss (float): The validation loss metric to monitor.
            model (torch.nn.Module): The model to save.
        """
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
        """
        Saves model when validation loss decreases.

        Args:
            val_loss (float): Current validation loss.
            model (torch.nn.Module): Current model.
        """
        if self.verbose:
            # Printing full precision as requested
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min} --> {val_loss}).  Saving model ..."
            )

        # Strictly preserve the best weights using deepcopy
        self.best_model_state = copy.deepcopy(model.state_dict())

        # Ensure the directory exists
        directory = os.path.dirname(self.path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Save to disk
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
