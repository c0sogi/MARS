import os
import random
import copy
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

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


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Stores the best model state in memory using deepcopy to avoid reference mutation.
    """

    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            trace_func (function): trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.trace_func = trace_func
        self.best_model_state = None

    def __call__(self, val_loss, model):
        """
        Call method to be executed after each epoch.

        Args:
            val_loss (float): The validation loss of the current epoch.
            model (torch.nn.Module): The model being trained.
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
        Saves model state when validation loss decreases.
        Uses deepcopy to strictly preserve the best weights in memory.
        """
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min} --> {val_loss}). Saving model state..."
            )

        # Deep copy the state dict to avoid reference mutation bugs
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss

    def load_best_weights(self, model):
        """
        Loads the best weights stored in memory into the provided model.

        Args:
            model (torch.nn.Module): The model to load weights into.
        """
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                self.trace_func("Restored best model weights from memory.")
        else:
            if self.verbose:
                self.trace_func("No best model state captured yet.")
