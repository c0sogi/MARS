import os
import torch
import numpy as np
import copy
from library.config import set_seed, DEVICE


def seed_everything(seed=42):
    """
    Enforces determinism across PyTorch, NumPy, and Python random generators.
    Wraps the set_seed function from library.config to ensure consistency.
    """
    set_seed(seed)


def save_checkpoint(model, path):
    """
    Saves the model state dictionary to the specified path.
    Creates the parent directory if it does not exist.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path to save the checkpoint to.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device=DEVICE):
    """
    Loads the model state dictionary from the specified path.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
        path (str): The file path of the checkpoint.
        device (torch.device): The device to map the location to.

    Returns:
        model (torch.nn.Module): The model with loaded weights.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model


class EarlyStopping:
    """
    Early stopping to stop the training when the monitored metric does not improve
    after a certain number of epochs.
    """

    def __init__(self, patience=7, delta=0, mode="min", verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            mode (str): One of {'min', 'max'}.
                        'min' for metrics like Loss (lower is better).
                        'max' for metrics like Accuracy (higher is better).
            verbose (bool): If True, prints a message for each validation improvement.
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

        if self.mode == "min":
            self.monitor_op = np.less
            self.best_score = np.inf
        elif self.mode == "max":
            self.monitor_op = np.greater
            self.best_score = -np.inf
        else:
            raise ValueError(
                f"EarlyStopping mode {mode} is unknown! Use 'min' or 'max'."
            )

    def __call__(self, metric, model, path):
        """
        Updates the early stopping state based on the current metric.

        Args:
            metric (float): The current validation metric (e.g., val_loss).
            model (torch.nn.Module): The model to save if metric improves.
            path (str): Path to save the best model checkpoint.
        """
        score = metric

        # Check if this is the first epoch or if the score improved
        improved = False
        if self.mode == "min":
            if score < self.best_score - self.delta:
                improved = True
        else:  # mode == 'max'
            if score > self.best_score + self.delta:
                improved = True

        if improved:
            if self.verbose:
                print(
                    f"Validation metric improved ({self.best_score} --> {score}). Saving model..."
                )
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(model, path)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
