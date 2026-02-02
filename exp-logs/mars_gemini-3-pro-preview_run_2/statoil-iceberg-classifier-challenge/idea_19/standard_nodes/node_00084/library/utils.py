import os
import random
import numpy as np
import torch
import copy


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model state in memory using copy.deepcopy.
    """

    def __init__(self, patience=10, min_delta=0, mode="min"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            mode (str): One of {'min', 'max'}. 'min' for loss, 'max' for accuracy/score.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

        # Configure comparison logic based on mode
        if mode == "min":
            self.monitor_op = np.less
            self.delta_sign = -1
        elif mode == "max":
            self.monitor_op = np.greater
            self.delta_sign = 1
        else:
            raise ValueError(
                f"EarlyStopping mode '{mode}' is unknown. Use 'min' or 'max'."
            )

    def __call__(self, metric, model):
        """
        Updates the internal state based on the current validation metric.

        Args:
            metric (float): The current validation metric (e.g., validation loss).
            model (torch.nn.Module): The model to save if the metric improves.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = metric
            self.save_checkpoint(model)
        else:
            # Check if metric improved by at least min_delta
            # For 'min': current < best - delta
            # For 'max': current > best + delta
            target = self.best_score + (self.delta_sign * self.min_delta)

            if self.monitor_op(metric, target):
                # Improved
                self.best_score = metric
                self.save_checkpoint(model)
                self.counter = 0
            else:
                # Not improved
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model):
        """Saves the best model state to memory."""
        self.best_model_state = copy.deepcopy(model.state_dict())
