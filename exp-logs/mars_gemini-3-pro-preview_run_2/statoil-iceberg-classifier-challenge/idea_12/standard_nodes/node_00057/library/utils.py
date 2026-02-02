import os
import random
import copy
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the log loss metric.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted probabilities for class 1.

    Returns:
        float: The log loss value.
    """
    # Sklearn log_loss expects probabilities.
    # We define labels=[0, 1] to ensure correct calculation even if a batch
    # is missing one of the classes.
    return log_loss(y_true, y_pred, labels=[0, 1])


class EarlyStopping:
    """
    Early stopping to stop the training when the monitored metric does not improve
    after a certain number of epochs. Stores the best model state in memory using deepcopy.
    """

    def __init__(self, patience=Config.PATIENCE, min_delta=0.0, mode="min"):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
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

        if mode not in ["min", "max"]:
            raise ValueError(
                f"EarlyStopping mode '{mode}' is unknown, please use 'min' or 'max'."
            )

    def __call__(self, current_score, model):
        """
        Updates the internal state based on the current score.

        Args:
            current_score (float): The metric value for the current epoch.
            model (torch.nn.Module): The model to save if the score improves.
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_model_state = copy.deepcopy(model.state_dict())
            return

        improvement = False
        if self.mode == "min":
            if current_score < (self.best_score - self.min_delta):
                improvement = True
        else:  # mode == 'max'
            if current_score > (self.best_score + self.min_delta):
                improvement = True

        if improvement:
            self.best_score = current_score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_weights(self, model):
        """
        Restores the best model weights to the provided model instance.

        Args:
            model (torch.nn.Module): The model to load weights into.

        Returns:
            model: The model with restored weights.
        """
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
        return model
