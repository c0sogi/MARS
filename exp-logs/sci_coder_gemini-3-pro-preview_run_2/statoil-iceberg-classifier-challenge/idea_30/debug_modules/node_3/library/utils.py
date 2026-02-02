import os
import random
import copy
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.

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
    Saves the best model state in memory using deepcopy.
    """

    def __init__(self, patience=7, mode="min", delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the
                        quantity monitored has stopped decreasing; in 'max' mode it will
                        stop when the quantity monitored has stopped increasing.
                        Default: 'min'
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

        if self.mode == "min":
            self.val_score_best = np.inf
        else:
            self.val_score_best = -np.inf

    def __call__(self, metric, model):
        """
        Update the state of early stopping based on the current metric.

        Args:
            metric (float): The current value of the validation metric (e.g., val_loss).
            model (torch.nn.Module): The model to save if the metric improves.
        """
        score = -metric if self.mode == "min" else metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(metric, model)
            self.counter = 0

    def save_checkpoint(self, metric, model):
        """
        Saves the model state dict via deepcopy.
        """
        self.val_score_best = metric
        self.best_model_state = copy.deepcopy(model.state_dict())
