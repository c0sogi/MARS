import os
import random
import copy
import numpy as np
import torch
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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs. Saves the best model state using deepcopy.
    """

    def __init__(self, patience=7, mode="min", delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the
                quantity monitored has stopped decreasing; in 'max' mode it will stop when
                the quantity monitored has stopped increasing.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.val_score = None

    def __call__(self, metric, model):
        """
        Call method to update the early stopping status.

        Args:
            metric (float): The current validation metric (e.g., validation loss).
            model (torch.nn.Module): The model to save if the metric improves.
        """
        score = metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        else:
            if self.mode == "min":
                improvement = score < (self.best_score - self.delta)
            elif self.mode == "max":
                improvement = score > (self.best_score + self.delta)
            else:
                # Default to min behavior if mode is unrecognized
                improvement = score < (self.best_score - self.delta)

            if improvement:
                self.best_score = score
                self.save_checkpoint(score, model)
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, score, model):
        """
        Saves the model state using deepcopy.
        """
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.val_score = score

    def restore_best_weights(self, model):
        """
        Restores the best model weights from the saved state.
        """
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
