import os
import random
import copy
import numpy as np
import torch
from library.config import SEED, CUDNN_DETERMINISTIC, CUDNN_BENCHMARK


def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across various libraries.
    Configures CuDNN for performance as per the task requirements.

    Args:
        seed (int): The random seed to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Configure CuDNN based on the specific strategy (Lesson 00070)
    # We disable strict determinism to maximize kernel performance on the A100
    torch.backends.cudnn.deterministic = CUDNN_DETERMINISTIC
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK


class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    Caches the best model state in memory using deepcopy.
    """

    def __init__(self, patience=7, mode="max", delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
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

        if self.mode == "min":
            self.val_score_fn = lambda x: -x
        else:
            self.val_score_fn = lambda x: x

    def __call__(self, metric, model):
        """
        Updates the internal state based on the current metric.

        Args:
            metric (float): The current validation metric (e.g., accuracy or loss).
            model (torch.nn.Module): The model to snapshot if the metric improves.
        """
        score = self.val_score_fn(metric)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """Saves a deep copy of the model state dict to memory."""
        self.best_model_state = copy.deepcopy(model.state_dict())

    def restore_best_weights(self, model):
        """Restores the best cached weights to the provided model instance."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
        else:
            print("Warning: No best model state found to restore.")
