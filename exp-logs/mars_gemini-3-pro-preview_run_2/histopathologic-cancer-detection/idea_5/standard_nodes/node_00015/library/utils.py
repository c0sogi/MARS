import os
import random
import shutil
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_auc(y_true, y_scores):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true (np.array): Ground truth binary labels.
        y_scores (np.array): Predicted probabilities for the positive class.

    Returns:
        float: The AUC score. Returns 0.5 if only one class is present in y_true.
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Check if we have both classes
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_scores)


def save_checkpoint(state, is_best, fold):
    """
    Saves the model checkpoint.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        fold (int): The current fold number.
    """
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Save regular checkpoint
    filename = os.path.join(Config.CHECKPOINT_DIR, f"checkpoint_fold_{fold}.pth")
    torch.save(state, filename)

    # Save best model copy if applicable
    if is_best:
        best_filename = os.path.join(
            Config.CHECKPOINT_DIR, f"best_model_fold_{fold}.pth"
        )
        shutil.copyfile(filename, best_filename)


class MetricMonitor:
    """
    A utility class to track and average metrics (loss, accuracy, etc.) over an epoch.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all tracked metrics."""
        self.val = {}
        self.avg = {}
        self.sum = {}
        self.count = {}

    def update(self, metric_name, val, n=1):
        """
        Updates a specific metric.

        Args:
            metric_name (str): Name of the metric (e.g., 'Loss').
            val (float): The current value of the metric.
            n (int): The number of samples associated with this value (usually batch size).
        """
        val = float(val)  # Ensure float
        if metric_name not in self.val:
            self.val[metric_name] = val
            self.sum[metric_name] = val * n
            self.count[metric_name] = n
            self.avg[metric_name] = val
        else:
            self.val[metric_name] = val
            self.sum[metric_name] += val * n
            self.count[metric_name] += n
            self.avg[metric_name] = self.sum[metric_name] / self.count[metric_name]

    def get_avg(self, metric_name):
        """Returns the running average of a metric."""
        return self.avg.get(metric_name, 0.0)

    def __str__(self):
        """Returns a string representation of the averages for logging."""
        return " | ".join(
            [
                "{}: {:.6f}".format(metric_name, self.avg[metric_name])
                for metric_name in self.avg
            ]
        )


class EarlyStopping:
    """
    Implements early stopping logic to terminate training when validation metric stops improving.
    """

    def __init__(self, patience=5, mode="max", min_delta=0.0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation metric improved.
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the
                        quantity monitored has stopped decreasing; in 'max' mode it will
                        stop when the quantity monitored has stopped increasing.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            raise ValueError(f"EarlyStopping mode {mode} is unknown!")

    def __call__(self, current_score):
        """
        Updates the internal state based on the current score.

        Args:
            current_score (float): The metric value to monitor (e.g., Val Loss or Val AUC).
        """
        if self.best_score is None:
            self.best_score = current_score
        else:
            # Check if current score is better than best score by at least min_delta
            if self.mode == "max":
                improved = current_score > (self.best_score + self.min_delta)
            else:
                improved = current_score < (self.best_score - self.min_delta)

            if improved:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
