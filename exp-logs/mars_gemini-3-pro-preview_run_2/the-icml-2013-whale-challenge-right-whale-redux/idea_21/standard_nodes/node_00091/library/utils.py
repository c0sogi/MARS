import os
import random
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (AUC).

    Args:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Predicted probabilities.

    Returns:
        float: The ROC AUC score.
    """
    return roc_auc_score(y_true, y_pred)


class MetricMonitor:
    """
    A utility class to track and average metrics (e.g., Loss, Accuracy)
    over a sequence of updates (batches).
    """

    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        """Resets the internal metric storage."""
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        """
        Updates the running average for a specific metric.

        Args:
            metric_name (str): The name of the metric.
            val (float): The value to add.
        """
        metric = self.metrics[metric_name]
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        """Returns a formatted string of the current average metrics."""
        return " | ".join(
            [
                "{}: {:.{prec}f}".format(
                    metric_name, metric["avg"], prec=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )

    @property
    def avg_metrics(self):
        """Returns a dictionary of the current average values for all metrics."""
        return {name: metric["avg"] for name, metric in self.metrics.items()}


def save_checkpoint(model, optimizer, epoch, score, filename):
    """
    Saves a model checkpoint to the configured working directory.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): The current training epoch.
        score (float): The validation score (AUC) associated with this checkpoint.
        filename (str): The name of the file (e.g., 'model_fold_0.pth').
    """
    # Ensure the directory exists (Config.WORKING_DIR is created by Config.setup(),
    # but this handles cases where filename includes a subdirectory)
    save_path = os.path.join(Config.WORKING_DIR, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "score": score,
    }

    torch.save(state, save_path)
