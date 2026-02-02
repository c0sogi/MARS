import os
import random
import numpy as np
import torch
from collections import defaultdict
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

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricMonitor:
    """
    A helper class to track and average metrics (loss, accuracy, AUC, etc.)
    during training and validation.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets the internal metric storage.
        """
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        """
        Updates the metric with a new value.

        Args:
            metric_name (str): The name of the metric (e.g., 'Loss', 'AUC').
            val (float): The value to add.
        """
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        """
        Returns a string representation of the averaged metrics.
        Prints full precision without rounding as requested.
        """
        return " | ".join(
            [
                "{}: {}".format(metric_name, metric["avg"])
                for metric_name, metric in self.metrics.items()
            ]
        )


def get_checkpoint_path(model_name):
    """
    Generates the path for saving the model checkpoint and ensures the
    directory exists.

    Args:
        model_name (str): The name of the model (e.g., 'efficientnet_b0').

    Returns:
        str: The full path to the checkpoint file.
    """
    save_dir = Config.WORKING_DIR
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, f"{model_name}_best.pth")
