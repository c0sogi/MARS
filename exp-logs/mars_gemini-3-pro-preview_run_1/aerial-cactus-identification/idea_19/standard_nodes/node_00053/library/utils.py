import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.optim.swa_utils import AveragedModel
import warnings

# Suppress warnings to keep output clean
warnings.filterwarnings("ignore")


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricMonitor:
    """
    A utility class to track and average metrics over an epoch.
    """

    def __init__(self, float_precision=4):
        self.reset()
        self.float_precision = float_precision

    def reset(self):
        self.metrics = {}

    def update(self, metric_name, val, n=1):
        """
        Update the metric with a new value.
        Args:
            metric_name (str): Name of the metric.
            val (float or torch.Tensor): Value of the metric.
            n (int): Number of samples associated with this value (usually batch size).
        """
        if isinstance(val, torch.Tensor):
            val = val.item()

        if metric_name not in self.metrics:
            self.metrics[metric_name] = {"sum": 0.0, "count": 0}

        self.metrics[metric_name]["sum"] += val * n
        self.metrics[metric_name]["count"] += n

    def get_avg(self, metric_name):
        """
        Returns the average value of the metric.
        """
        if metric_name not in self.metrics:
            return 0.0
        return self.metrics[metric_name]["sum"] / self.metrics[metric_name]["count"]

    def __str__(self):
        """
        Returns a string representation of all metrics with high precision.
        """
        return " | ".join(
            [
                "{}: {:.16f}".format(metric_name, self.get_avg(metric_name))
                for metric_name in self.metrics
            ]
        )


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve.
    Handles tensor inputs and edge cases (e.g., single class in batch).
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Handle NaN or Inf in predictions
    if np.isnan(y_pred).any() or np.isinf(y_pred).any():
        return 0.5

    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        # Occurs if y_true has only one class
        return 0.5


class SWAHandler:
    """
    Manages Stochastic Weight Averaging (SWA) updates and BatchNorm statistics.
    """

    def __init__(self, model, swa_start_epoch, device=None):
        self.swa_model = AveragedModel(model)
        self.swa_start_epoch = swa_start_epoch
        self.device = device
        if self.device:
            self.swa_model.to(self.device)

    def update(self, model, epoch):
        """
        Updates the SWA model parameters if the current epoch is >= start epoch.
        """
        if epoch >= self.swa_start_epoch:
            self.swa_model.update_parameters(model)

    def update_bn(self, loader):
        """
        Updates BatchNorm statistics for the SWA model.
        Implements a custom loop to handle multi-input models (e.g., image + metadata)
        which standard torch utilities do not support out-of-the-box.
        """
        self.swa_model.train()

        # Reset BN running stats
        for module in self.swa_model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                module.momentum = None

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    # Move all elements to device
                    batch = [
                        b.to(self.device) if hasattr(b, "to") else b for b in batch
                    ]

                    # Handle specific batch structures
                    if len(batch) == 3:
                        # Assumes (image, label, metadata) -> Model expects (image, metadata)
                        img, _, meta = batch
                        self.swa_model(img, meta)
                    elif len(batch) == 2:
                        # Assumes (image, label) -> Model expects (image)
                        img, _ = batch
                        self.swa_model(img)
                    else:
                        # Fallback
                        self.swa_model(batch[0])
                else:
                    batch = batch.to(self.device)
                    self.swa_model(batch)
