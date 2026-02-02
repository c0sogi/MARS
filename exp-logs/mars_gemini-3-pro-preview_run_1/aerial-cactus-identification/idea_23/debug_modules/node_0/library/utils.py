import os
import random
import numpy as np
import torch
from collections import defaultdict


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    Configures CuDNN for deterministic execution.

    Args:
        seed (int): The random seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y_class, y_quality, alpha=0.2, device="cuda"):
    """
    Generates mixed inputs and target pairs for Mixup regularization.
    Applied to both classification labels and quality regression targets.

    Args:
        x (torch.Tensor): Input images.
        y_class (torch.Tensor): Classification targets.
        y_quality (torch.Tensor): Quality regression targets.
        alpha (float): Mixup alpha parameter.
        device (str): Device to perform computations on.

    Returns:
        mixed_x (torch.Tensor): Mixed images.
        y_class_a (torch.Tensor): Classification targets A.
        y_class_b (torch.Tensor): Classification targets B.
        y_qual_a (torch.Tensor): Quality targets A.
        y_qual_b (torch.Tensor): Quality targets B.
        lam (float): Mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_class_a, y_class_b = y_class, y_class[index]
    y_qual_a, y_qual_b = y_quality, y_quality[index]

    return mixed_x, y_class_a, y_class_b, y_qual_a, y_qual_b, lam


def mixup_criterion(
    criterion_class,
    criterion_qual,
    pred_class,
    pred_qual,
    y_class_a,
    y_class_b,
    y_qual_a,
    y_qual_b,
    lam,
    aux_weight,
):
    """
    Computes the weighted Mixup loss for Multi-Task Learning (Classification + Quality Regression).

    Args:
        criterion_class: Loss function for classification (e.g., BCEWithLogitsLoss).
        criterion_qual: Loss function for regression (e.g., MSELoss).
        pred_class (torch.Tensor): Predicted class logits.
        pred_qual (torch.Tensor): Predicted quality scores.
        y_class_a (torch.Tensor): Classification targets A.
        y_class_b (torch.Tensor): Classification targets B.
        y_qual_a (torch.Tensor): Quality targets A.
        y_qual_b (torch.Tensor): Quality targets B.
        lam (float): Mixing coefficient.
        aux_weight (float): Weight for the auxiliary quality loss.

    Returns:
        torch.Tensor: Combined weighted loss.
    """
    # Classification loss (mix of target A and B)
    loss_class = lam * criterion_class(pred_class, y_class_a) + (
        1 - lam
    ) * criterion_class(pred_class, y_class_b)

    # Quality regression loss (mix of target A and B)
    loss_qual = lam * criterion_qual(pred_qual, y_qual_a) + (1 - lam) * criterion_qual(
        pred_qual, y_qual_b
    )

    # Total weighted loss
    return loss_class + aux_weight * loss_qual


class MetricMonitor:
    """
    Tracks and averages metrics during training/validation.
    """

    def __init__(self, float_precision=None):
        """
        Args:
            float_precision (int, optional): Number of decimal places for string representation.
                                             If None, prints full precision.
        """
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.val = defaultdict(float)
        self.count = defaultdict(int)
        self.avg = defaultdict(float)

    def update(self, metric_name, val, n=1):
        """
        Updates the metric with a new value.

        Args:
            metric_name (str): Name of the metric.
            val (float): Value to add (e.g., batch loss).
            n (int): Weight of the value (e.g., batch size).
        """
        self.val[metric_name] += val * n
        self.count[metric_name] += n
        self.avg[metric_name] = self.val[metric_name] / self.count[metric_name]

    def __str__(self):
        results = []
        for name in self.avg:
            val = self.avg[name]
            if self.float_precision is None:
                res = "{}: {}".format(name, val)
            else:
                res = "{}: {:.{prec}f}".format(name, val, prec=self.float_precision)
            results.append(res)
        return " | ".join(results)
