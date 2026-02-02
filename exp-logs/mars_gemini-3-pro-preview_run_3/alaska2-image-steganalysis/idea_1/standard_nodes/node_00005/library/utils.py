import os
import random
import numpy as np
import torch
from sklearn import metrics
from library.config import TPR_THRESHOLDS, AUC_WEIGHTS


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def weighted_auc_score(
    y_true, y_score, tpr_thresholds=TPR_THRESHOLDS, weights=AUC_WEIGHTS
):
    """
    Calculates the Weighted AUC score as defined in the task.

    The area under the ROC curve is divided into horizontal strips based on TPR thresholds.
    Each strip is weighted differently.

    Args:
        y_true (array-like): Ground truth binary labels (0 or 1).
        y_score (array-like): Predicted probabilities or scores.
        tpr_thresholds (list): List of TPR thresholds defining the regions (e.g., [0.0, 0.4, 1.0]).
        weights (list): List of weights for each region defined by the thresholds.

    Returns:
        float: The normalized weighted AUC score between 0 and 1.
    """
    # Compute the standard ROC curve
    # pos_label=1 ensures we are tracking the 'Stego' class
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)

    area_total = 0
    max_area_total = 0

    # Iterate through the regions defined by TPR thresholds
    # If thresholds are [0.0, 0.4, 1.0], we have two regions: [0.0, 0.4] and [0.4, 1.0]
    for i in range(len(weights)):
        low = tpr_thresholds[i]
        high = tpr_thresholds[i + 1]
        w = weights[i]

        # We want to calculate the area contribution of the TPR curve within [low, high].
        # We mathematically isolate this strip by clipping the TPR curve.
        # np.clip(tpr, low, high) constrains the curve to the strip.
        # Subtracting 'low' shifts the bottom of the strip to 0, effectively creating
        # a partial curve representing the height accumulated in this specific region.
        tpr_subset = np.clip(tpr, low, high) - low

        # Calculate the area under this partial curve using the trapezoidal rule
        # The x-axis is FPR (which ranges from 0 to 1)
        area = metrics.auc(fpr, tpr_subset)

        # Add weighted area
        area_total += area * w

        # Calculate the maximum possible area for this region.
        # In a perfect classifier, TPR is 1.0 for all FPR > 0.
        # The max height of this strip is (high - low).
        # The width of the domain (FPR) is 1.0.
        max_area = high - low
        max_area_total += max_area * w

    # Normalize the total weighted area
    if max_area_total == 0:
        return 0.0

    return area_total / max_area_total
