import os
import random
import numpy as np
import torch
from sklearn import metrics


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms can slow down training, but ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
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


def alaska_weighted_auc(y_true, y_valid):
    """
    Calculates the weighted AUC metric for the ALASKA2 competition.

    The metric weights the area based on the True Positive Rate (TPR) regions:
    - TPR [0.0, 0.4]: Weight 2
    - TPR [0.4, 1.0]: Weight 1

    The calculation integrates (1 - FPR) with respect to TPR.

    Args:
        y_true (np.array): Ground truth binary labels.
        y_valid (np.array): Predicted probabilities or scores.

    Returns:
        float: The normalized weighted AUC score.
    """
    # Calculate basic ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)

    # Parameters defined in the task
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    # We integrate (1 - fpr) with respect to tpr.
    # Since roc_curve returns sorted arrays (by fpr usually, but tpr is monotonic),
    # we iterate through the segments.

    area_total = 0.0

    # Iterate through all segments of the ROC curve
    for i in range(len(tpr) - 1):
        tpr_i = tpr[i]
        tpr_next = tpr[i + 1]
        fpr_i = fpr[i]
        fpr_next = fpr[i + 1]

        # Skip if no vertical progress
        if tpr_i == tpr_next:
            continue

        # Define the current segment's TPR range
        tpr_min = tpr_i
        tpr_max = tpr_next

        # Calculate average FPR for this segment (Trapezoidal rule approximation)
        # We are calculating area to the left of the curve (FPR vs TPR),
        # then 1 - that area gives us area under (1-FPR).
        # Contribution = (tpr_next - tpr_i) * (1 - (fpr_next + fpr_i) / 2)

        # However, we need to apply weights based on TPR region.
        # We check if the segment crosses the boundary 0.4.
        boundary = 0.4

        # Case 1: Entirely in the first region [0, 0.4]
        if tpr_max <= boundary:
            w = weights[0]
            dy = tpr_max - tpr_min
            avg_fpr = (fpr_next + fpr_i) / 2.0
            area_total += w * dy * (1.0 - avg_fpr)

        # Case 2: Entirely in the second region [0.4, 1.0]
        elif tpr_min >= boundary:
            w = weights[1]
            dy = tpr_max - tpr_min
            avg_fpr = (fpr_next + fpr_i) / 2.0
            area_total += w * dy * (1.0 - avg_fpr)

        # Case 3: Crosses the boundary
        else:
            # We need to split the segment at TPR = 0.4
            # Interpolate FPR at TPR = 0.4
            # Linear interpolation: f(y) = f_0 + (f_1 - f_0) * (y - y_0) / (y_1 - y_0)
            fpr_at_boundary = fpr_i + (fpr_next - fpr_i) * (boundary - tpr_min) / (
                tpr_max - tpr_min
            )

            # Lower part (Weight 2)
            w_lower = weights[0]
            dy_lower = boundary - tpr_min
            avg_fpr_lower = (fpr_at_boundary + fpr_i) / 2.0
            area_total += w_lower * dy_lower * (1.0 - avg_fpr_lower)

            # Upper part (Weight 1)
            w_upper = weights[1]
            dy_upper = tpr_max - boundary
            avg_fpr_upper = (fpr_next + fpr_at_boundary) / 2.0
            area_total += w_upper * dy_upper * (1.0 - avg_fpr_upper)

    # Normalize by the maximum possible weighted area
    # Max area occurs when FPR is always 0.
    # Area = Sum(weight_i * length_of_region_i)
    # Region 1: length 0.4, weight 2 -> 0.8
    # Region 2: length 0.6, weight 1 -> 0.6
    # Total max area = 1.4
    max_area = weights[0] * (tpr_thresholds[1] - tpr_thresholds[0]) + weights[1] * (
        tpr_thresholds[2] - tpr_thresholds[1]
    )

    return area_total / max_area
