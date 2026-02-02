import os
import random
import numpy as np
import torch
from sklearn import metrics


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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


def weighted_auc(y_true, y_pred):
    """
    Calculates the weighted AUC metric.

    The area between the true positive rate of 0 and 0.4 is weighted 2X.
    The area between 0.4 and 1 is weighted 1X.
    The total area is normalized by the sum of weights.
    """
    # Convert tensors to numpy if necessary
    if hasattr(y_true, "cpu"):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "cpu"):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check for single class case to prevent errors
    if len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)

    # Weights and Thresholds
    t_threshold = 0.4
    w_low = 2
    w_high = 1

    area_low = 0.0
    area_high = 0.0

    # Integrate using trapezoidal rule, splitting by TPR threshold
    for i in range(len(fpr) - 1):
        dx = fpr[i + 1] - fpr[i]
        if dx == 0:
            continue

        y1 = tpr[i]
        y2 = tpr[i + 1]

        # Determine overlap with low region [0, 0.4] and high region [0.4, 1.0]

        # Case 1: Segment entirely in low region (TPR <= 0.4)
        if y2 <= t_threshold:
            avg_y = (y1 + y2) / 2.0
            area_low += avg_y * dx

        # Case 2: Segment entirely in high region (TPR >= 0.4)
        elif y1 >= t_threshold:
            # The rectangular base below 0.4 counts towards area_low
            area_low += t_threshold * dx

            # The part above 0.4 counts towards area_high
            avg_y = (y1 + y2) / 2.0
            area_high += (avg_y - t_threshold) * dx

        # Case 3: Segment crosses threshold (y1 < 0.4 < y2)
        else:
            # Find intersection point where TPR = 0.4
            slope = (y2 - y1) / dx
            dx_low = (t_threshold - y1) / slope
            dx_high = dx - dx_low

            # Part 1: y1 to 0.4 (Trapezoid entirely in low region)
            avg_y_low = (y1 + t_threshold) / 2.0
            area_low += avg_y_low * dx_low

            # Part 2: 0.4 to y2
            # Base rectangle of height 0.4 counts to area_low
            area_low += t_threshold * dx_high

            # Top triangle/trapezoid (from 0.4 to y2) counts to area_high
            avg_y_high = (t_threshold + y2) / 2.0
            area_high += (avg_y_high - t_threshold) * dx_high

    # Normalization
    # Max possible area_low is when TPR=1 everywhere -> rectangle 1.0 * 0.4 = 0.4
    # Max possible area_high is when TPR=1 everywhere -> rectangle 1.0 * 0.6 = 0.6
    norm_factor = w_low * 0.4 + w_high * 0.6

    weighted_score = w_low * area_low + w_high * area_high

    return weighted_score / norm_factor
