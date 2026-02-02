import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_curve
from library.config import Config


def seed_everything(seed=Config.seed):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
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


def alaska_weighted_auc(y_true, y_score):
    """
    Calculates the weighted AUC metric as defined in the ALASKA2 competition.

    The area under the ROC curve is calculated via integration over the TPR axis.
    Different regions of the TPR axis are weighted differently.

    Args:
        y_true (np.array): Ground truth binary labels (0 or 1).
        y_score (np.array): Predicted probabilities or logits for the positive class (1).

    Returns:
        float: The weighted AUC score, normalized between 0 and 1.
    """
    # Retrieve parameters from Config
    tpr_thresholds = Config.tpr_thresholds
    weights = Config.auc_weights

    # Generate ROC curve points
    # roc_curve returns fpr, tpr, and thresholds sorted by decreasing score (increasing FPR/TPR)
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)

    # Calculate the normalization factor (Max possible weighted area)
    # This corresponds to a perfect classifier where FPR=0 for all TPR > 0.
    # Area contribution is (1 - 0) * weight * height = weight * height.
    normalization = 0.0
    for i, weight in enumerate(weights):
        height = tpr_thresholds[i + 1] - tpr_thresholds[i]
        normalization += weight * height

    weighted_area = 0.0

    # Iterate through ROC segments
    # Since we integrate over TPR (y-axis), we look at vertical segments
    for i in range(len(tpr) - 1):
        t1 = tpr[i]
        t2 = tpr[i + 1]
        f1 = fpr[i]
        f2 = fpr[i + 1]

        # Skip if no vertical progress (TPR didn't change)
        if t2 == t1:
            continue

        # For each defined weight region, calculate the overlap with the current ROC segment
        for j, weight in enumerate(weights):
            region_low = tpr_thresholds[j]
            region_high = tpr_thresholds[j + 1]

            # Determine intersection of [t1, t2] and [region_low, region_high]
            overlap_low = max(t1, region_low)
            overlap_high = min(t2, region_high)

            if overlap_high > overlap_low:
                dh = overlap_high - overlap_low

                # Linear interpolation to find FPR at the midpoint of the overlap
                # f(t) = f1 + (f2 - f1) * (t - t1) / (t2 - t1)
                t_mid = (overlap_low + overlap_high) / 2.0
                f_mid = f1 + (f2 - f1) * (t_mid - t1) / (t2 - t1)

                # Calculate area to the right of the curve (1 - FPR) for this strip
                # Area = Width * Height = (1 - FPR_mid) * dh
                segment_area = (1 - f_mid) * dh

                # Add weighted contribution
                weighted_area += weight * segment_area

    return weighted_area / normalization
