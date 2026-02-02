import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_curve
from library.config import Config


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
        # Ensure deterministic behavior for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def weighted_auc_score(y_true, y_score, tpr_thresholds=None, weights=None):
    """
    Calculates the Weighted AUC score based on specific TPR thresholds.
    The area under the ROC curve is divided into regions based on TPR,
    and each region is weighted differently.

    Args:
        y_true: Array-like of ground truth labels.
        y_score: Array-like of predicted probabilities or logits.
        tpr_thresholds: List of TPR thresholds defining the regions.
                        Default uses Config.tpr_thresholds (e.g., [0.0, 0.4, 1.0]).
        weights: List of weights for each region.
                 Default uses Config.weights (e.g., [2, 1]).

    Returns:
        float: The weighted AUC score.
    """
    if tpr_thresholds is None:
        tpr_thresholds = Config.tpr_thresholds
    if weights is None:
        weights = Config.weights

    # Compute ROC curve
    # roc_curve returns fpr and tpr sorted by thresholds descending (i.e., fpr and tpr ascending)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    # Ensure strictly sorted by TPR for consistent processing, though roc_curve usually guarantees this
    if not np.all(np.diff(tpr) >= 0):
        sorted_indices = np.argsort(tpr)
        tpr = tpr[sorted_indices]
        fpr = fpr[sorted_indices]

    # Insert threshold points into the ROC curve if they don't exist
    # This ensures exact integration boundaries
    for threshold in tpr_thresholds:
        if threshold <= 0 or threshold >= 1:
            continue

        # Find where to insert
        # searchsorted returns the index where threshold should be inserted to maintain order
        idx = np.searchsorted(tpr, threshold)

        # If the threshold is not already present (check previous or current index)
        # Note: floating point comparison requires tolerance, but exact match check is safe here
        # because we only interpolate if we are strictly between points
        if idx < len(tpr) and tpr[idx] == threshold:
            continue

        # Interpolate
        tpr_prev, tpr_next = tpr[idx - 1], tpr[idx]
        fpr_prev, fpr_next = fpr[idx - 1], fpr[idx]

        # Linear interpolation of FPR at the TPR threshold
        # (y - y0) / (x - x0) = (y1 - y0) / (x1 - x0) -> y = y0 + slope * (x - x0)
        # Here x is TPR, y is FPR
        slope = (fpr_next - fpr_prev) / (tpr_next - tpr_prev)
        fpr_interp = fpr_prev + slope * (threshold - tpr_prev)

        # Insert into arrays
        tpr = np.insert(tpr, idx, threshold)
        fpr = np.insert(fpr, idx, fpr_interp)

    score_numerator = 0.0
    score_denominator = 0.0

    # Iterate over the defined regions
    for i in range(len(weights)):
        t_start = tpr_thresholds[i]
        t_end = tpr_thresholds[i + 1]
        w = weights[i]

        # Select points belonging to this region [t_start, t_end]
        # We use >= and <= to include boundary points in the integration
        mask = (tpr >= t_start) & (tpr <= t_end)

        tpr_region = tpr[mask]
        fpr_region = fpr[mask]

        # Calculate the area under the FPR curve vs TPR for this region
        # This corresponds to Integral(FPR(t) dt) from t_start to t_end
        area_fpr = np.trapz(fpr_region, tpr_region)

        # The partial AUC (pAUC) for this region is the area to the left of the ROC curve
        # if plotted as TPR vs FPR.
        # pAUC = Integral((1 - FPR(t)) dt) = (t_end - t_start) - Integral(FPR(t) dt)
        width = t_end - t_start
        p_auc = width - area_fpr

        score_numerator += w * p_auc
        score_denominator += w * width

    if score_denominator == 0:
        return 0.0

    return score_numerator / score_denominator
