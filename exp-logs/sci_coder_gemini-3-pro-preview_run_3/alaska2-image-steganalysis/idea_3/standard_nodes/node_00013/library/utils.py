import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_curve


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weighted_auc_score(y_true, y_score):
    """
    Calculates the weighted AUC score based on the competition metric.

    Weights:
    - TPR [0.0, 0.4]: Weight 2
    - TPR [0.4, 1.0]: Weight 1

    The score is normalized to be between 0 and 1.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # Check if there are at least two classes to calculate ROC
    if len(np.unique(y_true)) < 2:
        # Fallback for batches with single class, though ideally shouldn't happen in validation
        return 0.5

    # Compute ROC curve
    # fpr and tpr are increasing arrays
    fpr, tpr, _ = roc_curve(y_true, y_score)

    # Ensure the threshold 0.4 is present in the arrays for precise integration
    tpr_threshold = 0.4

    # Use searchsorted to find insertion point
    insert_idx = np.searchsorted(tpr, tpr_threshold)

    # If 0.4 is not exactly in the array, insert it
    # We check if the value at insert_idx is 0.4 (handling out of bounds)
    if insert_idx == len(tpr) or tpr[insert_idx] != tpr_threshold:
        # Interpolate FPR at TPR = 0.4
        fpr_interp = np.interp(tpr_threshold, tpr, fpr)

        # Insert into arrays
        tpr = np.insert(tpr, insert_idx, tpr_threshold)
        fpr = np.insert(fpr, insert_idx, fpr_interp)

    # Calculate the area under the curve of (1 - FPR) vs TPR
    # This is equivalent to calculating the area to the left of the ROC curve if axes were flipped
    # We use the trapezoidal rule: Area = sum( (y1+y2)/2 * dx )

    dt = np.diff(tpr)
    avg_fpr = 0.5 * (fpr[:-1] + fpr[1:])

    # Determine weights for each segment
    # Segments are defined by [tpr[i], tpr[i+1]]
    # Since we inserted 0.4, no segment crosses the boundary.
    # We can check the start point of the segment.
    t_start = tpr[:-1]

    # Weight is 2 if segment is in [0, 0.4], else 1
    # Since 0.4 is a boundary point, t_start < 0.4 implies the segment is <= 0.4
    weights = np.where(t_start < 0.4, 2, 1)

    # Calculate weighted area segments
    # We integrate (1 - FPR) with respect to TPR
    segment_areas = weights * (1 - avg_fpr) * dt

    weighted_sum = np.sum(segment_areas)

    # Normalize by the theoretical max area
    # Max area is when FPR = 0 for all TPR.
    # Integral of 1 * dt * weight
    # Range [0, 0.4]: width 0.4, weight 2 -> 0.8
    # Range [0.4, 1.0]: width 0.6, weight 1 -> 0.6
    # Total norm = 1.4
    normalization = 2 * 0.4 + 1 * 0.6

    return weighted_sum / normalization
