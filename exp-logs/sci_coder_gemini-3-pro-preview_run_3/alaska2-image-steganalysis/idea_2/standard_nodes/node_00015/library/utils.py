import os
import random
import numpy as np
import torch
import cv2
from sklearn import metrics
from scipy import interpolate
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility using the Config class.
    """
    Config.set_seed(seed)


def get_hpf_kernel():
    """
    Returns a fixed 5x5 High-Pass Filter (KV Kernel) for residual extraction.
    Normalized by 12.0 as per best practices (Cite solution_lesson_node_00011).
    """
    # KV kernel
    k = np.array(
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ],
        dtype=np.float32,
    )
    # Normalize
    k = k / 12.0

    # Reshape to (Out=1, In=1, H=5, W=5)
    return torch.from_numpy(k).view(1, 1, 5, 5)


def weighted_auc(y_true, y_pred):
    """
    Calculates the Weighted AUC metric.

    The metric is defined as the weighted average of the area under the ROC curve
    (specifically, the integral of TPR with respect to FPR, or equivalently 1 - integral of FPR w.r.t TPR)
    partitioned by TPR thresholds.

    Parameters:
        y_true (array-like): True binary labels.
        y_pred (array-like): Target scores (probabilities or logits).

    Returns:
        float: The weighted AUC score.
    """
    tpr_thresholds = Config.TPR_THRESHOLDS
    weights = Config.TPR_WEIGHTS

    # Compute ROC curve
    # fpr and tpr are increasing arrays starting at (0,0) and ending at (1,1)
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)

    # We calculate the weighted AUC by integrating (1 - FPR) with respect to TPR.
    # Weighted AUC = (Integral w(t) * (1 - FPR(t)) dt) / (Integral w(t) dt)
    #              = (Integral w(t) dt - Integral w(t) * FPR(t) dt) / (Integral w(t) dt)

    # 1. Calculate Denominator (Total Weighted Length of TPR axis)
    score_denominator = 0.0
    for i in range(len(tpr_thresholds) - 1):
        score_denominator += weights[i] * (tpr_thresholds[i + 1] - tpr_thresholds[i])

    if score_denominator == 0:
        return 0.0

    # 2. Calculate Integral w(t) * FPR(t) dt
    # This represents the weighted area to the "left" of the ROC curve (between y-axis and curve).
    # We compute this geometrically by iterating through the ROC segments.
    area_left = 0.0

    for i in range(len(tpr) - 1):
        # Current ROC segment
        x0, y0 = fpr[i], tpr[i]
        x1, y1 = fpr[i + 1], tpr[i + 1]

        # Skip horizontal segments in ROC (vertical in FPR vs TPR) as they have 0 width in TPR
        if y1 == y0:
            continue

        # Iterate through weight regions to apply correct weights
        for j in range(len(tpr_thresholds) - 1):
            t_start = tpr_thresholds[j]
            t_end = tpr_thresholds[j + 1]
            w = weights[j]

            # Determine overlap between current ROC segment [y0, y1] and weight region [t_start, t_end]
            overlap_start = max(y0, t_start)
            overlap_end = min(y1, t_end)

            if overlap_end > overlap_start:
                # We have an overlap. We need to integrate FPR(t) * w over [overlap_start, overlap_end].
                # In the segment [y0, y1], FPR(t) is linear (assuming linear interpolation between points).
                # x(y) = x0 + (y - y0) * (x1 - x0) / (y1 - y0)

                # Calculate x values at the boundaries of the overlap
                if overlap_start == y0:
                    x_s = x0
                else:
                    x_s = x0 + (overlap_start - y0) * (x1 - x0) / (y1 - y0)

                if overlap_end == y1:
                    x_e = x1
                else:
                    x_e = x0 + (overlap_end - y0) * (x1 - x0) / (y1 - y0)

                # Calculate area of the trapezoid for this overlap
                # Area = width * average_height
                dy = overlap_end - overlap_start
                avg_x = (x_s + x_e) / 2.0
                area_segment = dy * avg_x

                # Add weighted contribution
                area_left += w * area_segment

    # 3. Final Score
    # Numerator = Total Weighted Area - Weighted Area Left
    score_numerator = score_denominator - area_left

    return score_numerator / score_denominator


def read_image(path):
    """
    Reads an image from path and extracts the Y (Luminance) channel.

    Args:
        path (str): Path to the image file.

    Returns:
        np.ndarray: 2D array of the Y channel (height, width).
    """
    # Read image in BGR format
    img = cv2.imread(path)
    if img is None:
        # Return a zero placeholder if image is corrupt/missing (should not happen in clean data)
        # Assuming 512x512 based on config
        return np.zeros((512, 512), dtype=np.uint8)

    # Convert to YCrCb
    img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # Extract Y channel (index 0)
    y_channel = img_ycc[:, :, 0]

    return y_channel
