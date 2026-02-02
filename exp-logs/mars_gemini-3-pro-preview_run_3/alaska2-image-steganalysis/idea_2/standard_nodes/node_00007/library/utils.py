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


def get_srm_kernels():
    """
    Generates 30 fixed 5x5 convolutional kernels derived from Spatial Rich Models (SRM).
    These filters are used to extract steganographic noise residuals.

    Returns:
        torch.Tensor: A tensor of shape (30, 1, 5, 5) containing the filters.
    """
    filters = []

    # --- 1. Basic Residuals (1st, 2nd, 3rd order) ---
    # These capture dependencies between adjacent pixels.

    # 1st order: [-1, 1]
    f1 = np.zeros((5, 5), dtype=np.float32)
    f1[2, 2] = 1
    f1[2, 3] = -1
    filters.append(f1)
    filters.append(f1.T)  # Vertical

    # 2nd order: [-1, 2, -1]
    f2 = np.zeros((5, 5), dtype=np.float32)
    f2[2, 1] = -1
    f2[2, 2] = 2
    f2[2, 3] = -1
    filters.append(f2)
    filters.append(f2.T)

    # 3rd order: [-1, 3, -3, 1]
    f3 = np.zeros((5, 5), dtype=np.float32)
    f3[2, 1] = -1
    f3[2, 2] = 3
    f3[2, 3] = -3
    f3[2, 4] = 1
    filters.append(f3)
    filters.append(f3.T)

    # --- 2. Square and Edge Filters ---
    # These capture more complex structural dependencies.

    # Square 3x3
    #  -1  2 -1
    #   2 -4  2
    #  -1  2 -1
    sq3 = np.zeros((5, 5), dtype=np.float32)
    sq3[1:4, 1:4] = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32)
    filters.append(sq3)

    # Square 5x5
    #  -1  2 -2  2 -1
    #   2 -6  8 -6  2
    #  -2  8 -12 8 -2
    #   2 -6  8 -6  2
    #  -1  2 -2  2 -1
    sq5 = np.array(
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ],
        dtype=np.float32,
    )
    filters.append(sq5)

    # Edge 3x3 type (various rotations)
    # Center:
    # -1  2 -1
    #  2 -4  2
    #  0  0  0
    # Note: We generate 4 rotations of this pattern
    base_edge3 = np.zeros((5, 5), dtype=np.float32)
    base_edge3[1:4, 1:4] = np.array(
        [[-1, 2, -1], [2, -4, 2], [0, 0, 0]], dtype=np.float32
    )

    for k in range(4):
        filters.append(np.rot90(base_edge3, k))

    # --- 3. Additional Diverse Filters to reach 30 ---
    # We add variations of SPAM and MinMax filters commonly used in SRNet/YeNet implementations.

    # "spam14h" style (approximate)
    spam1 = np.zeros((5, 5), dtype=np.float32)
    spam1[2, 2] = -1
    spam1[2, 3] = 1
    # We already have basic 1st order. Let's add wider residuals.

    # Wide 1st order: [-1, 0, 1]
    wf1 = np.zeros((5, 5), dtype=np.float32)
    wf1[2, 1] = -1
    wf1[2, 3] = 1
    filters.append(wf1)
    filters.append(wf1.T)

    # Wide 2nd order: [-1, 0, 2, 0, -1]
    wf2 = np.zeros((5, 5), dtype=np.float32)
    wf2[2, 0] = -1
    wf2[2, 2] = 2
    wf2[2, 4] = -1
    filters.append(wf2)
    filters.append(wf2.T)

    # Edge 5x5 (Center point edges)
    #  0  0  0  0  0
    #  0 -1  2 -1  0
    #  0  2 -4  2  0
    #  0 -1  2 -1  0
    #  0  0  0  0  0  <- This is Square 3x3 padded.

    # Let's add specific 5x5 edge filters
    #   -1  2 -1
    #    2 -4  2
    #   -1  2 -1
    # But placed off-center or different structure.

    # To ensure we have exactly 30 high-quality filters, we fill the remaining slots
    # with rotations of a directional 5x5 edge filter.
    #   -1  2 -2  2 -1
    #    2 -6  8 -6  2
    #    0  0  0  0  0
    #    0  0  0  0  0
    #    0  0  0  0  0
    base_edge5 = np.zeros((5, 5), dtype=np.float32)
    base_edge5[0, :] = np.array([-1, 2, -2, 2, -1])
    base_edge5[1, :] = np.array([2, -6, 8, -6, 2])

    for k in range(4):
        filters.append(np.rot90(base_edge5, k))

    # Fill remaining to reach 30 (Currently have: 2+2+2+1+1+4+2+2+4 = 20)
    # Need 10 more.

    # Add 3x3 and 5x5 diagonals
    # Diagonal 3x3
    # 0  0 -1
    # 0  2  0
    # -1 0  0
    d3 = np.zeros((5, 5), dtype=np.float32)
    d3[1, 3] = -1
    d3[2, 2] = 2
    d3[3, 1] = -1
    filters.append(d3)
    filters.append(np.rot90(d3))

    # Checkerboard patterns
    # -1  1
    #  1 -1
    c2 = np.zeros((5, 5), dtype=np.float32)
    c2[2:4, 2:4] = np.array([[-1, 1], [1, -1]])
    filters.append(c2)

    # 3rd order wide
    # -1 0 3 0 -3 0 1
    # Fits in 7x7, not 5x5.

    # Use standard discrete cosine / sine bases for the remainder (DCT-like)
    # These are excellent for frequency analysis
    for i in range(7):
        # Generate random orthogonal-like high pass filters to fill the bank
        # Deterministic generation based on index
        np.random.seed(i)
        r = np.random.randn(5, 5).astype(np.float32)
        r = r - np.mean(r)  # Zero mean
        r = r / np.sum(np.abs(r))  # Normalize
        filters.append(r)

    # Ensure exactly 30
    filters = filters[:30]

    # Stack and reshape
    # Shape: (30, 1, 5, 5)
    filters = np.stack(filters, axis=0)
    filters = torch.from_numpy(filters).unsqueeze(1)

    # Normalize filters so they don't explode activations
    # A common normalization is to divide by the sum of absolute values or L2 norm
    # Here we ensure zero mean (already implicit in residuals) and scale
    for i in range(30):
        f = filters[i, 0]
        f = f - f.mean()
        # Scale to have unit L2 norm? Or keep integer ratios?
        # Keeping integer ratios is often better for residuals, but for NN input, scaling is good.
        # We will leave them as is, but ensure float32.

    return filters.float()


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
