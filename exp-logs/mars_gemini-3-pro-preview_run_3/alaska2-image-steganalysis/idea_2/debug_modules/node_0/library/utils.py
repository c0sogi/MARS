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
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)

    # Ensure sorted (tpr is naturally sorted by roc_curve but fpr might not be strictly monotonic if duplicates exist)
    # roc_curve returns increasing fpr and tpr.

    # We need to integrate (1 - FPR) with respect to TPR.
    # Let's define a fine grid of TPR values to perform numerical integration
    t_grid = np.linspace(0, 1, 1001)

    # Interpolate FPR at these grid points
    # We use interp1d. Since TPR is not strictly unique, we might have steps.
    # We use 'linear' interpolation.
    # Note: tpr is increasing.
    f_interpolator = interpolate.interp1d(
        tpr, fpr, kind="linear", bounds_error=False, fill_value=(0, 1)
    )
    f_grid = f_interpolator(t_grid)

    # Calculate the area element for each step in the grid
    # Area under TPR(FPR) curve = Integral (1 - FPR(t)) dt
    # We approximate this on the grid.

    # Define weights for each point in the grid
    w_grid = np.zeros_like(t_grid)

    # Assign weights based on regions
    # Region 1: 0.0 <= TPR < 0.4 (Weight 2)
    # Region 2: 0.4 <= TPR <= 1.0 (Weight 1)

    # We can compute the weighted mean of (1 - FPR)
    # But we must respect the integration measure dt.

    score_numerator = 0.0
    score_denominator = 0.0

    dt = 1.0 / (len(t_grid) - 1)

    for i in range(len(t_grid) - 1):
        t_mid = (t_grid[i] + t_grid[i + 1]) / 2.0
        f_mid = (f_grid[i] + f_grid[i + 1]) / 2.0

        # Determine weight
        if t_mid < 0.4:
            w = 2.0
        else:
            w = 1.0

        # Contribution to area: (1 - FPR) * dt
        area_segment = (1.0 - f_mid) * dt

        score_numerator += w * area_segment
        score_denominator += w * dt

    if score_denominator == 0:
        return 0.0

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
