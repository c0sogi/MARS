import os
import random
import numpy as np
import torch
from sklearn import metrics


def seed_everything(seed: int):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weighted_auc_score(y_true, y_score, tpr_thresholds, weights):
    """
    Calculates the weighted AUC score.
    The area is calculated using the trapezoidal rule on the ROC curve,
    where the TPR axis is stretched according to the defined weights in specific regions.

    Args:
        y_true: Array-like of ground truth labels (0 or 1).
        y_score: Array-like of predicted probabilities or scores.
        tpr_thresholds: List of TPR thresholds defining the regions (e.g., [0.0, 0.4, 1.0]).
        weights: List of weights for each region (e.g., [2.0, 1.0]).

    Returns:
        float: The normalized weighted AUC score between 0 and 1.
    """
    # Compute standard ROC curve
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)

    # Define a function to transform TPR based on weights
    # This effectively maps the y-axis to a weighted y-axis
    def transform_tpr(t_val):
        weighted_height = 0.0
        for i in range(len(weights)):
            w = weights[i]
            low = tpr_thresholds[i]
            high = tpr_thresholds[i + 1]

            if t_val <= low:
                break

            # Calculate the overlap of the current TPR value with the current weight region
            overlap = min(high, t_val) - low
            weighted_height += overlap * w

        return weighted_height

    # Transform all TPR values
    # Since TPR is sorted, this is efficient
    tpr_weighted = np.array([transform_tpr(t) for t in tpr])

    # Calculate AUC using the transformed TPR
    # We use FPR as x-axis and Weighted TPR as y-axis
    area = metrics.auc(fpr, tpr_weighted)

    # Normalize by the maximum possible weighted height (corresponding to TPR=1.0)
    max_weighted_height = transform_tpr(1.0)

    return area / max_weighted_height


def get_srm_weights():
    """
    Generates the 30 Spatial Rich Model (SRM) 5x5 filters.
    These filters are used to extract noise residuals from images for steganalysis.

    Returns:
        torch.Tensor: A tensor of shape (30, 1, 5, 5) containing the filters.
    """
    # Definitions of basic kernels
    # 1st order
    s1 = np.array([-1, 1], dtype=np.float32)
    # 2nd order
    s2 = np.array([-1, 2, -1], dtype=np.float32)
    # 3rd order
    s3 = np.array([-1, 3, -3, 1], dtype=np.float32)

    # Helper to pad to 5x5
    def pad_to_5x5(k):
        h, w = k.shape
        pad_h = (5 - h) // 2
        pad_w = (5 - w) // 2
        return np.pad(k, ((pad_h, 5 - h - pad_h), (pad_w, 5 - w - pad_w)), "constant")

    kernels = []

    # --- 1. Spam Filters (Derivatives) ---
    # 1st order (Horizontal, Vertical)
    k = s1.reshape(1, 2)
    kernels.append(pad_to_5x5(k))
    kernels.append(pad_to_5x5(k.T))

    # 2nd order (Horizontal, Vertical)
    k = s2.reshape(1, 3)
    kernels.append(pad_to_5x5(k))
    kernels.append(pad_to_5x5(k.T))

    # 3rd order (Horizontal, Vertical)
    k = s3.reshape(1, 4)
    kernels.append(pad_to_5x5(k))
    kernels.append(pad_to_5x5(k.T))

    # --- 2. Square & Edge Filters ---
    # 3x3 Square
    # [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]
    k_sq3 = np.outer(s2, s2)
    kernels.append(pad_to_5x5(k_sq3))

    # 5x5 Square (using s3 approximation or wider s2)
    # Let's use a 5x5 version of the square kernel
    s2_5 = np.array([-1, 0, 2, 0, -1], dtype=np.float32)
    k_sq5 = np.outer(s2_5, s2_5)
    kernels.append(k_sq5)  # Already 5x5

    # Edge 3x3 (Center vs Neighbors)
    # [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    k_edge3 = np.full((3, 3), -1.0, dtype=np.float32)
    k_edge3[1, 1] = 8.0
    kernels.append(pad_to_5x5(k_edge3))

    # Edge 5x5
    k_edge5 = np.full((5, 5), -1.0, dtype=np.float32)
    k_edge5[2, 2] = 24.0
    kernels.append(k_edge5)

    # --- 3. Mixed / Diagonal Filters to fill up to 30 ---
    # We need 20 more filters. We will generate rotations and mixed orders.

    # Diagonals for 1st, 2nd, 3rd order
    # 1st order diagonal: [[-1, 0], [0, 1]]
    k_d1 = np.eye(2, dtype=np.float32) * 2 - 1  # Results in 1, -1? No.
    k_d1 = np.array([[-1, 0], [0, 1]], dtype=np.float32)
    kernels.append(pad_to_5x5(k_d1))  # Main diag
    kernels.append(pad_to_5x5(np.fliplr(k_d1)))  # Anti diag

    # 2nd order diagonal: [[-1, 0, 0], [0, 2, 0], [0, 0, -1]]
    k_d2 = np.diag(s2)
    kernels.append(pad_to_5x5(k_d2))
    kernels.append(pad_to_5x5(np.fliplr(k_d2)))

    # 3rd order diagonal
    k_d3 = np.diag(s3)
    kernels.append(pad_to_5x5(k_d3))
    kernels.append(pad_to_5x5(np.fliplr(k_d3)))

    # Mixed 1st and 2nd (e.g., dx * dy^2)
    # outer(s1, s2) -> 2x3
    k_m12 = np.outer(s1, s2)
    kernels.append(pad_to_5x5(k_m12))
    kernels.append(pad_to_5x5(k_m12.T))

    # Mixed 1st and 3rd
    k_m13 = np.outer(s1, s3)
    kernels.append(pad_to_5x5(k_m13))
    kernels.append(pad_to_5x5(k_m13.T))

    # Mixed 2nd and 3rd
    k_m23 = np.outer(s2, s3)
    kernels.append(pad_to_5x5(k_m23))
    kernels.append(pad_to_5x5(k_m23.T))

    # Discrete Cosine Transform (DCT) like basis (High freq)
    # Checkerboard
    k_check = np.array([[1, -1], [-1, 1]], dtype=np.float32)
    kernels.append(pad_to_5x5(k_check))

    # 4th order (central)
    s4 = np.array([1, -4, 6, -4, 1], dtype=np.float32)
    k_s4h = s4.reshape(1, 5)
    kernels.append(k_s4h)
    kernels.append(k_s4h.T)

    # 4th order diagonal
    k_s4d = np.diag(s4)
    kernels.append(k_s4d)
    kernels.append(np.fliplr(k_s4d))

    # Add a few more mixed variations to reach exactly 30
    # Current count:
    # 1st(2) + 2nd(2) + 3rd(2) + Sq3(1) + Sq5(1) + Ed3(1) + Ed5(1) = 10
    # D1(2) + D2(2) + D3(2) = 6 -> Total 16
    # M12(2) + M13(2) + M23(2) = 6 -> Total 22
    # Check(1) + S4(2) + S4D(2) = 5 -> Total 27

    # Need 3 more.
    # Mixed 1st order diagonal with 2nd order?
    # Let's add standard "MinMax" inspired linear components
    # 3x3 Cross: [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    k_cross3 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    kernels.append(pad_to_5x5(k_cross3))

    # 5x5 Cross
    k_cross5 = np.zeros((5, 5), dtype=np.float32)
    k_cross5[2, :] = [-1, 2, -6, 2, -1]  # Custom high pass
    kernels.append(k_cross5)
    kernels.append(k_cross5.T)

    # Ensure we have exactly 30
    kernels = kernels[:30]

    # Normalize kernels to have sum 0 (high-pass constraint)
    # Most are already sum 0.
    final_kernels = []
    for k in kernels:
        k = k - k.mean()  # Enforce zero mean
        # Normalize energy? Not strictly necessary for weights, but good for stability
        # k = k / (np.sum(np.abs(k)) + 1e-6)
        final_kernels.append(k)

    # Stack into tensor: (30, 1, 5, 5)
    weights = np.stack(final_kernels, axis=0)
    weights = torch.from_numpy(weights).unsqueeze(1)

    return weights
