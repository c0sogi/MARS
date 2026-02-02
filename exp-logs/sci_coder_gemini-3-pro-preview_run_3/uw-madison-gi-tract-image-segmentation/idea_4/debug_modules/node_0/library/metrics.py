import numpy as np
from sklearn.neighbors import KDTree


def compute_dice_coefficient(y_true, y_pred):
    """
    Computes the Dice coefficient between two binary masks.

    Formula: 2 * |X intersect Y| / (|X| + |Y|)

    According to task instructions, the Dice coefficient is defined to be 0
    when both X and Y are empty.

    Args:
        y_true (np.ndarray): Ground truth binary mask (any shape).
        y_pred (np.ndarray): Predicted binary mask (same shape as y_true).

    Returns:
        float: Dice coefficient.
    """
    # Flatten arrays to ensure we are working with sets of pixels
    y_true_f = y_true.flatten() > 0
    y_pred_f = y_pred.flatten() > 0

    intersection = np.sum(y_true_f * y_pred_f)
    cardinality = np.sum(y_true_f) + np.sum(y_pred_f)

    # Specific instruction: 0 when both are empty
    if cardinality == 0:
        return 0.0

    return (2.0 * intersection) / cardinality


def compute_hausdorff_score(y_true, y_pred):
    """
    Computes the 3D Hausdorff score for two binary volumes.

    The metric is calculated by:
    1. Extracting point clouds from the binary masks.
    2. Normalizing coordinates to [0, 1] based on volume dimensions (Depth, Height, Width).
    3. Computing the directed Hausdorff distance using KDTrees.
    4. Converting distance to a score: max(0, 1 - distance).

    Args:
        y_true (np.ndarray): Ground truth 3D binary mask (Depth, Height, Width).
        y_pred (np.ndarray): Predicted 3D binary mask (Depth, Height, Width).

    Returns:
        float: Hausdorff score between 0.0 and 1.0.
    """
    # Ensure inputs are boolean
    y_true_bool = y_true > 0
    y_pred_bool = y_pred > 0

    true_count = np.sum(y_true_bool)
    pred_count = np.sum(y_pred_bool)

    # Handle empty cases
    if true_count == 0 and pred_count == 0:
        # Both empty: Distance is 0, Score is 1.0
        return 1.0
    elif true_count == 0 or pred_count == 0:
        # One empty: Distance is max (assumed >= 1), Score is 0.0
        return 0.0

    # Extract coordinates (z, y, x)
    # argwhere returns indices in order of dimensions (D, H, W)
    true_points = np.argwhere(y_true_bool).astype(np.float32)
    pred_points = np.argwhere(y_pred_bool).astype(np.float32)

    # Normalize coordinates to [0, 1]
    shape = np.array(y_true.shape, dtype=np.float32)
    true_points /= shape
    pred_points /= shape

    # Compute Directed Hausdorff Distance using KDTree for efficiency
    # h(A, B) = max(min(d(a, b)))

    # Forward: True -> Pred
    tree_pred = KDTree(pred_points)
    dist_true_pred, _ = tree_pred.query(true_points, k=1)
    h_true_pred = np.max(dist_true_pred)

    # Backward: Pred -> True
    tree_true = KDTree(true_points)
    dist_pred_true, _ = tree_true.query(pred_points, k=1)
    h_pred_true = np.max(dist_pred_true)

    # Hausdorff Distance
    hd = max(h_true_pred, h_pred_true)

    # Convert to bounded score
    return max(0.0, 1.0 - hd)


def get_competition_score(y_true, y_pred):
    """
    Computes the combined competition score.

    Score = 0.4 * Dice + 0.6 * Hausdorff_Score

    Args:
        y_true (np.ndarray): Ground truth 3D binary mask.
        y_pred (np.ndarray): Predicted 3D binary mask.

    Returns:
        float: Combined score.
    """
    dice = compute_dice_coefficient(y_true, y_pred)
    hd_score = compute_hausdorff_score(y_true, y_pred)

    return 0.4 * dice + 0.6 * hd_score
