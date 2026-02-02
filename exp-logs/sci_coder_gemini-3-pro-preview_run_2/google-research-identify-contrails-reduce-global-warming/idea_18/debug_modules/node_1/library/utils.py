import numpy as np
import torch


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).

    The metric checks that the pairs are sorted, positive, and the decoded pixel
    values are not duplicated. The pixels are numbered from top to bottom,
    then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).
                           1 indicates mask, 0 indicates background.

    Returns:
        str: Space-delimited list of pairs (start, length) or '-' if empty.
    """
    # Flatten in column-major order (Fortran-style) as per requirements
    pixels = mask.flatten(order="F")

    # If the mask is empty, return '-'
    if np.sum(pixels) == 0:
        return "-"

    # Pad with zeros at the beginning and end to detect all transitions
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0] is the start of the first run
    # runs[1] is the end (exclusive) of the first run
    # Calculate lengths by subtracting start indices from end indices
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def dice_coef_metric(y_pred, y_true, threshold=0.5, epsilon=1e-6):
    """
    Computes the Dice Coefficient.

    The formula is: 2 * |X \cap Y| / (|X| + |Y|)
    where X is the set of predicted pixels and Y is the ground truth.

    This function can handle both PyTorch Tensors and NumPy arrays.

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted probabilities or binary mask.
        y_true (torch.Tensor or np.ndarray): Ground truth binary mask.
        threshold (float): Threshold to convert probabilities to binary mask.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: The Dice coefficient.
    """
    # Convert NumPy arrays to PyTorch tensors if necessary
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)

    # Ensure both tensors are on the same device
    if y_pred.device != y_true.device:
        y_true = y_true.to(y_pred.device)

    # Binarize predictions based on the threshold
    y_pred_bin = (y_pred > threshold).float()
    y_true_bin = y_true.float()

    # Flatten the tensors to compute the global intersection and union for the inputs provided
    y_pred_flat = y_pred_bin.reshape(-1)
    y_true_flat = y_true_bin.reshape(-1)

    # Compute Intersection (|X ∩ Y|)
    intersection = (y_pred_flat * y_true_flat).sum()

    # Compute Union (|X| + |Y|)
    union = y_pred_flat.sum() + y_true_flat.sum()

    # Compute Dice
    dice = (2.0 * intersection) / (union + epsilon)

    return dice.item()
