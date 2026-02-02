import numpy as np
from scipy.spatial.distance import directed_hausdorff
from library.config import Config


def robust_normalize(
    image, lower=Config.LOWER_PERCENTILE, upper=Config.UPPER_PERCENTILE
):
    """
    Normalize the image based on robust percentiles to handle outliers.

    Args:
        image (np.ndarray): Input image array.
        lower (float): Lower percentile threshold (default: 1.0).
        upper (float): Upper percentile threshold (default: 99.0).

    Returns:
        np.ndarray: Normalized image with values in range [0, 1].
    """
    image = image.astype(np.float32)

    # Calculate robust statistics
    p_lower = np.percentile(image, lower)
    p_upper = np.percentile(image, upper)

    # Clip values to the percentile range
    image = np.clip(image, p_lower, p_upper)

    # Scale to [0, 1]
    if p_upper > p_lower:
        image = (image - p_lower) / (p_upper - p_lower)
    else:
        # Handle constant image case
        image = np.zeros_like(image)

    return image


def rle_encode(mask):
    """
    Encodes a binary mask to Run-Length Encoding (RLE) format.
    The pixels are numbered from top to bottom, then left to right (Fortran order).

    Args:
        mask (np.ndarray): Binary mask (0 or 1).

    Returns:
        str: Space-delimited RLE string.
    """
    # Flatten column-wise (Fortran style) to match competition format
    pixels = mask.flatten(order="F")

    # We need to find the start and end of runs of 1s
    # Concatenate 0 at both ends to detect runs at the boundaries
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The 'runs' array contains start indices of value changes.
    # Because we padded with 0, the first change is 0->1 (start of run),
    # second is 1->0 (end of run), etc.
    # runs[1::2] are ends, runs[::2] are starts.
    # Length of run = end - start
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded (RLE) string to a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Expected shape of the output mask (height, width).

    Returns:
        np.ndarray: Binary mask of the specified shape.
    """
    if not isinstance(mask_rle, str) or mask_rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    # Parse starts and lengths
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # Adjust 1-based indexing to 0-based
    starts -= 1
    ends = starts + lengths

    # Create flattened array
    total_pixels = shape[0] * shape[1]
    img = np.zeros(total_pixels, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape column-wise (Fortran style)
    return img.reshape(shape, order="F")


def compute_metrics(pred, gt):
    """
    Computes Dice coefficient and normalized 3D Hausdorff distance.

    Args:
        pred (np.ndarray): Predicted binary mask (can be 2D or 3D).
        gt (np.ndarray): Ground truth binary mask (can be 2D or 3D).

    Returns:
        dict: Dictionary containing 'dice', 'hausdorff', and combined 'score'.
    """
    # Ensure inputs are binary uint8
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0.5).astype(np.uint8)

    # --- Dice Coefficient ---
    intersection = np.sum(pred * gt)
    sum_volumes = np.sum(pred) + np.sum(gt)

    # Task definition: Dice is 0 when both X and Y are empty.
    if sum_volumes == 0:
        dice = 0.0
    else:
        dice = (2.0 * intersection) / sum_volumes

    # --- 3D Hausdorff Distance ---
    # Ensure inputs are 3D for consistency (add channel/depth dim if 2D)
    if pred.ndim == 2:
        pred_3d = pred[np.newaxis, :, :]
        gt_3d = gt[np.newaxis, :, :]
    else:
        pred_3d = pred
        gt_3d = gt

    # Get coordinates of non-zero pixels
    pred_coords = np.argwhere(pred_3d)
    gt_coords = np.argwhere(gt_3d)

    # Handle empty masks for Hausdorff
    if len(pred_coords) == 0 and len(gt_coords) == 0:
        # Both empty: distance is 0
        hausdorff = 0.0
    elif len(pred_coords) == 0 or len(gt_coords) == 0:
        # One empty: max penalty. Since we normalize to [0, 1],
        # we assign 1.0 as the penalty for a completely missing/hallucinated object.
        hausdorff = 1.0
    else:
        # Normalize coordinates by image dimensions to create a bounded score
        # shape is (D, H, W)
        shape = np.array(pred_3d.shape, dtype=np.float32)

        pred_norm = pred_coords / shape
        gt_norm = gt_coords / shape

        # Compute directed Hausdorff distances
        d_pred_gt = directed_hausdorff(pred_norm, gt_norm)[0]
        d_gt_pred = directed_hausdorff(gt_norm, pred_norm)[0]

        # Symmetric Hausdorff distance is the max of the two
        hausdorff = max(d_pred_gt, d_gt_pred)

    # Combined score: 0.4 * Dice + 0.6 * (1 - Hausdorff)
    # Note: Hausdorff is a distance (lower is better), so we invert it for the score.
    score = 0.4 * dice + 0.6 * (1.0 - hausdorff)

    return {"dice": dice, "hausdorff": hausdorff, "score": score}
