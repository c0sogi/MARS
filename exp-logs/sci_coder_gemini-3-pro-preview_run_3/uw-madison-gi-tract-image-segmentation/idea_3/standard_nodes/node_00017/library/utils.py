import os
import numpy as np
import torch
import cv2
import random
from library.config import Config

# Try importing scipy.ndimage for Connected Component Analysis
try:
    from scipy import ndimage
except ImportError:
    ndimage = None


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).

    Args:
        img (np.ndarray): Binary mask (0 or 1).

    Returns:
        str: Space-delimited string of start positions and lengths.
             Pixels are numbered from top to bottom, then left to right (Fortran order).
    """
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Target shape (Height, Width).

    Returns:
        np.ndarray: Binary mask with the specified shape.
    """
    if not isinstance(mask_rle, str) or mask_rle == "" or mask_rle == "nan":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order="F")


def get_dice_score(y_pred, y_true):
    """
    Calculates the Dice Coefficient between two binary masks.

    Args:
        y_pred (np.ndarray or torch.Tensor): Predicted binary mask.
        y_true (np.ndarray or torch.Tensor): Ground truth binary mask.

    Returns:
        float: Dice coefficient.
    """
    smooth = 1e-5

    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()

    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    intersect = np.sum(y_pred * y_true)
    union = np.sum(y_pred) + np.sum(y_true)

    return (2.0 * intersect + smooth) / (union + smooth)


def get_3d_hausdorff(pred_mask, true_mask):
    """
    Calculates the 3D Hausdorff distance between predicted and true masks.
    Coordinates are normalized by the image dimensions to create a bounded score.

    Args:
        pred_mask (np.ndarray): 3D Predicted binary mask (D, H, W).
        true_mask (np.ndarray): 3D Ground truth binary mask (D, H, W).

    Returns:
        float: The Hausdorff distance. Returns 0.0 if both empty, 1.0 if one is empty.
    """
    # Handle empty cases
    pred_sum = np.sum(pred_mask)
    true_sum = np.sum(true_mask)

    if pred_sum == 0 and true_sum == 0:
        return 0.0
    if pred_sum == 0 or true_sum == 0:
        return 1.0  # Max normalized distance penalty

    device = Config.DEVICE

    # Extract coordinates of non-zero pixels (z, y, x)
    p_coords = np.argwhere(pred_mask)
    t_coords = np.argwhere(true_mask)

    # Normalize coordinates by dimensions (D, H, W) to map to unit cube [0, 1]^3
    # This aligns with the requirement: "pixel locations are normalized by image size"
    dims = np.array(pred_mask.shape, dtype=np.float32)

    p_tensor = torch.tensor(
        p_coords, dtype=torch.float32, device=device
    ) / torch.tensor(dims, device=device)
    t_tensor = torch.tensor(
        t_coords, dtype=torch.float32, device=device
    ) / torch.tensor(dims, device=device)

    def directed_hausdorff(A, B):
        """
        Computes max_a min_b ||a - b|| efficiently using batches.
        """
        max_dist = 0.0
        num_points_A = A.shape[0]
        num_points_B = B.shape[0]

        if num_points_A == 0 or num_points_B == 0:
            return 0.0

        # Dynamic batch sizing to prevent OOM
        # Target max matrix size: ~256MB (approx 64 million float32s)
        MAX_MATRIX_ELEMENTS = 64 * 1024 * 1024
        batch_size = max(1, MAX_MATRIX_ELEMENTS // num_points_B)

        # Cap at 5000 to avoid overly large batches when B is small
        batch_size = min(batch_size, 5000)

        for i in range(0, num_points_A, batch_size):
            # Get batch of points from A
            batch_A = A[i : i + batch_size]

            # Compute squared Euclidean distances between batch_A and all B
            # cdist returns (Batch, Num_B)
            dists = torch.cdist(batch_A, B)

            # Find min distance for each point in batch_A to any point in B
            min_dists, _ = torch.min(dists, dim=1)

            # Update global max
            curr_max = torch.max(min_dists).item()
            if curr_max > max_dist:
                max_dist = curr_max

        return max_dist

    # Hausdorff(A, B) = max(h(A, B), h(B, A))
    d_pt = directed_hausdorff(p_tensor, t_tensor)
    d_tp = directed_hausdorff(t_tensor, p_tensor)

    return max(d_pt, d_tp)


def keep_largest_component(mask):
    """
    Post-processing: Keeps only the largest connected component in the 3D mask.
    Removes small noise which heavily penalizes Hausdorff distance.

    Args:
        mask (np.ndarray): 3D binary mask.

    Returns:
        np.ndarray: Processed 3D binary mask.
    """
    if ndimage is None:
        return mask

    mask_bool = mask.astype(bool)

    # Label connected components
    labeled_mask, num_features = ndimage.label(mask_bool)

    if num_features == 0:
        return mask

    # Calculate size of each component
    # label 0 is background, so we look at 1..num_features
    sizes = ndimage.sum(mask_bool, labeled_mask, range(1, num_features + 1))

    # Check if the largest component meets the minimum size requirement
    max_size = np.max(sizes)
    if max_size < Config.MIN_COMPONENT_SIZE:
        return np.zeros_like(mask)

    max_label = np.argmax(sizes) + 1

    # Create new mask with only the largest component
    new_mask = (labeled_mask == max_label).astype(np.uint8)

    return new_mask
