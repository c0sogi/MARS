import os
import random
import numpy as np
import torch
import cv2
from sklearn.metrics import pairwise_distances


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.
    The pixels are numbered from top to bottom, then left to right (Fortran order).

    Args:
        img (numpy.ndarray): Binary mask (0 for background, 1 for object).

    Returns:
        str: Space-delimited string of start positions and run lengths.
    """
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes an RLE string into a binary mask.

    Args:
        mask_rle (str): Run-length encoded string.
        shape (tuple): Target shape (height, width) of the mask.

    Returns:
        numpy.ndarray: Binary mask with shape `shape`.
    """
    if mask_rle is None or not isinstance(mask_rle, str) or mask_rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Computes the Dice coefficient between ground truth and prediction.

    Args:
        y_true (numpy.ndarray or torch.Tensor): Ground truth mask.
        y_pred (numpy.ndarray or torch.Tensor): Predicted mask.
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        float: The Dice coefficient score.
    """
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        np.sum(y_true_f) + np.sum(y_pred_f) + smooth
    )


def hausdorff_distance(y_true, y_pred):
    """
    Computes the Hausdorff distance between two binary masks.
    Coordinates are normalized by image dimensions to create a bounded score.
    Uses contour extraction for efficiency.

    Args:
        y_true (numpy.ndarray or torch.Tensor): Ground truth mask (H, W).
        y_pred (numpy.ndarray or torch.Tensor): Predicted mask (H, W).

    Returns:
        float: The normalized Hausdorff distance. Returns 0.0 if both are empty,
               and 1.0 if only one is empty.
    """
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are uint8 binary masks for cv2
    y_true = (y_true > 0.5).astype(np.uint8)
    y_pred = (y_pred > 0.5).astype(np.uint8)

    h, w = y_true.shape

    def get_contour_points(mask):
        # Find contours to reduce the number of points for distance calculation
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return np.empty((0, 2))
        return np.vstack(contours).reshape(-1, 2)

    true_pts = get_contour_points(y_true)
    pred_pts = get_contour_points(y_pred)

    # Handle empty mask cases
    if len(true_pts) == 0 and len(pred_pts) == 0:
        return 0.0
    if len(true_pts) == 0 or len(pred_pts) == 0:
        return 1.0

    # Normalize coordinates (x/width, y/height)
    # Contour points are in (x, y) format
    scaler = np.array([w, h], dtype=np.float32)
    true_pts_norm = true_pts.astype(np.float32) / scaler
    pred_pts_norm = pred_pts.astype(np.float32) / scaler

    # Compute pairwise Euclidean distances between boundary points
    dists = pairwise_distances(true_pts_norm, pred_pts_norm)

    # Hausdorff distance = max(directed_A_B, directed_B_A)
    # Directed A->B: max(min(d(a, B)))
    hd_AB = np.max(np.min(dists, axis=1))
    # Directed B->A: max(min(d(b, A)))
    hd_BA = np.max(np.min(dists, axis=0))

    return max(hd_AB, hd_BA)
