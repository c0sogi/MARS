import os
import random
import numpy as np
import torch
import cv2
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import label
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.
    The pixels are numbered from top to bottom, then left to right (column-major).

    Args:
        img (np.array): Binary mask image (0 or 1).

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
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): Space-delimited string of start positions and run lengths.
        shape (tuple): The (height, width) of the target mask.

    Returns:
        np.array: Binary mask of shape `shape`.
    """
    if not isinstance(mask_rle, str) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Computes the Dice coefficient between ground truth and prediction.

    Args:
        y_true (torch.Tensor or np.array): Ground truth mask.
        y_pred (torch.Tensor or np.array): Predicted mask (binary or probabilities).
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        float: The Dice coefficient.
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


def hausdorff_3d(y_true, y_pred):
    """
    Computes the 3D Hausdorff distance between two binary volumes.
    Coordinates are normalized by image dimensions to create a bounded score.

    Args:
        y_true (np.array): Ground truth 3D binary volume (Depth, Height, Width).
        y_pred (np.array): Predicted 3D binary volume (Depth, Height, Width).

    Returns:
        float: The normalized Hausdorff distance. Returns 0.0 if both are empty,
               and 1.0 if only one is empty.
    """
    # Ensure inputs are binary
    y_true = (y_true > 0.5).astype(np.uint8)
    y_pred = (y_pred > 0.5).astype(np.uint8)

    # Extract coordinates of non-zero pixels (z, y, x)
    true_points = np.argwhere(y_true)
    pred_points = np.argwhere(y_pred)

    # Handle empty sets
    if len(true_points) == 0 and len(pred_points) == 0:
        return 0.0
    if len(true_points) == 0 or len(pred_points) == 0:
        # Penalize if one volume is empty but the other is not
        return 1.0

    # Normalize coordinates by volume dimensions
    depth, height, width = y_true.shape
    scale = np.array([depth, height, width])

    true_points_norm = true_points / scale
    pred_points_norm = pred_points / scale

    # Compute directed Hausdorff distance in both directions
    d_forward = directed_hausdorff(true_points_norm, pred_points_norm)[0]
    d_backward = directed_hausdorff(pred_points_norm, true_points_norm)[0]

    return max(d_forward, d_backward)


def post_process_3d(mask_volume):
    """
    Applies 3D Connected Component Analysis to the binary volume.
    Retains only the largest connected component to reduce Hausdorff distance errors.
    Cite solution_lesson_node_00005.

    Args:
        mask_volume (np.ndarray): Binary volume of shape (Depth, Height, Width).

    Returns:
        np.ndarray: Processed binary volume.
    """
    if not Config.USE_3D_CCA:
        return mask_volume

    # Label connected components
    labeled_mask, num_features = label(mask_volume)

    # If no features found, return original (empty)
    if num_features == 0:
        return mask_volume

    # Calculate size of each component
    # bincount works on flattened array of non-negative integers
    # Index 0 corresponds to background (value 0)
    component_sizes = np.bincount(labeled_mask.ravel())

    # If only background exists (should be covered by num_features check, but for safety)
    if len(component_sizes) < 2:
        return mask_volume

    # Find the label of the largest component (ignoring background at index 0)
    largest_component_label = component_sizes[1:].argmax() + 1

    # Create mask containing only the largest component
    processed_mask = (labeled_mask == largest_component_label).astype(np.uint8)

    return processed_mask


class MetricMonitor:
    """
    A helper class to accumulate and track metrics (e.g., Loss, Dice) over batches.
    """

    def __init__(self, float_precision=6):
        self.float_precision = float_precision
        self.metrics = {}

    def reset(self):
        self.metrics = {}

    def update(self, metric_name, val, n=1):
        """
        Updates the metric tracker.

        Args:
            metric_name (str): Name of the metric.
            val (float): Value of the metric for the current batch.
            n (int): Number of items in the batch (weight).
        """
        val = float(val)
        if metric_name not in self.metrics:
            self.metrics[metric_name] = {"sum": 0, "count": 0}
        self.metrics[metric_name]["sum"] += val * n
        self.metrics[metric_name]["count"] += n

    def get(self, metric_name):
        """Returns the average value of the specified metric."""
        if metric_name in self.metrics and self.metrics[metric_name]["count"] > 0:
            return self.metrics[metric_name]["sum"] / self.metrics[metric_name]["count"]
        return 0.0

    def __str__(self):
        """Returns a formatted string of all tracked metrics."""
        return " | ".join(
            "{}: {:.{prec}f}".format(
                name,
                metric["sum"] / metric["count"] if metric["count"] > 0 else 0.0,
                prec=self.float_precision,
            )
            for name, metric in self.metrics.items()
        )
