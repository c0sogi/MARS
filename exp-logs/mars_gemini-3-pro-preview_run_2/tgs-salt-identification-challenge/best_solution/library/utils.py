import numpy as np
import torch
from library.config import seed_everything, SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility using the config utility.

    Args:
        seed (int): The seed value to use. Defaults to the value in config.py.
    """
    seed_everything(seed)


def rle_encode(mask):
    """
    Encodes a binary mask to Run-Length Encoding (RLE) format.

    The pixels are one-indexed and numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited RLE string "start length start length ..."
    """
    # Flatten in column-major order (Fortran-style) as per requirements
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect transitions at start/end
    # We use 0 as the background class
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The runs array now contains start indices of value changes.
    # Since we padded with 0, the first change is 0->1 (start of run),
    # second is 1->0 (end of run), etc.
    # Length calculation: end_pos - start_pos
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded string to a binary mask.

    Args:
        mask_rle (str): Space-delimited RLE string.
        shape (tuple): The shape of the output mask (H, W).

    Returns:
        np.ndarray: Binary mask of the specified shape.
    """
    if not isinstance(mask_rle, str) or mask_rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    # Parse starts and lengths
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # Convert 1-indexed to 0-indexed
    starts -= 1

    # Calculate end indices
    ends = starts + lengths

    # Create flat array and fill
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape back to image using Fortran order
    return img.reshape(shape, order="F")


def metric_map(predictions, ground_truths):
    """
    Calculates the Mean Average Precision (mAP) at IoU thresholds from 0.5 to 0.95.

    The metric sweeps over thresholds (0.5, 0.55, ..., 0.95).
    At each threshold, a prediction is a hit if IoU > threshold.
    The score for an image is the average precision over these thresholds.
    The final metric is the mean score over all images.

    Args:
        predictions (np.ndarray or torch.Tensor): Predicted masks.
            Shape (N, H, W) or (N, 1, H, W).
        ground_truths (np.ndarray or torch.Tensor): Ground truth masks.
            Shape (N, H, W) or (N, 1, H, W).

    Returns:
        float: The mean average precision score.
    """
    # Convert tensors to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(ground_truths, torch.Tensor):
        ground_truths = ground_truths.detach().cpu().numpy()

    # Binarize predictions and ground truths
    predictions = (predictions > 0.5).astype(np.uint8)
    ground_truths = (ground_truths > 0.5).astype(np.uint8)

    # Flatten spatial dimensions: (N, H*W)
    predictions = predictions.reshape(predictions.shape[0], -1)
    ground_truths = ground_truths.reshape(ground_truths.shape[0], -1)

    # Calculate Intersection and Union
    intersection = np.sum(predictions & ground_truths, axis=1)
    union = np.sum(predictions | ground_truths, axis=1)

    # Calculate IoU
    # If union is 0, it means both pred and gt are empty -> IoU = 1
    iou = np.ones_like(intersection, dtype=np.float32)
    non_empty_union = union > 0
    iou[non_empty_union] = intersection[non_empty_union] / union[non_empty_union]

    # Define thresholds
    thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

    # Compare IoU to thresholds
    # broadcasting: (N, 1) > (1, T) -> (N, T)
    # Result is boolean matrix where True indicates a "hit" (Precision=1)
    matches = iou[:, None] > thresholds[None, :]

    # Average over thresholds for each image
    image_scores = np.mean(matches, axis=1)

    # Average over the dataset
    final_score = np.mean(image_scores)

    return float(final_score)
