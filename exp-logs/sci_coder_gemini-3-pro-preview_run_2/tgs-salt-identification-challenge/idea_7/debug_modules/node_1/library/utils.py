import os
import random
import numpy as np
import torch
from library import config


def set_seed(seed=config.SEED):
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


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.

    The pixels are numbered from top to bottom, then left to right (Fortran order).

    Args:
        mask (np.ndarray): Binary mask of shape (H, W). 1 represents the object.

    Returns:
        str: Space-delimited string of start positions and run lengths.
    """
    # Flatten in column-major order (F) to match the "top-to-bottom, left-to-right" requirement
    pixels = mask.T.flatten()

    # Pad with 0s at ends to detect changes at boundaries
    pixels = np.concatenate([[0], pixels, [0]])

    # Find where the value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths (end - start)
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): The shape of the output mask (H, W).

    Returns:
        np.ndarray: Binary mask of the specified shape.
    """
    if not isinstance(mask_rle, str) or mask_rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    total_pixels = shape[0] * shape[1]
    img = np.zeros(total_pixels, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape back to image dimensions using Fortran order
    return img.reshape(shape, order="F")


def do_kaggle_metric(predict, truth, threshold=0.5):
    """
    Calculates the Mean Average Precision at different IoU thresholds (0.5 to 0.95).

    This metric corresponds to the competition evaluation criteria.

    Args:
        predict (np.ndarray or torch.Tensor): Predicted probabilities or binary masks.
                                              Shape: (N, H, W) or (N, 1, H, W).
        truth (np.ndarray or torch.Tensor): Ground truth binary masks.
                                            Shape: (N, H, W) or (N, 1, H, W).
        threshold (float): Threshold to binarize probability maps (default 0.5).

    Returns:
        float: The mean average precision score over the batch.
    """
    # Convert tensors to numpy
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    # Binarize predictions
    predict = (predict > threshold).astype(np.uint8)
    truth = (truth > 0.5).astype(np.uint8)

    # Flatten spatial dimensions to (N, H*W) for batch processing
    predict = predict.reshape(predict.shape[0], -1)
    truth = truth.reshape(truth.shape[0], -1)

    # Calculate Intersection and Union
    intersection = (predict & truth).sum(axis=1)
    union = (predict | truth).sum(axis=1)

    # Calculate IoU per image
    # If Union is 0 (both empty), IoU is defined as 1.0 (perfect match of emptiness)
    iou = np.ones(shape=predict.shape[0], dtype=np.float32)
    non_empty = union > 0
    iou[non_empty] = intersection[non_empty] / union[non_empty]

    # Define thresholds: 0.5, 0.55, ..., 0.95
    thresholds = np.arange(0.5, 1.0, 0.05)

    # Compare IoU against thresholds
    # Shape: (N, 1) > (1, 10) -> Result (N, 10)
    matches = iou[:, None] > thresholds[None, :]

    # Average precision for each image (mean over thresholds)
    image_scores = matches.mean(axis=1)

    # Return mean score over the batch
    return image_scores.mean()
