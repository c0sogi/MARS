import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def do_length_encode(x):
    """
    Computes run-length encoding for a 1D binary array.
    Returns a list of run lengths: [start, length, start, length, ...]
    """
    # Pad with 0s to detect changes at start/end
    pixels = np.concatenate([[0], x, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # runs[1::2] are the ends, runs[::2] are starts. Calculate lengths.
    runs[1::2] -= runs[::2]
    return runs


def rle_encode(mask):
    """
    Encodes a mask (H, W) or (H, W, 1) to RLE string.
    Pixels are 1-indexed, column-major order (top-to-bottom, then left-to-right).
    """
    if hasattr(mask, "numpy"):
        mask = mask.numpy()

    if mask.ndim == 3:
        mask = mask.squeeze()

    # Flatten in Fortran order (column-major)
    pixels = mask.flatten(order="F")
    runs = do_length_encode(pixels)
    return " ".join(str(x) for x in runs)


def do_length_decode(rle_list, shape):
    """
    Decodes a list/array of RLE numbers to a mask of given shape.
    """
    starts, lengths = [
        np.asarray(x, dtype=int) for x in (rle_list[0:][::2], rle_list[1:][::2])
    ]
    starts -= 1  # Convert 1-indexed to 0-indexed
    ends = starts + lengths

    total_pixels = shape[0] * shape[1]
    img = np.zeros(total_pixels, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order="F")


def rle_decode(rle_str, shape=(101, 101)):
    """
    Decodes an RLE string to a mask (H, W).
    """
    if not isinstance(rle_str, str) or rle_str.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    rle_list = rle_str.split()
    return do_length_decode(rle_list, shape)


def calc_iou_batch(preds, labels):
    """
    Calculates IoU for a batch of predictions and labels.
    Inputs:
        preds: (N, H, W) binary numpy array or tensor
        labels: (N, H, W) binary numpy array or tensor
    Returns:
        ious: (N,) numpy array
    """
    # Convert tensors to numpy
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()

    # Ensure binary
    preds = (preds > 0).astype(np.uint8)
    labels = (labels > 0).astype(np.uint8)

    # Flatten spatial dims: (N, H*W)
    preds_flat = preds.reshape(preds.shape[0], -1)
    labels_flat = labels.reshape(labels.shape[0], -1)

    intersection = (preds_flat & labels_flat).sum(axis=1)
    union = (preds_flat | labels_flat).sum(axis=1)

    # Initialize IoUs
    ious = np.ones(preds.shape[0], dtype=np.float32)

    # If union > 0, calculate IoU. If union == 0 (both empty), IoU remains 1.0.
    non_empty = union > 0
    ious[non_empty] = intersection[non_empty] / union[non_empty]

    return ious


def get_score(preds, labels, threshold_value=None):
    """
    Calculates the Mean Average Precision (mAP) over IoU thresholds [0.5, 0.95, 0.05].

    Inputs:
        preds: (N, H, W) probabilities or binary masks
        labels: (N, H, W) binary masks
        threshold_value: float (optional). If provided, preds are binarized > threshold_value.

    Returns:
        score: float (mean score over the batch)
    """
    # Binarize predictions if a threshold is given
    if threshold_value is not None:
        if torch.is_tensor(preds):
            preds = (preds > threshold_value).float()
        else:
            preds = (preds > threshold_value).astype(np.uint8)

    ious = calc_iou_batch(preds, labels)

    # Metric sweeps over thresholds
    thresholds = np.arange(0.5, 1.0, 0.05)
    scores = []

    for t in thresholds:
        # At each threshold t, a "hit" (TP) is if IoU > t.
        # Since it's one object per image (salt mask), Precision is 1 if Hit, 0 if Miss.
        tp = (ious > t).astype(np.float32)
        scores.append(tp)

    # Stack scores: (num_thresholds, N)
    scores = np.stack(scores, axis=0)

    # Average over thresholds for each image -> (N,)
    image_scores = scores.mean(axis=0)

    # Return mean over the batch
    return float(image_scores.mean())
