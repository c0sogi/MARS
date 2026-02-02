import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).
    The mask is flattened in column-major order (Fortran-style) as per task reqs.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited RLE string.
    """
    # Flatten column-wise
    pixels = mask.flatten(order="F")
    # Pad to detect changes at start/end
    pixels = np.concatenate([[0], pixels, [0]])
    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # Calculate lengths (runs[1::2] are ends, runs[::2] are starts)
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(rle_str, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        rle_str (str): Space-delimited RLE string.
        shape (tuple): Target shape (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if pd.isna(rle_str) or str(rle_str).strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = str(rle_str).split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order="F")


def calculate_iou(pred_mask, gt_mask):
    """
    Calculates the Intersection over Union (IoU) for a single pair of masks.
    Handles the empty-empty case as 1.0.

    Args:
        pred_mask (np.ndarray): Predicted binary mask.
        gt_mask (np.ndarray): Ground truth binary mask.

    Returns:
        float: IoU score.
    """
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()

    if union == 0:
        # Both empty means perfect match
        return 1.0
    else:
        return intersection / union


def calc_map_score(preds, targs, thresholds=None):
    """
    Calculates the Mean Average Precision (mAP) over a range of IoU thresholds.
    Standard thresholds: 0.5 to 0.95 with step 0.05.

    Args:
        preds (np.ndarray or torch.Tensor): Batch of predicted masks (binary or logits).
        targs (np.ndarray or torch.Tensor): Batch of ground truth masks.
        thresholds (list or np.ndarray, optional): List of IoU thresholds.

    Returns:
        float: The mean average precision over the batch.
    """
    if thresholds is None:
        thresholds = np.arange(0.5, 0.96, 0.05)

    # Convert tensors to numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targs, torch.Tensor):
        targs = targs.detach().cpu().numpy()

    # Binarize if not already (assuming 0.5 threshold for logits/probs)
    # If inputs are already binary (0/1), this doesn't change anything.
    preds = (preds > 0.5).astype(np.uint8)
    targs = (targs > 0.5).astype(np.uint8)

    ious = []
    # Iterate over batch
    for i in range(len(preds)):
        # Handle (C, H, W) or (H, W) shapes
        p = preds[i].squeeze()
        t = targs[i].squeeze()
        ious.append(calculate_iou(p, t))

    # Calculate score
    # For each image, calculate precision at each threshold
    # Precision(t) = 1 if IoU > t else 0 (for single object/mask)
    # Average Precision per image = Mean(Precision(t) for all t)

    total_score = 0.0
    for iou in ious:
        # Vectorized comparison: count how many thresholds are passed
        matches = iou > thresholds
        image_score = np.mean(matches)
        total_score += image_score

    return total_score / len(preds)


def cache_data(filename):
    """
    Decorator to cache function results to the working directory using .npy.

    Args:
        filename (str): Name of the cache file (e.g., 'train_images.npy').
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            load_cached = kwargs.get("load_cached_data", False)
            cache_path = os.path.join(Config.CACHE_DIR, filename)

            if load_cached and os.path.exists(cache_path):
                # print(f"Loading cached data from {cache_path}")
                try:
                    return np.load(cache_path)
                except Exception:
                    # print("Cache load failed, recomputing...")
                    pass

            result = func(*args, **kwargs)

            # Save to cache
            os.makedirs(Config.CACHE_DIR, exist_ok=True)
            np.save(cache_path, result)
            # print(f"Saved data to {cache_path}")

            return result

        return wrapper

    return decorator


def create_submission(ids, pred_masks, output_path=Config.SUBMISSION_PATH):
    """
    Generates the submission CSV file.

    Args:
        ids (list): List of image IDs.
        pred_masks (np.ndarray or list): List/Array of predicted binary masks.
        output_path (str): Path to save the CSV.
    """
    rles = []
    for mask in pred_masks:
        # Ensure mask is binary and correct shape
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8)
        # Squeeze channel dim if present
        if mask.ndim == 3:
            mask = mask.squeeze()
        rles.append(rle_encode(mask))

    df = pd.DataFrame({"id": ids, "rle_mask": rles})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
