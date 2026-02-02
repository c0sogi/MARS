import numpy as np
import torch
import random
import os


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
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

    Args:
        img (np.ndarray): Binary mask of shape (H, W). 1 - mask, 0 - background.

    Returns:
        str: Space-delimited string of pairs (start_position, run_length).
             Pixels are numbered from top to bottom, then left to right.
    """
    # Flatten column-wise (Fortran order) to match the requirement:
    # "numbered from top to bottom, then left to right"
    pixels = img.T.flatten()

    # We pad the pixels with 0 at the beginning and end to detect runs easily
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The runs array now contains start indices.
    # Even indices are starts of 1s, odd indices are ends of 1s (starts of 0s).
    # We calculate lengths by subtracting start positions from end positions.
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Shape of the output mask (height, width).

    Returns:
        np.ndarray: Binary mask of shape `shape`.
    """
    if not isinstance(mask_rle, str) or mask_rle == "" or str(mask_rle) == "nan":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    # Initialize flat array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape using Fortran order to match the column-wise encoding
    return img.reshape(shape, order="F")


def calculate_iou_map(preds, masks, verbose=False):
    """
    Calculates the Mean Average Precision at IoU thresholds (0.5 to 0.95).

    Args:
        preds (np.ndarray): Predicted masks. Shape (N, H, W) or (N, H*W).
        masks (np.ndarray): Ground truth masks. Shape (N, H, W) or (N, H*W).
        verbose (bool): If True, prints the final score.

    Returns:
        float: The mean average precision score.
    """
    # Ensure inputs are binary
    preds = (preds > 0).astype(np.uint8)
    masks = (masks > 0).astype(np.uint8)

    # Flatten spatial dimensions if necessary
    if preds.ndim > 2:
        preds = preds.reshape(preds.shape[0], -1)
    if masks.ndim > 2:
        masks = masks.reshape(masks.shape[0], -1)

    ious = []
    for p, m in zip(preds, masks):
        intersection = np.sum(p * m)
        union = np.sum(p) + np.sum(m) - intersection

        if union == 0:
            # Both prediction and mask are empty -> Perfect match
            iou = 1.0
        else:
            iou = intersection / union
        ious.append(iou)

    ious = np.array(ious)

    # Define thresholds: 0.5, 0.55, ..., 0.95
    thresholds = np.arange(0.5, 0.96, 0.05)

    # Calculate matches for each threshold
    # matches shape: (N_samples, N_thresholds)
    matches = ious[:, None] > thresholds[None, :]

    # Average precision per image (mean over thresholds)
    ap_per_image = np.mean(matches, axis=1)

    # Mean Average Precision over dataset
    map_score = np.mean(ap_per_image)

    if verbose:
        print(f"mAP over {len(preds)} samples: {map_score}")

    return map_score
