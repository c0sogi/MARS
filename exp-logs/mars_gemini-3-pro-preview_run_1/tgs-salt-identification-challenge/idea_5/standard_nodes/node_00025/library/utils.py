import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(mask):
    """
    Encodes a binary mask to Run-Length Encoding (RLE).
    Pixels are one-indexed and numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited RLE string.
    """
    # Flatten column-wise (Fortran order) to match the top-to-bottom, left-to-right requirement
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect start and end of runs
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Convert end indices to lengths
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded string to a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Shape of the output mask (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if (
        not isinstance(mask_rle, str)
        or mask_rle.strip() == ""
        or str(mask_rle) == "nan"
    ):
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape column-wise to match the encoding order
    return img.reshape(shape, order="F")


def compute_map_score(preds, targets, pixel_threshold=0.5):
    """
    Computes the Mean Average Precision (mAP) at IoU thresholds [0.5, 0.55, ..., 0.95].

    Args:
        preds (torch.Tensor or np.ndarray): Predictions (N, H, W) or (N, 1, H, W).
                                            Should be probabilities (0-1).
        targets (torch.Tensor or np.ndarray): Ground truth masks (N, H, W) or (N, 1, H, W).
        pixel_threshold (float): Threshold to binarize predicted probabilities.

    Returns:
        float: The mean average precision score over the batch.
    """
    # Convert tensors to numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Remove channel dimension if present
    if preds.ndim == 4:
        preds = preds.squeeze(1)
    if targets.ndim == 4:
        targets = targets.squeeze(1)

    # Binarize predictions and targets
    preds = (preds > pixel_threshold).astype(np.uint8)
    targets = (targets > 0.5).astype(np.uint8)

    ious = []

    # Calculate IoU for each image in the batch
    for i in range(len(preds)):
        pred = preds[i]
        target = targets[i]

        intersection = (pred & target).sum()
        union = (pred | target).sum()

        if union == 0:
            # Both prediction and target are empty -> Perfect match
            iou = 1.0
        else:
            iou = intersection / union

        # Calculate score over thresholds: 0.5 to 0.95 with step 0.05
        thresholds = np.arange(0.5, 0.96, 0.05)

        # For a single image, precision at threshold t is 1 if IoU > t, else 0.
        # The AP for the image is the mean of these precisions.
        matches = iou > thresholds
        image_score = np.mean(matches)
        ious.append(image_score)

    return np.mean(ious)


def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer, etc.
        is_best (bool): Whether this is the best model so far.
        checkpoint_dir (str): Directory to save checkpoints.
        filename (str): Filename for the checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)
