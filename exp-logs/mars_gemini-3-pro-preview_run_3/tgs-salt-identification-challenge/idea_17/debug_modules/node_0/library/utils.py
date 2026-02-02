import numpy as np
import torch
import os
import random
import shutil
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).
    The mask is flattened in column-major order (Fortran-style), consistent with
    the competition requirement: top-to-bottom, then left-to-right.

    Args:
        mask (np.array): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited list of pairs (start, length).
    """
    # Flatten in column-major order
    pixels = mask.T.flatten()

    # Handle empty mask
    if not np.any(pixels):
        return ""

    # Pad with 0s to detect start/end of runs
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Convert end indices to lengths
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(rle_str, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded string into a binary mask.

    Args:
        rle_str (str): Space-delimited list of pairs (start, length).
        shape (tuple): Expected shape of the mask (H, W).

    Returns:
        np.array: Binary mask of shape (H, W).
    """
    if not isinstance(rle_str, str) or rle_str.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape using Fortran order to match column-major encoding
    return img.reshape(shape, order="F")


def iou_metric(y_pred_bin, y_true_bin):
    """
    Calculates the IoU for a single pair of binary masks.
    """
    pred_flat = y_pred_bin.reshape(-1)
    true_flat = y_true_bin.reshape(-1)

    intersection = (pred_flat * true_flat).sum()
    union = pred_flat.sum() + true_flat.sum() - intersection

    if union == 0:
        # Both masks are empty, which is a perfect match
        return 1.0

    return intersection / union


def calc_map(preds, targs, threshold=0.5):
    """
    Calculates the Mean Average Precision at IoU thresholds [0.5, 0.95] with step 0.05.

    Args:
        preds (torch.Tensor or np.array): Predicted probabilities or binary masks.
        targs (torch.Tensor or np.array): Ground truth masks.
        threshold (float): Threshold to binarize probability predictions.

    Returns:
        float: The mean average precision score.
    """
    # Convert tensors to numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targs, torch.Tensor):
        targs = targs.detach().cpu().numpy()

    # Remove channel dimension if present (B, 1, H, W) -> (B, H, W)
    if preds.ndim == 4:
        preds = preds.squeeze(1)
    if targs.ndim == 4:
        targs = targs.squeeze(1)

    # Binarize predictions
    preds_bin = (preds > threshold).astype(np.uint8)
    targs_bin = (targs > 0.5).astype(np.uint8)

    ious = []
    for i in range(len(preds_bin)):
        iou = iou_metric(preds_bin[i], targs_bin[i])
        ious.append(iou)

    ious = np.array(ious)

    # Thresholds: 0.5, 0.55, ..., 0.95
    thresholds = np.arange(0.5, 1.0, 0.05)

    # Calculate Average Precision for each image
    # AP = Mean of (Precision at each threshold)
    # Since this is binary segmentation treated as object detection:
    # Precision(t) = 1 if IoU > t else 0

    scores = []
    for iou in ious:
        # Vectorized comparison: matches[j] is True if iou > thresholds[j]
        # Note: Competition metric specifies strict inequality "greater than"
        matches = iou > thresholds
        score = np.mean(matches)
        scores.append(score)

    # Return mean over the batch
    return np.mean(scores)


def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint is the best so far.
        checkpoint_dir (str): Directory to save the checkpoint.
        filename (str): Filename for the checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, "best_model.pth"))


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """
    Loads a model checkpoint.

    Args:
        path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        device (str): Device to map the checkpoint to.

    Returns:
        dict: The full checkpoint dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    # Handle DataParallel wrapping (remove 'module.' prefix if present)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
