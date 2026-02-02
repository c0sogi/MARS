import os
import random
import numpy as np
import torch
import pandas as pd
from torch.optim.swa_utils import AveragedModel, update_bn
from library.config import Config


# -------------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------------
def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic algorithms are used where possible.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------------------------------------------------
# RLE Encoding / Decoding
# -------------------------------------------------------------------------
def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).
    The mask is flattened in column-major order (top-to-bottom, then left-to-right).

    Args:
        mask (np.ndarray): Binary mask of shape (H, W). 1 for salt, 0 for background.

    Returns:
        str: Space-delimited RLE string.
    """
    # Transpose to handle column-major order (top-down, then left-right)
    pixels = mask.T.flatten()
    # Pad with 0s to detect runs at start/end
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # Convert end indices to lengths
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): Space-delimited RLE string.
        shape (tuple): Target shape (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if pd.isna(mask_rle) or mask_rle == "" or mask_rle is None:
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    # Reshape and transpose back to original orientation
    return img.reshape(shape[1], shape[0]).T


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------
def calculate_iou_batch(y_pred_batch, y_true_batch):
    """
    Calculates IoU for a batch of images.

    Args:
        y_pred_batch (np.ndarray): Predicted masks (B, H, W).
        y_true_batch (np.ndarray): Ground truth masks (B, H, W).

    Returns:
        np.ndarray: IoU scores for each image in the batch (B,).
    """
    # Binarize inputs
    y_pred_batch = (y_pred_batch > 0.5).astype(bool)
    y_true_batch = (y_true_batch > 0.5).astype(bool)

    # Flatten spatial dims: (B, H*W)
    batch_size = y_pred_batch.shape[0]
    y_pred_flat = y_pred_batch.reshape(batch_size, -1)
    y_true_flat = y_true_batch.reshape(batch_size, -1)

    intersection = (y_pred_flat & y_true_flat).sum(axis=1)
    union = (y_pred_flat | y_true_flat).sum(axis=1)

    # Default to 1.0 for empty-empty case (perfect prediction of no salt)
    iou = np.ones(batch_size, dtype=np.float32)

    # If union > 0, calculate IoU. Else (union=0), it remains 1.0.
    mask = union > 0
    iou[mask] = intersection[mask] / union[mask]

    return iou


def do_kaggle_metric(predict, truth, threshold_range=np.arange(0.5, 1.0, 0.05)):
    """
    Calculates the mean average precision at different IoU thresholds.

    Args:
        predict (list/np.ndarray/torch.Tensor): Predicted masks.
        truth (list/np.ndarray/torch.Tensor): Ground truth masks.
        threshold_range (np.ndarray): Array of IoU thresholds.

    Returns:
        float: The mean average precision score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(predict, list):
        predict = np.array(predict)
    elif isinstance(predict, torch.Tensor):
        predict = predict.detach().cpu().numpy()

    if isinstance(truth, list):
        truth = np.array(truth)
    elif isinstance(truth, torch.Tensor):
        truth = truth.detach().cpu().numpy()

    # Calculate IoU for each image
    ious = calculate_iou_batch(predict, truth)

    # Compare against thresholds: (B, T)
    thresholds = np.array(threshold_range)
    # matches[i, j] is True if iou[i] > thresholds[j]
    matches = ious[:, None] > thresholds[None, :]

    # Average precision per image (mean over thresholds)
    ap_per_image = matches.mean(axis=1)

    # Mean over dataset
    map_score = ap_per_image.mean()

    return map_score


# -------------------------------------------------------------------------
# SWA Utilities
# -------------------------------------------------------------------------
class SWAHandler:
    """
    Handles Stochastic Weight Averaging (SWA) logic.
    Wraps torch.optim.swa_utils for cleaner integration.
    """

    def __init__(self, model):
        self.swa_model = AveragedModel(model)

    def update(self, model):
        """Updates the averaged model parameters with the current model."""
        self.swa_model.update_parameters(model)

    def update_bn(self, loader, device="cuda"):
        """
        Updates BatchNorm statistics for the SWA model using the data loader.
        This is required because SWA averages weights but not BN stats.
        """

        # Custom wrapper to extract images from loader if it yields dicts
        # or tuples, ensuring compatibility with update_bn
        class LoaderWrapper:
            def __init__(self, loader):
                self.loader = loader

            def __iter__(self):
                for batch in self.loader:
                    if isinstance(batch, dict):
                        # Assuming 'image' key based on standard pipelines
                        yield batch["image"].to(device)
                    elif isinstance(batch, (list, tuple)):
                        yield batch[0].to(device)
                    else:
                        yield batch.to(device)

            def __len__(self):
                return len(self.loader)

        # Update BN statistics
        update_bn(LoaderWrapper(loader), self.swa_model, device=device)

    def get_model(self):
        """Returns the averaged model."""
        return self.swa_model
