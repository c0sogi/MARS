import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    The RLE format is expected to be pairs of (start_index, length), where pixels are
    numbered from top to bottom, then left to right (column-major order).

    Args:
        mask_rle (str or float/nan): The RLE string. If NaN or empty, returns a zero mask.
        shape (tuple): The shape of the output mask (height, width).

    Returns:
        np.ndarray: A binary mask of shape `shape` with values 0 and 1.
    """
    if (
        mask_rle is None
        or (isinstance(mask_rle, float) and np.isnan(mask_rle))
        or mask_rle == ""
    ):
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    # Create flat array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape using Fortran order (column-major) as per standard RLE format
    return img.reshape(shape, order="F")


def get_score(y_true, y_pred):
    """
    Calculates the average Area Under the ROC Curve (AUC) for multi-label classification.

    Args:
        y_true (np.ndarray or pd.DataFrame): Ground truth binary labels.
        y_pred (np.ndarray or pd.DataFrame): Predicted probabilities.

    Returns:
        float: The macro-averaged AUC score.
    """
    # Convert to numpy arrays if they are pandas objects
    if hasattr(y_true, "values"):
        y_true = y_true.values
    if hasattr(y_pred, "values"):
        y_pred = y_pred.values

    # Calculate macro-averaged ROC AUC
    # We use a try-except block to handle cases where a column might have only one class in the batch/split
    try:
        score = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # Fallback for edge cases in small batches or single-class targets
        # Calculate per column and ignore columns with only one class
        scores = []
        for i in range(y_true.shape[1]):
            try:
                s = roc_auc_score(y_true[:, i], y_pred[:, i])
                scores.append(s)
            except ValueError:
                pass
        score = np.mean(scores) if scores else 0.5

    return score
