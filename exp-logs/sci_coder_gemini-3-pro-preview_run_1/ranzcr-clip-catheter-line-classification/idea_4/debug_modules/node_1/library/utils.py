import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from timm.utils import ModelEmaV2


def seed_everything(seed=42):
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_auc_score(y_true, y_pred):
    """
    Calculates the average Area Under the ROC Curve (AUC) for multi-label classification.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels of shape (N, num_classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, num_classes).

    Returns:
        float: The average AUC score across all valid columns.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    aucs = []
    num_classes = y_true.shape[1]

    for i in range(num_classes):
        # Calculate AUC only if there are at least two unique classes in the ground truth column
        if len(np.unique(y_true[:, i])) > 1:
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                aucs.append(auc)
            except ValueError:
                # In case of unforeseen sklearn errors (e.g., all one class despite check)
                pass

    if len(aucs) == 0:
        return 0.0

    return np.mean(aucs)


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): RLE string formatted as 'start length start length ...'.
        shape (tuple): (height, width) of the output mask.

    Returns:
        np.ndarray: Binary mask of shape `shape` (uint8).
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


class ModelEma(ModelEmaV2):
    """
    Wrapper for timm.utils.ModelEmaV2 to handle Exponential Moving Average of model weights.
    """

    def __init__(self, model, decay=0.9999, device=None):
        """
        Args:
            model (torch.nn.Module): The model to track.
            decay (float): The decay rate for EMA.
            device (str or torch.device): Device to store EMA weights.
        """
        super().__init__(model, decay=decay, device=device)
