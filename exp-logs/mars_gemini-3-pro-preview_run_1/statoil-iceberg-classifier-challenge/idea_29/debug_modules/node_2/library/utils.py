import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss


def set_seed(seed=42):
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the log loss metric.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted probabilities.

    Returns:
        float: The log loss value.
    """
    # sklearn log_loss handles clipping internally (eps=1e-15)
    # We provide labels=[0, 1] to ensure correct calculation even if y_true
    # in a specific batch contains only one class.
    return log_loss(y_true, y_pred, labels=[0, 1])


def predict_with_klein_tta(model, images, angles=None):
    """
    Performs inference using Klein Four-Group Test Time Augmentation.
    Augmentations: Identity, Horizontal Flip, Vertical Flip, Rotate 180.

    The function handles converting logits to probabilities and averaging them.

    Args:
        model: The PyTorch model in eval mode.
        images: Input tensor of shape (B, C, H, W).
        angles: Optional tensor of incidence angles of shape (B,).

    Returns:
        torch.Tensor: Averaged probabilities of shape (B, 1).
    """
    # Ensure model is in eval mode
    model.eval()

    probs_list = []

    # Define the 4 transformations corresponding to the Klein Four-Group
    # 1. Identity
    # 2. Horizontal Flip (dim 3)
    # 3. Vertical Flip (dim 2)
    # 4. Rotate 180 (equivalent to H-Flip + V-Flip, i.e., flip dims 2 and 3)
    transforms = [
        lambda x: x,
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(x, [2]),
        lambda x: torch.flip(x, [2, 3]),
    ]

    with torch.no_grad():
        for t in transforms:
            aug_images = t(images)

            # Handle Late Fusion input if angles are provided
            if angles is not None:
                logits = model(aug_images, angles)
            else:
                logits = model(aug_images)

            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(logits)
            probs_list.append(probs)

    # Stack probabilities and compute arithmetic mean
    # probs_list contains 4 tensors of shape (B, 1)
    stacked_probs = torch.stack(probs_list)
    mean_probs = torch.mean(stacked_probs, dim=0)

    return mean_probs
