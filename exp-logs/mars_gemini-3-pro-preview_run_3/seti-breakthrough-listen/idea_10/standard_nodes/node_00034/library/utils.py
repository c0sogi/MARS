import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

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


def pad_image(image: np.ndarray) -> np.ndarray:
    """
    Pads the input spectrogram image to the target height specified in Config.

    The input image is expected to have shape (C, H, W) -> (6, 273, 256).
    It will be padded to (6, 288, 256) by appending zeros to the end of the
    height (frequency) dimension.

    Args:
        image (np.ndarray): Input image array of shape (6, 273, 256).

    Returns:
        np.ndarray: Padded image array of shape (6, 288, 256).
    """
    current_h = image.shape[1]
    target_h = Config.IMG_HEIGHT

    if current_h >= target_h:
        return image

    pad_h = target_h - current_h

    # Pad width format: ((top, bottom), (top, bottom), (top, bottom))
    # We are padding the 2nd dimension (height/frequency) at the end.
    # Shape is (Channels, Height, Width)
    padding = ((0, 0), (0, pad_h), (0, 0))

    padded_image = np.pad(image, padding, mode="constant", constant_values=0)
    return padded_image


def mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, device: str = "cuda"
):
    """
    Performs Mixup augmentation on the input batch.

    Args:
        x (torch.Tensor): Input batch of images.
        y (torch.Tensor): Input batch of targets.
        alpha (float): Parameter for the Beta distribution.
        device (str): Device to perform computations on.

    Returns:
        mixed_x (torch.Tensor): Mixed inputs.
        y_a (torch.Tensor): Targets for the first set of images.
        y_b (torch.Tensor): Targets for the second set of images.
        lam (float): Mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the Mixup loss.

    Args:
        criterion: The loss function (e.g., BCEWithLogitsLoss).
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): Targets for the first set of images.
        y_b (torch.Tensor): Targets for the second set of images.
        lam (float): Mixing coefficient.

    Returns:
        torch.Tensor: The calculated loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Area Under the ROC Curve (AUC).

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred (np.ndarray): Predicted probabilities.

    Returns:
        float: The ROC AUC score.
    """
    # Handle edge case where only one class is present in y_true
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)
