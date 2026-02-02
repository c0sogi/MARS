import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score
from timm.utils import ModelEmaV2
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
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(y_true, y_pred, threshold=0.5):
    """
    Calculates the Mean F1-Score (Macro Average) for multi-label classification.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth binary labels (N, num_classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities (N, num_classes).
        threshold (float): Threshold for converting probabilities to binary labels.

    Returns:
        float: The mean F1-score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Binarize predictions based on threshold
    y_pred_bin = (y_pred > threshold).astype(int)

    # Calculate Macro F1 Score
    # average='macro': Calculate metrics for each label, and find their unweighted mean.
    # zero_division=0: Sets the score to 0 when there are no true or predicted samples for a class.
    score = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)

    return score


def get_ema_model(model):
    """
    Initializes the Model Exponential Moving Average (EMA) utility.

    Args:
        model (torch.nn.Module): The model to wrap.

    Returns:
        ModelEmaV2 or None: The initialized EMA wrapper if USE_EMA is True, else None.
    """
    if not Config.USE_EMA:
        return None

    # Initialize ModelEmaV2 with decay from config
    # We use the device specified in Config to ensure shadow weights are stored correctly
    ema = ModelEmaV2(model, decay=Config.EMA_DECAY, device=Config.DEVICE)
    return ema
