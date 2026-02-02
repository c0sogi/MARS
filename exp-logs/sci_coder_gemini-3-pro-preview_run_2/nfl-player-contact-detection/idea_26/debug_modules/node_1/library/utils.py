import os
import random
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def shortest_arc_distance(angle1, angle2):
    """
    Computes the shortest arc distance between two angles in degrees.
    This handles the manifold discontinuity where 0 and 360 degrees are the same.
    Formula: min(|a-b|, 360-|a-b|)

    Args:
        angle1 (float, np.ndarray, or torch.Tensor): First angle(s) in degrees.
        angle2 (float, np.ndarray, or torch.Tensor): Second angle(s) in degrees.

    Returns:
        The shortest distance between the angles (always positive).
        Type matches input (np.ndarray or torch.Tensor).
    """
    # Handle PyTorch Tensors
    if isinstance(angle1, torch.Tensor) or isinstance(angle2, torch.Tensor):
        if not isinstance(angle1, torch.Tensor):
            angle1 = torch.tensor(angle1)
        if not isinstance(angle2, torch.Tensor):
            angle2 = torch.tensor(angle2)

        diff = torch.abs(angle1 - angle2) % 360
        return torch.minimum(diff, 360 - diff)

    # Handle NumPy Arrays / Scalars
    diff = np.abs(np.array(angle1) - np.array(angle2)) % 360
    return np.minimum(diff, 360 - diff)


def clamp_values(data, min_val=None, max_val=None):
    """
    Strictly clamps values in the input data to a specified range [min_val, max_val].
    Used to prevent outliers in kinematic features from destabilizing the model.

    Args:
        data (np.ndarray, pd.Series, or torch.Tensor): The input data to clamp.
        min_val (float, optional): The lower bound. Defaults to Config.CLAMP_MIN.
        max_val (float, optional): The upper bound. Defaults to Config.CLAMP_MAX.

    Returns:
        The data with values clamped to the specified range.
    """
    if min_val is None:
        min_val = Config.CLAMP_MIN
    if max_val is None:
        max_val = Config.CLAMP_MAX

    if isinstance(data, torch.Tensor):
        return torch.clamp(data, min=min_val, max=max_val)

    # Works for np.ndarray and pd.Series
    return np.clip(data, a_min=min_val, a_max=max_val)


def compute_mcc(y_true, y_pred):
    """
    Computes the Matthews Correlation Coefficient (MCC) between ground truth and predictions.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    # Ensure inputs are on CPU and numpy format if they are tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return matthews_corrcoef(y_true, y_pred)
