import os
import random
import numpy as np
import torch
from scipy.stats import spearmanr


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


def compute_spearman_metric(preds, targets):
    """
    Calculates the mean column-wise Spearman's correlation coefficient.

    This function handles the concatenation of question and answer predictions
    if they are provided as separate arrays/tensors in a list.

    Args:
        preds: Predictions. Can be:
               - A single np.ndarray or torch.Tensor of shape (N, 30).
               - A list/tuple of two np.ndarrays or torch.Tensors [(N, 21), (N, 9)].
        targets: Ground truth labels. np.ndarray or torch.Tensor of shape (N, 30).

    Returns:
        float: The mean Spearman's rank correlation coefficient across all 30 targets.
    """

    # Helper function to convert tensors to numpy
    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return x

    # Convert targets to numpy
    targets = to_numpy(targets)

    # Handle predictions: Concatenate if provided as a list (Question Head + Answer Head)
    if isinstance(preds, (list, tuple)):
        preds = [to_numpy(p) for p in preds]
        # Concatenate along the feature axis (axis 1)
        preds = np.concatenate(preds, axis=1)
    else:
        preds = to_numpy(preds)

    # Ensure shapes match
    if preds.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"
        )

    corrs = []
    num_cols = preds.shape[1]

    for i in range(num_cols):
        p = preds[:, i]
        t = targets[:, i]

        # Handle constant values which cause spearmanr to return nan
        if np.std(p) == 0 or np.std(t) == 0:
            corr = 0.0
        else:
            try:
                corr, _ = spearmanr(p, t)
                if np.isnan(corr):
                    corr = 0.0
            except Exception:
                corr = 0.0

        corrs.append(corr)

    return float(np.mean(corrs))
