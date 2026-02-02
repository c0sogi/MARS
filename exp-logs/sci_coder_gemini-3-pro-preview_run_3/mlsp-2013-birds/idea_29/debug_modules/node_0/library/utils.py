import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the mean Area Under the Receiver Operating Characteristic Curve (ROC AUC).
    Explicitly handles cases where specific classes are missing in the provided batch
    (i.e., only one unique label is present) by skipping them in the average calculation.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels of shape (N, num_classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, num_classes).

    Returns:
        float: The mean ROC AUC score over all valid classes. Returns 0.5 if no classes are valid.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    n_classes = y_true.shape[1]
    auc_scores = []

    for i in range(n_classes):
        # A class is valid for AUC calculation only if it has both positive (1) and negative (0) samples
        if len(np.unique(y_true[:, i])) > 1:
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                auc_scores.append(score)
            except ValueError:
                # In case of any other sklearn error, skip this class
                continue

    if not auc_scores:
        return 0.5

    return np.mean(auc_scores)


class MixupHandler:
    """
    Implements Calibrated Mixup regularization logic.
    Can be applied to both image batches and feature vectors.
    """

    def __init__(self, alpha=None):
        """
        Args:
            alpha (float, optional): The mixup interpolation coefficient.
                                     If None, uses Config.MIXUP_ALPHA.
        """
        self.alpha = alpha if alpha is not None else Config.MIXUP_ALPHA

    def apply(self, x, y):
        """
        Applies mixup to the inputs and targets.

        Args:
            x (torch.Tensor): Input data (images or features).
            y (torch.Tensor): Target labels.

        Returns:
            tuple: (mixed_x, mixed_y)
        """
        # If alpha is 0 or less, return original data
        if self.alpha <= 0:
            return x, y

        batch_size = x.size(0)

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Calibrated Mixup: Bias lambda towards the original sample (lam > 0.5)
        # This ensures the synthetic sample is closer to the ground truth 'x'
        lam = max(lam, 1 - lam)

        # Generate random permutation for mixing
        index = torch.randperm(batch_size).to(x.device)

        # Perform mixup
        # Note: PyTorch handles scalar multiplication broadcasting automatically
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]

        return mixed_x, mixed_y
