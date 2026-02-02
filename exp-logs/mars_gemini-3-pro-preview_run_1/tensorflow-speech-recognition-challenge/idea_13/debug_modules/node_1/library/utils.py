import os
import random
import numpy as np
import torch
from library.config import TARGET_LABELS, SILENCE_LABEL, UNKNOWN_LABEL


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def map_prediction_to_label(predicted_label):
    """
    Maps a fine-grained label (from the 30+ class set) to the 12-class competition format.

    Args:
        predicted_label (str): The label predicted by the model (e.g., 'bed', 'yes', 'silence').

    Returns:
        str: One of {'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown'}.
    """
    if predicted_label in TARGET_LABELS:
        return predicted_label
    if predicted_label == SILENCE_LABEL:
        return SILENCE_LABEL
    return UNKNOWN_LABEL


class Mixup:
    """
    Implements Mixup augmentation logic.
    Reference: mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)
    """

    def __init__(self, alpha=1.0):
        """
        Args:
            alpha (float): Hyperparameter for the Beta distribution.
                           lambda ~ Beta(alpha, alpha).
        """
        self.alpha = alpha

    def __call__(self, x, y):
        """
        Applies mixup to the input batch.

        Args:
            x (torch.Tensor): Input data batch (Batch, Channels, Time, Freq).
            y (torch.Tensor): Target labels batch (Batch,).

        Returns:
            mixed_x (torch.Tensor): Mixed input data.
            y_a (torch.Tensor): Targets for the first component.
            y_b (torch.Tensor): Targets for the second component.
            lam (float): The mixing coefficient lambda.
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        # Generate random permutation on the same device as input
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def criterion(self, criterion_fn, pred, y_a, y_b, lam):
        """
        Calculates the mixup loss.

        Args:
            criterion_fn: The loss function (e.g., CrossEntropyLoss).
            pred (torch.Tensor): Model predictions.
            y_a (torch.Tensor): Targets A.
            y_b (torch.Tensor): Targets B.
            lam (float): Mixing coefficient.

        Returns:
            torch.Tensor: Weighted loss.
        """
        return lam * criterion_fn(pred, y_a) + (1 - lam) * criterion_fn(pred, y_b)
