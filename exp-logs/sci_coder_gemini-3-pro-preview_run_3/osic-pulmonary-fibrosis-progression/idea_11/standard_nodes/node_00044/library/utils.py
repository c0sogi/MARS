import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
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


def metric_score(y_true, y_pred, sigma_pred):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: Ground truth FVC values (numpy array or torch tensor).
        y_pred: Predicted FVC values (numpy array or torch tensor).
        sigma_pred: Predicted confidence (sigma) values (numpy array or torch tensor).

    Returns:
        float: The average metric score over the batch.
    """
    # Convert tensors to numpy if necessary
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(sigma_pred):
        sigma_pred = sigma_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sigma_pred = np.asarray(sigma_pred)

    # Constants
    sigma_clip_val = 70.0
    delta_clip_val = 1000.0
    sqrt_2 = np.sqrt(2)

    # 1. Clip sigma
    sigma_clipped = np.maximum(sigma_pred, sigma_clip_val)

    # 2. Calculate absolute error and clip it
    abs_error = np.abs(y_true - y_pred)
    delta = np.minimum(abs_error, delta_clip_val)

    # 3. Compute metric
    # Term 1: - (sqrt(2) * delta) / sigma_clipped
    term1 = -(sqrt_2 * delta) / sigma_clipped

    # Term 2: - ln(sqrt(2) * sigma_clipped)
    term2 = -np.log(sqrt_2 * sigma_clipped)

    metric = term1 + term2

    return np.mean(metric)


class InverseScaler:
    """
    Handles the inverse transformation of standardized predictions back to the original scale.

    Since the model is trained on Z-scored FVC values (mean=0, std=1), we need to
    transform predictions back to ml.

    Transformations:
        mu_original = mu_pred * std + mean
        sigma_original = sigma_pred * std (Scale only, no shift)
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, data):
        """
        Computes mean and std from a pandas Series or numpy array.
        """
        self.mean = np.mean(data)
        self.std = np.std(data)

    def __call__(self, mu_pred, sigma_pred=None):
        """
        Inverse transforms the predictions.

        Args:
            mu_pred: Predicted FVC (standardized).
            sigma_pred: Predicted Confidence (standardized), optional.

        Returns:
            Tuple (mu_orig, sigma_orig) if sigma_pred is provided, else mu_orig.
        """
        if self.mean is None or self.std is None:
            raise ValueError(
                "InverseScaler must be fitted or initialized with mean/std before use."
            )

        # Convert to numpy if tensor
        if torch.is_tensor(mu_pred):
            mu_pred = mu_pred.detach().cpu().numpy()

        mu_orig = mu_pred * self.std + self.mean

        if sigma_pred is not None:
            if torch.is_tensor(sigma_pred):
                sigma_pred = sigma_pred.detach().cpu().numpy()
            # Sigma is a scale parameter, so we only multiply by std, we do not add mean.
            sigma_orig = sigma_pred * self.std
            return mu_orig, sigma_orig

        return mu_orig
