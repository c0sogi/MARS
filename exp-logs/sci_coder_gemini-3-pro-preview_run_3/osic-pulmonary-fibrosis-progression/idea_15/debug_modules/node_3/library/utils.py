import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TargetScaler:
    """
    Handles Z-score standardization for the FVC target variable.
    Stores mean and std from the training set to scale inputs and
    inverse-scale predictions.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, y):
        """
        Computes the mean and standard deviation of the target array.

        Args:
            y (np.array or pd.Series): The target values (FVC).
        """
        self.mean = np.mean(y)
        self.std = np.std(y)

    def transform(self, y):
        """
        Standardizes the input array using the fitted mean and std.

        Args:
            y (np.array or pd.Series): The target values to scale.

        Returns:
            np.array: The scaled values.
        """
        if self.mean is None or self.std is None:
            raise ValueError("TargetScaler has not been fitted yet.")
        return (y - self.mean) / self.std

    def inverse_transform(self, y_scaled):
        """
        Converts scaled predictions back to the original FVC domain (ml).

        Args:
            y_scaled (np.array or torch.Tensor): The scaled predicted mean.

        Returns:
            The prediction in original units.
        """
        if self.mean is None or self.std is None:
            raise ValueError("TargetScaler has not been fitted yet.")
        return y_scaled * self.std + self.mean

    def inverse_transform_sigma(self, sigma_scaled):
        """
        Converts scaled uncertainty predictions back to the original domain.
        Since sigma represents a spread/magnitude, it is only multiplied by std.

        Args:
            sigma_scaled (np.array or torch.Tensor): The scaled predicted uncertainty.

        Returns:
            The uncertainty in original units.
        """
        if self.std is None:
            raise ValueError("TargetScaler has not been fitted yet.")
        return sigma_scaled * self.std


def LaplaceLogLikelihood(y_true, y_pred, sigma):
    """
    Computes the modified Laplace Log Likelihood metric defined for the competition.

    Metric formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|y_true - y_pred|, 1000)
        score = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (torch.Tensor or np.array): True FVC values (in original ml units).
        y_pred (torch.Tensor or np.array): Predicted FVC values (in original ml units).
        sigma (torch.Tensor or np.array): Predicted Confidence/Sigma (in original ml units).

    Returns:
        float: The average metric score (higher is better, values are negative).
    """
    # Convert inputs to torch tensors if they aren't already
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor(sigma, dtype=torch.float32)

    # Move to CPU for calculation to ensure consistency
    y_true = y_true.detach().cpu()
    y_pred = y_pred.detach().cpu()
    sigma = sigma.detach().cpu()

    # 1. Clip sigma at 70 ml
    sigma_clipped = torch.clamp(sigma, min=Config.SIGMA_CLIP)

    # 2. Calculate absolute error (delta) and clip at 1000 ml
    delta = torch.abs(y_true - y_pred)
    delta = torch.clamp(delta, max=Config.MAX_ERROR)

    # 3. Compute metric
    # metric = - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - torch.log(sqrt_2 * sigma_clipped)

    return torch.mean(metric).item()
