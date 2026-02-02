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
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TargetScaler:
    """
    Handles Z-score standardization for the target variable (FVC) and
    scaling for the confidence measure (Sigma).
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, y):
        """
        Computes the mean and standard deviation of the target variable from the training data.

        Args:
            y (np.array or pd.Series): The target values (FVC).
        """
        self.mean = np.mean(y)
        self.std = np.std(y)

    def transform(self, y):
        """
        Standardizes the target variable using the computed mean and std.

        Args:
            y (np.array or pd.Series): The target values.

        Returns:
            np.array: Standardized values.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")
        return (y - self.mean) / self.std

    def inverse_transform(self, y_scaled, sigma_scaled=None):
        """
        Reverts the standardization for predictions and scales the confidence.

        Args:
            y_scaled (np.array or torch.Tensor): Predicted scaled FVC.
            sigma_scaled (np.array or torch.Tensor, optional): Predicted scaled confidence.

        Returns:
            y (np.array or torch.Tensor): Original scale FVC.
            sigma (np.array or torch.Tensor): Original scale confidence (if sigma_scaled is provided).
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")

        # Handle PyTorch Tensors
        is_tensor = torch.is_tensor(y_scaled)

        if is_tensor:
            mean = torch.tensor(self.mean, device=y_scaled.device, dtype=y_scaled.dtype)
            std = torch.tensor(self.std, device=y_scaled.device, dtype=y_scaled.dtype)
        else:
            mean = self.mean
            std = self.std

        # Inverse transform FVC: y = z * std + mean
        y = y_scaled * std + mean

        if sigma_scaled is not None:
            # Inverse transform Sigma: sigma = z_sigma * std
            # Note: Sigma is a scale parameter, so we only multiply by std, no mean addition.
            sigma = sigma_scaled * std
            return y, sigma

        return y


class MetricMonitor:
    """
    Tracks the modified Laplace Log Likelihood metric during training/validation.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the internal state of the monitor."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, true_fvc, pred_fvc, pred_sigma):
        """
        Updates the metric tracker with a batch of predictions.

        Args:
            true_fvc (np.array or torch.Tensor): Ground truth FVC values.
            pred_fvc (np.array or torch.Tensor): Predicted FVC values.
            pred_sigma (np.array or torch.Tensor): Predicted Confidence (sigma) values.
        """
        # Convert tensors to numpy arrays for calculation
        if torch.is_tensor(true_fvc):
            true_fvc = true_fvc.detach().cpu().numpy()
        if torch.is_tensor(pred_fvc):
            pred_fvc = pred_fvc.detach().cpu().numpy()
        if torch.is_tensor(pred_sigma):
            pred_sigma = pred_sigma.detach().cpu().numpy()

        # Apply metric constraints defined in Config
        # sigma_clipped = max(sigma, 70)
        sigma_clipped = np.maximum(pred_sigma, Config.CONFIDENCE_CLIP)

        # delta = min(|true - pred|, 1000)
        delta = np.abs(true_fvc - pred_fvc)
        delta = np.minimum(delta, Config.ERROR_THRESHOLD)

        # Calculate metric
        # metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)
        sqrt_2 = np.sqrt(2)
        metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

        batch_size = len(true_fvc)
        self.val = np.mean(metric)
        self.sum += np.sum(metric)
        self.count += batch_size
        self.avg = self.sum / self.count
