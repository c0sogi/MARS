import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class InverseScaler:
    """
    Handles the inverse transformation of Z-score standardized FVC predictions.
    Calculates statistics (mean and std) from the training metadata to restore
    values to their original milliliter scale.
    """

    def __init__(self, train_csv_path=None):
        """
        Initialize the scaler by loading training statistics.

        Args:
            train_csv_path (str, optional): Path to the training metadata CSV.
                                            Defaults to Config.TRAIN_CSV.
        """
        if train_csv_path is None:
            train_csv_path = Config.TRAIN_CSV

        # Load training data to calculate global statistics for Z-score inversion
        if os.path.exists(train_csv_path):
            df = pd.read_csv(train_csv_path)
            self.mean = df["FVC"].mean()
            self.std = df["FVC"].std()
        else:
            # Fallback or error handling if file doesn't exist (mostly for unit testing without data)
            raise FileNotFoundError(f"Training metadata not found at {train_csv_path}")

    def __call__(self, scaled_fvc, scaled_sigma):
        """
        Unscales the predicted FVC and Sigma from Z-score space to original units (ml).

        Args:
            scaled_fvc (torch.Tensor or np.array): Z-scored FVC predictions.
            scaled_sigma (torch.Tensor or np.array): Scaled confidence predictions.

        Returns:
            tuple: (unscaled_fvc, unscaled_sigma) in milliliters.
        """
        # Inverse Z-score for mean: x = z * std + mean
        unscaled_fvc = scaled_fvc * self.std + self.mean

        # Inverse scale for std: sigma = scaled_sigma * std
        # Note: Standard deviation scaling is multiplicative only (shift invariant)
        unscaled_sigma = scaled_sigma * self.std

        return unscaled_fvc, unscaled_sigma


class LaplaceMetric:
    """
    Computes the modified Laplace Log Likelihood metric as defined in the task.
    Accumulates results over batches to compute the global average score.

    Metric Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the internal accumulators."""
        self.score_sum = 0.0
        self.count = 0

    def update(self, preds_fvc, preds_sigma, targets_fvc):
        """
        Updates the metric with a batch of predictions.

        IMPORTANT: Inputs must be in the original scale (ml), NOT Z-scored.

        Args:
            preds_fvc (torch.Tensor or np.array): Predicted FVC (ml).
            preds_sigma (torch.Tensor or np.array): Predicted Confidence (ml).
            targets_fvc (torch.Tensor or np.array): True FVC (ml).
        """
        # Convert to numpy for calculation if inputs are tensors
        if isinstance(preds_fvc, torch.Tensor):
            preds_fvc = preds_fvc.detach().cpu().numpy()
        if isinstance(preds_sigma, torch.Tensor):
            preds_sigma = preds_sigma.detach().cpu().numpy()
        if isinstance(targets_fvc, torch.Tensor):
            targets_fvc = targets_fvc.detach().cpu().numpy()

        # Apply metric constraints
        # 1. Clip confidence at 70ml
        sigma_clipped = np.maximum(preds_sigma, 70)

        # 2. Clip error (delta) at 1000ml
        delta = np.minimum(np.abs(targets_fvc - preds_fvc), 1000)

        # Calculate metric
        sqrt_2 = np.sqrt(2)
        metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

        # Accumulate
        self.score_sum += np.sum(metric)
        self.count += len(metric)

    def compute(self):
        """
        Returns the average metric over all updated samples.

        Returns:
            float: The average Laplace Log Likelihood.
        """
        if self.count == 0:
            return 0.0
        return self.score_sum / self.count
