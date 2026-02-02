import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and metrics during training.
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


class CouplingStandardizer:
    """
    Handles standardization (Z-score normalization) of scalar coupling constants
    individually for each coupling type.
    """

    def __init__(self):
        self.means = {}
        self.stds = {}

    def fit(self, df):
        """
        Computes mean and std for each coupling type in the provided dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing 'type' and 'scalar_coupling_constant'.
        """
        # Ensure we are working with the types defined in Config or present in data
        coupling_types = df["type"].unique()

        for c_type in coupling_types:
            subset = df[df["type"] == c_type]["scalar_coupling_constant"]
            mean = subset.mean()
            std = subset.std()

            # Handle NaN (single sample) or 0.0 (constant values) to prevent NaN loss
            if pd.isna(std) or std == 0:
                std = 1.0

            self.means[c_type] = mean
            self.stds[c_type] = std

    def get_params(self, c_type):
        """
        Returns (mean, std) for a specific coupling type.
        """
        return self.means.get(c_type, 0.0), self.stds.get(c_type, 1.0)

    def inverse_transform(self, values, coupling_types):
        """
        Denormalizes predicted values based on their coupling type.

        Args:
            values (np.ndarray or torch.Tensor): Normalized predictions.
            coupling_types (list or np.ndarray): Corresponding coupling types for each prediction.

        Returns:
            np.ndarray: Denormalized values in the original scale.
        """
        # Convert tensor to numpy if necessary
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()

        # Flatten values to 1D array
        values = np.array(values).flatten()
        coupling_types = np.array(coupling_types)

        if len(values) != len(coupling_types):
            raise ValueError(
                f"Shape mismatch: values {values.shape} vs types {coupling_types.shape}"
            )

        output = np.zeros_like(values, dtype=np.float64)

        # Vectorized inverse transform
        unique_types = np.unique(coupling_types)

        for c_type in unique_types:
            if c_type in self.means:
                mean = self.means[c_type]
                std = self.stds[c_type]

                # Create mask for current type
                mask = coupling_types == c_type

                # Apply inverse z-score: x = z * std + mean
                output[mask] = values[mask] * std + mean
            else:
                # Fallback if type not found (should not happen if fitted correctly)
                mask = coupling_types == c_type
                output[mask] = values[mask]

        return output
