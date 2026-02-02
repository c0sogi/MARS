import os
import random
import numpy as np
import torch
import pandas as pd
import json
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for python, numpy, and torch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class TargetScaler:
    """
    Handles standardization (zero mean, unit variance) of scalar coupling constants,
    performed individually for each coupling type.
    """

    def __init__(self):
        self.stats = {}  # Stores mean and std for each type
        self.fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "scalar_coupling_constant",
        type_col: str = "type",
    ):
        """
        Computes mean and std for the target variable for each unique coupling type.
        """
        # Group by type and calculate statistics
        groups = df.groupby(type_col)[target_col]

        self.stats = {}
        for name, group in groups:
            self.stats[name] = {"mean": float(group.mean()), "std": float(group.std())}

        self.fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        target_col: str = "scalar_coupling_constant",
        type_col: str = "type",
    ) -> np.ndarray:
        """
        Standardizes the target column in the dataframe using the fitted statistics.
        Returns a numpy array of transformed values.
        """
        if not self.fitted:
            raise RuntimeError("TargetScaler must be fitted before calling transform.")

        # Create a copy to avoid modifying the original dataframe
        # We map the mean and std to the dataframe based on the type column
        means = df[type_col].map(lambda x: self.stats.get(x, {"mean": 0})["mean"])
        stds = df[type_col].map(lambda x: self.stats.get(x, {"std": 1})["std"])

        values = df[target_col].values
        transformed_values = (values - means.values) / stds.values

        return transformed_values

    def inverse_transform(
        self, predictions: np.ndarray, types: np.ndarray
    ) -> np.ndarray:
        """
        Inverse transforms normalized predictions back to the original scale.

        Args:
            predictions: Numpy array of normalized predictions.
            types: Numpy array of coupling types corresponding to the predictions.

        Returns:
            Numpy array of predictions in the original scale.
        """
        if not self.fitted:
            raise RuntimeError(
                "TargetScaler must be fitted before calling inverse_transform."
            )

        if len(predictions) != len(types):
            raise ValueError(
                f"Length of predictions ({len(predictions)}) and types ({len(types)}) must match."
            )

        # Convert types to array if not already
        types = np.array(types)

        # Vectorized lookup is tricky with string keys and numpy, so we use a list comprehension or map
        # For performance with large arrays, we can pre-convert stats to arrays if types were integers,
        # but here types are strings. We'll use a fast lookup.

        # Create lookup arrays/dicts
        mean_lookup = np.array([self.stats[t]["mean"] for t in types])
        std_lookup = np.array([self.stats[t]["std"] for t in types])

        original_values = predictions * std_lookup + mean_lookup
        return original_values

    def save(self, path: str):
        """
        Saves the fitted statistics to a JSON file.
        """
        if not self.fitted:
            raise RuntimeError("Cannot save an unfitted TargetScaler.")

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.stats, f, indent=4)

    def load(self, path: str):
        """
        Loads statistics from a JSON file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at {path}")

        with open(path, "r") as f:
            self.stats = json.load(f)

        self.fitted = True
        return self
