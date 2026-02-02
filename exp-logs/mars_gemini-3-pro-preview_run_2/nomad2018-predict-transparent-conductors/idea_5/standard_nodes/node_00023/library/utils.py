import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class StandardScaler:
    """
    A utility class for standardizing data (zero mean, unit variance)
    using PyTorch tensors. Useful for normalizing targets and global features.
    """

    def __init__(self, mean=None, std=None, device=None):
        """
        Initialize the scaler. Can be initialized with pre-computed statistics.

        Args:
            mean (torch.Tensor, optional): Pre-computed mean.
            std (torch.Tensor, optional): Pre-computed standard deviation.
            device (torch.device, optional): Device to store the statistics on.
        """
        self.mean = mean
        self.std = std
        self.device = device if device else torch.device("cpu")

        if self.mean is not None:
            self.mean = self.mean.to(self.device)
        if self.std is not None:
            self.std = self.std.to(self.device)

    def fit(self, data: torch.Tensor):
        """
        Compute the mean and std to be used for later scaling.

        Args:
            data (torch.Tensor): The data to fit, typically shape [N, Features].
        """
        self.mean = torch.mean(data, dim=0).to(self.device)
        self.std = torch.std(data, dim=0).to(self.device)

        # Handle zero standard deviation to avoid division by zero
        # Replace 0s with 1s (no scaling for those features)
        self.std = torch.where(
            self.std == 0, torch.tensor(1.0, device=self.device), self.std
        )

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Perform standardization by centering and scaling.

        Args:
            data (torch.Tensor): Data to transform.

        Returns:
            torch.Tensor: Transformed data.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler has not been fitted yet.")

        data = data.to(self.device)
        return (data - self.mean) / self.std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Scale back the data to the original representation.

        Args:
            data (torch.Tensor): Data to inverse transform (e.g., model predictions).

        Returns:
            torch.Tensor: Data in original scale.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler has not been fitted yet.")

        data = data.to(self.device)
        return (data * self.std) + self.mean

    def to(self, device):
        """
        Moves the statistics to the specified device.

        Args:
            device (torch.device): The target device.

        Returns:
            self
        """
        self.device = device
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.std is not None:
            self.std = self.std.to(device)
        return self

    def state_dict(self):
        """
        Returns the state of the scaler for saving.
        """
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        """
        Loads the state of the scaler.
        """
        self.mean = state_dict["mean"].to(self.device)
        self.std = state_dict["std"].to(self.device)
