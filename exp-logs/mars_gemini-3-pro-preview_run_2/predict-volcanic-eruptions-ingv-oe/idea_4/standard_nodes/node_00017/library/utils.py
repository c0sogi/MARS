import os
import random
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from library.config import Config


def seed_everything(seed=42):
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
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TargetScaler:
    """
    A wrapper around StandardScaler to handle target variable scaling and
    persistence of scaler statistics (mean and std) to disk.

    This ensures that the exact same scaling used during training can be
    loaded and applied (inversely) during inference.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.mean_path = Config.TARGET_SCALER_MEAN
        self.std_path = Config.TARGET_SCALER_STD
        self.is_fitted = False

    def fit(self, y):
        """
        Compute the mean and std to be used for later scaling.

        Args:
            y (np.array): The target values.
        """
        # Ensure input is 2D (n_samples, n_features)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.scaler.fit(y)
        self.is_fitted = True
        self._save_params()

    def transform(self, y):
        """
        Perform standardization by centering and scaling.

        Args:
            y (np.array): The data to transform.

        Returns:
            np.array: Transformed data.
        """
        if not self.is_fitted:
            self._load_params()

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        return self.scaler.transform(y)

    def inverse_transform(self, y):
        """
        Scale back the data to the original representation.
        Supports both numpy arrays and torch tensors.

        Args:
            y (np.array or torch.Tensor): The scaled data.

        Returns:
            np.array or torch.Tensor: The data in original scale.
        """
        if not self.is_fitted:
            self._load_params()

        is_tensor = torch.is_tensor(y)
        device = None

        if is_tensor:
            device = y.device
            y_np = y.detach().cpu().numpy()
        else:
            y_np = y

        # Ensure 2D for sklearn
        original_shape = y_np.shape
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)

        y_inv = self.scaler.inverse_transform(y_np)

        # Reshape back if input was 1D
        if len(original_shape) == 1:
            y_inv = y_inv.flatten()

        if is_tensor:
            return torch.from_numpy(y_inv).to(device)
        return y_inv

    def _save_params(self):
        """
        Save the scaler's mean and scale to disk using numpy.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.mean_path), exist_ok=True)

        np.save(self.mean_path, self.scaler.mean_)
        np.save(self.std_path, self.scaler.scale_)

    def _load_params(self):
        """
        Load the scaler's mean and scale from disk.
        """
        if os.path.exists(self.mean_path) and os.path.exists(self.std_path):
            mean = np.load(self.mean_path)
            scale = np.load(self.std_path)

            # Manually set the attributes of StandardScaler
            self.scaler.mean_ = mean
            self.scaler.scale_ = scale
            self.scaler.var_ = scale**2
            self.scaler.n_features_in_ = len(mean)
            self.scaler.n_samples_seen_ = 0  # Not strictly needed for transform

            self.is_fitted = True
        else:
            raise FileNotFoundError(
                f"Scaler parameters not found at {self.mean_path} or {self.std_path}. "
                "Please fit the scaler on training data first."
            )
