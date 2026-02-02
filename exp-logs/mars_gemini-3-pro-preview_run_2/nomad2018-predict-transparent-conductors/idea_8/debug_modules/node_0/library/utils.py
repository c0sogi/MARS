import torch
import numpy as np
import random
import os


class GaussianSmearing(torch.nn.Module):
    """
    Expands interatomic distances into a set of Gaussian Radial Basis Functions (RBFs).
    This static edge feature expansion provides physical inductive bias to the CGCNN.
    """

    def __init__(
        self, start=0.0, stop=5.0, n_gaussians=50, centered=False, trainable=False
    ):
        super(GaussianSmearing, self).__init__()
        self.start = start
        self.stop = stop
        self.n_gaussians = n_gaussians

        # Compute centers (offsets) of the Gaussians
        offset = torch.linspace(start, stop, n_gaussians)

        # Compute the width (sigma) based on the spacing between centers
        # We set sigma equal to the step size.
        # The coefficient in exp(-coeff * (x - mu)^2) is 1 / (2 * sigma^2)
        # However, typical CGCNN implementations often use coeff = -0.5 / (step**2) inside the exp directly
        # or define gamma = 1/sigma^2.
        # Let's stick to a standard definition: exp(-(x - mu)^2 / (2*sigma^2))
        # If step = offset[1] - offset[0], let sigma = step.
        step = offset[1] - offset[0]
        self.coeff = -0.5 / (step**2)

        if trainable:
            self.offset = torch.nn.Parameter(offset)
            self.coeff = torch.nn.Parameter(torch.tensor(self.coeff))
        else:
            self.register_buffer("offset", offset)
            # Register coeff as a buffer so it moves with the model
            self.register_buffer("coeff_tensor", torch.tensor(self.coeff))

    def forward(self, dist):
        """
        Args:
            dist (torch.Tensor): Tensor of distances of shape (N,) or (N, 1)
        Returns:
            torch.Tensor: Tensor of RBF expansions of shape (N, n_gaussians)
        """
        # Ensure dist is at least 2D for broadcasting: (N, 1)
        if dist.dim() == 1:
            dist = dist.unsqueeze(-1)

        # Calculate squared difference: (dist - offset)^2
        # Broadcasting: (N, 1) - (n_gaussians,) -> (N, n_gaussians)
        diff = dist - self.offset

        # Compute Gaussian expansion
        return torch.exp(self.coeff_tensor * torch.pow(diff, 2))


class Standardizer:
    """
    Utility class for Z-score normalization of tensors.
    Can save and load its state (mean and std) using numpy formats to avoid pickle.
    """

    def __init__(self, mean=None, std=None, device="cpu"):
        self.mean = mean
        self.std = std
        self.device = device

        if self.mean is not None and isinstance(self.mean, torch.Tensor):
            self.mean = self.mean.to(self.device)
        if self.std is not None and isinstance(self.std, torch.Tensor):
            self.std = self.std.to(self.device)

    def fit(self, data):
        """
        Compute mean and standard deviation from the provided data.

        Args:
            data (torch.Tensor): Data to fit, shape (N, Features)
        """
        # Ensure data is on the correct device or move it temporarily
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        self.mean = torch.mean(data, dim=0).to(self.device)
        self.std = torch.std(data, dim=0).to(self.device)

        # Handle constant features (std = 0) to avoid division by zero
        # Replace 0 with 1 in std
        self.std[self.std == 0] = 1.0

    def transform(self, data):
        """
        Apply Z-score normalization: (data - mean) / std
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer must be fitted before calling transform.")

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        data = data.to(self.device)
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Revert Z-score normalization: (data * std) + mean
        """
        if self.mean is None or self.std is None:
            raise RuntimeError(
                "Standardizer must be fitted before calling inverse_transform."
            )

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        data = data.to(self.device)
        return (data * self.std) + self.mean

    def save(self, path):
        """
        Save the mean and std to a .npz file.

        Args:
            path (str): Path to save the file (e.g., 'scalers.npz')
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer has not been fitted, nothing to save.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Convert to numpy for saving
        mean_np = self.mean.detach().cpu().numpy()
        std_np = self.std.detach().cpu().numpy()

        np.savez(path, mean=mean_np, std=std_np)
        # print(f"Scaler saved to {path}")

    def load(self, path):
        """
        Load mean and std from a .npz file.

        Args:
            path (str): Path to the .npz file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at {path}")

        data = np.load(path)
        self.mean = torch.from_numpy(data["mean"]).float().to(self.device)
        self.std = torch.from_numpy(data["std"]).float().to(self.device)
        # print(f"Scaler loaded from {path}")


def set_seed(seed=42):
    """
    Set random seeds for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior in cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
