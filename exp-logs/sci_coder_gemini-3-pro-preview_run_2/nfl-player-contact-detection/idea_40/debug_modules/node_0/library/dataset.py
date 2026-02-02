import torch
from torch.utils.data import Dataset
import numpy as np
from library.config import Config


class ContactDataset(Dataset):
    """
    PyTorch Dataset for the LRP-Net pipeline.

    Handles the serving of wide feature tensors and implements Stochastic Noise Injection
    on kinematic features during the training phase to force manifold learning.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray = None, training: bool = False):
        """
        Args:
            X (np.ndarray): The feature matrix (float32).
            y (np.ndarray, optional): The target vector. Defaults to None.
            training (bool): Flag to enable stochastic noise injection. Defaults to False.
        """
        super().__init__()

        # Convert inputs to tensors
        # We keep them on CPU to avoid VRAM saturation with the full dataset
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float() if y is not None else None
        self.training = training

        # Calculate the number of kinematic features to apply noise to.
        # Structure in features.py:
        # [Kinematic Cols (Windowed)] + [Derived Cols] + [Visual Cols]
        # We only want to perturb the raw windowed kinematic inputs.

        num_base_kinematic = len(Config.KINEMATIC_COLS)
        window_span = (
            Config.WINDOW_SIZE * 2 + 1
        )  # e.g., 5 past + current + 5 future = 11
        num_players = 2  # p1 and p2

        self.num_kinematic_features = num_base_kinematic * window_span * num_players

        # Pre-validate that X has enough columns
        if self.X.shape[1] < self.num_kinematic_features:
            raise ValueError(
                f"Feature matrix width ({self.X.shape[1]}) is smaller than the calculated "
                f"kinematic feature count ({self.num_kinematic_features}). Check feature generation."
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Clone the feature vector to ensure we don't modify the original dataset in memory
        # when applying noise.
        x = self.X[idx].clone()

        if self.training:
            # Stochastic Noise Injection
            # Apply Gaussian noise N(0, sigma) only to the kinematic features
            # This prevents the model from memorizing sensor noise and forces it to learn
            # robust physical manifolds.
            noise = torch.randn(self.num_kinematic_features) * Config.INPUT_NOISE_SIGMA
            x[: self.num_kinematic_features] += noise

        if self.y is not None:
            return x, self.y[idx]
        else:
            return x
