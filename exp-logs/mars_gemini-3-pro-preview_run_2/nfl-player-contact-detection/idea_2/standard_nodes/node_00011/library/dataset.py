import torch
from torch.utils.data import Dataset
import numpy as np
from library.config import FEATURE_COLS, WINDOW_SIZE


class NFLSequenceDataset(Dataset):
    """
    PyTorch Dataset for handling temporal sequences of NFL contact data.

    Expects input X to be of shape (N_samples, Window_Size, N_features).
    Extracts the 'is_ground' feature specifically to support the Dual-Stream
    architecture's gating mechanism.
    """

    def __init__(self, X, y=None):
        """
        Args:
            X (np.ndarray): Feature array of shape (N, Window, Features).
            y (np.ndarray, optional): Target array of shape (N,).
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

        # Identify the index of the 'is_ground' feature to extract it separately
        try:
            self.is_ground_idx = FEATURE_COLS.index("is_ground")
        except ValueError:
            raise ValueError(
                "'is_ground' must be present in FEATURE_COLS configuration."
            )

        # Middle index for extracting static properties like is_ground
        self.mid_idx = WINDOW_SIZE // 2

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Extract the full sequence of features
        features = self.X[idx]

        # Extract is_ground flag.
        # It is constant across the window for a specific pair, so we take the middle frame.
        # Shape is scalar (0 or 1)
        is_ground = features[self.mid_idx, self.is_ground_idx]

        sample = {"features": features, "is_ground": is_ground}

        if self.y is not None:
            sample["target"] = self.y[idx]

        return sample
