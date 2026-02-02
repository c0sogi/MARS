import torch
import numpy as np
from torch.utils.data import Dataset
from library import config, utils


class ContactSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapper for the Kinematic Center-Attention Network (K-CAN).

    This dataset takes the wide-format feature matrix (where temporal steps are flattened)
    and reshapes it into a temporal sequence tensor (Window, Features). It also
    separately extracts the features of the center frame (t=0) to support the
    explicit skip connection in the model architecture.
    """

    def __init__(self, X, y=None, ids=None):
        """
        Args:
            X (np.ndarray): Input features of shape (N, Window_Size * Num_Features).
                            The columns must be ordered temporally:
                            [Step_-5, Step_-4, ..., Step_0, ..., Step_+5].
            y (np.ndarray, optional): Target binary labels of shape (N,).
            ids (np.ndarray, optional): Unique contact IDs of shape (N,).
        """
        # Ensure reproducibility
        utils.set_seed()

        self.window_size = config.WINDOW_SIZE
        self.num_features = len(config.INPUT_FEATURES)

        # Validate input dimensions
        expected_width = self.window_size * self.num_features
        if X.shape[1] != expected_width:
            raise ValueError(
                f"Input dimension mismatch. Expected {expected_width} columns "
                f"({self.window_size} steps * {self.num_features} features), "
                f"but got {X.shape[1]}."
            )

        # Convert to torch tensors
        # Note: We keep data on CPU here to avoid GPU OOM with large datasets.
        # Data is moved to device in the training loop via DataLoader.
        self.X = torch.from_numpy(X).float()

        if y is not None:
            self.y = torch.from_numpy(y).float()
        else:
            self.y = None

        self.ids = ids

        # Calculate the index of the center frame (t=0)
        # Assuming window size is odd (e.g., 11), center is index 5 (0-10)
        self.center_idx = self.window_size // 2

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns:
            inputs (tuple):
                sequence (Tensor): Shape (Window_Size, Num_Features)
                center_features (Tensor): Shape (Num_Features) - Features at t=0
            label (Tensor, optional): Shape () - Binary target
        """
        # 1. Retrieve the flat window vector
        flat_window = self.X[idx]

        # 2. Reshape into temporal sequence (Window_Size, Num_Features)
        # The data processing pipeline concatenates steps horizontally in order.
        # view() works row-major, which aligns with [Step_-5_Feats, Step_-4_Feats, ...]
        sequence = flat_window.view(self.window_size, self.num_features)

        # 3. Extract Center Frame Features for Skip Connection
        # This forces the model to maintain strict focus on the event at t=0
        center_features = sequence[self.center_idx]

        # 4. Return
        # We package inputs as a tuple for the model's forward method
        inputs = (sequence, center_features)

        if self.y is not None:
            label = self.y[idx]
            return inputs, label
        else:
            return inputs
