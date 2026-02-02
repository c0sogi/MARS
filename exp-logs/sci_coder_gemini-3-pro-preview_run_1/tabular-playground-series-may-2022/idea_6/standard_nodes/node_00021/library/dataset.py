import torch
import numpy as np
from torch.utils.data import Dataset


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Granular Unified Transformer model.
    Wraps preprocessed numerical and sequence features.
    """

    def __init__(self, x_num: np.ndarray, x_seq: np.ndarray, y: np.ndarray = None):
        """
        Initialize the dataset.

        Args:
            x_num (np.ndarray): Numerical features array of shape (N, num_features).
                                Expected to be float32.
            x_seq (np.ndarray): Sequence features array of shape (N, seq_len).
                                Expected to be int32 or int64 (indices).
            y (np.ndarray, optional): Target labels array of shape (N,).
                                      Expected to be float32. Defaults to None.
        """
        # Convert numpy arrays to tensors immediately to avoid overhead during iteration
        # x_num is float32 for dense layers
        self.x_num = torch.tensor(x_num, dtype=torch.float32)

        # x_seq is long (int64) for embedding lookups
        self.x_seq = torch.tensor(x_seq, dtype=torch.long)

        # y is float32 for Binary Cross Entropy
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

        # Validation to ensure lengths match
        assert len(self.x_num) == len(
            self.x_seq
        ), "Mismatch between numerical and sequence data lengths."
        if self.y is not None:
            assert len(self.x_num) == len(
                self.y
            ), "Mismatch between features and target lengths."

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        """
        return len(self.x_num)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'x_num': Tensor of numerical features.
                - 'x_seq': Tensor of sequence indices.
                - 'target': Tensor of the label (if available).
        """
        item = {"x_num": self.x_num[idx], "x_seq": self.x_seq[idx]}

        if self.y is not None:
            # Unsqueeze to create shape (1,) for compatibility with BCEWithLogitsLoss
            item["target"] = self.y[idx].unsqueeze(0)

        return item
