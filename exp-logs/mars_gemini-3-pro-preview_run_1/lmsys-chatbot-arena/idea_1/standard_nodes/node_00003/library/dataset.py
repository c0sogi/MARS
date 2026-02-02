import numpy as np
import torch
from torch.utils.data import Dataset


class ArenaDataset(Dataset):
    """
    A PyTorch Dataset class for the Chatbot Arena task.
    It wraps pre-computed feature embeddings and target labels provided by the feature pipeline.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray = None):
        """
        Initializes the dataset by converting numpy arrays to PyTorch tensors.

        Args:
            features (np.ndarray): A numpy array of shape (num_samples, feature_dim)
                                   containing the pre-computed embeddings (concatenated prompt + responses).
            targets (np.ndarray, optional): A numpy array of shape (num_samples, num_classes)
                                            containing the target probabilities.
                                            Pass None for the test set.
        """
        super().__init__()

        # Convert features to FloatTensor
        self.features = torch.tensor(features, dtype=torch.float32)

        # Handle targets if they exist (Train/Val sets)
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.features)

    def __getitem__(self, index: int):
        """
        Retrieves the feature vector and target (if available) at the specified index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: (feature_vector, target_vector) if targets are provided.
            torch.Tensor: feature_vector if targets are None.
        """
        if self.targets is not None:
            return self.features[index], self.targets[index]
        else:
            return self.features[index]
