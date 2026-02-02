import torch
from torch.utils.data import Dataset
import numpy as np


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control task.

    This class wraps preprocessed numpy arrays for categorical and continuous features,
    as well as optional targets. It converts these arrays into PyTorch tensors
    suitable for model training and inference.
    """

    def __init__(self, cat_data, cont_data, targets=None):
        """
        Args:
            cat_data (np.ndarray): Numpy array containing categorical features (int64).
            cont_data (np.ndarray): Numpy array containing normalized continuous features (float32).
            targets (np.ndarray, optional): Numpy array containing target labels (float32).
                                           If None, the dataset is in inference mode.
        """
        self.cat_data = cat_data
        self.cont_data = cont_data
        self.targets = targets

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.cat_data)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple:
                - cat_features (torch.LongTensor): Categorical features.
                - cont_features (torch.FloatTensor): Continuous features.
                - target (torch.FloatTensor): Target label (only if targets were provided).
        """
        # Convert categorical features to LongTensor for embedding lookups
        cat_features = torch.tensor(self.cat_data[idx], dtype=torch.long)

        # Convert continuous features to FloatTensor for dense layers
        cont_features = torch.tensor(self.cont_data[idx], dtype=torch.float32)

        if self.targets is not None:
            # Convert target to FloatTensor for loss calculation
            # Note: Depending on the loss function, the training loop might need to
            # unsqueeze this to shape (1,) or match the model output shape.
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return cat_features, cont_features, target

        # For inference (test set), return only features
        return cat_features, cont_features
