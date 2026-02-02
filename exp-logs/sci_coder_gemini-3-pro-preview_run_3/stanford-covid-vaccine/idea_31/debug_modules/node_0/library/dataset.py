import torch
import numpy as np
from torch.utils.data import Dataset
from library.config import Config
from library.preprocessing import RNAPreprocessor


class RNADataset(Dataset):
    """
    PyTorch Dataset for the RNA degradation prediction task.
    Wraps the RNAPreprocessor to load and serve data tensors.
    """

    def __init__(self, split="train", load_cached_data=True):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): Whether to load preprocessed data from cache.
        """
        self.split = split

        # Initialize preprocessor and load data
        preprocessor = RNAPreprocessor()
        data = preprocessor.process_data(split=split, load_cached_data=load_cached_data)

        # Store data as class attributes
        self.ids = data["ids"]
        self.inputs = data["inputs"]
        self.pair_indices = data["pair_indices"]

        # Targets are only available for train and val splits
        if "targets" in data:
            self.targets = data["targets"]
        else:
            self.targets = None

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary containing:
                - 'input': Tensor of shape (SeqLen, 14)
                - 'pair_indices': LongTensor of shape (SeqLen,)
                - 'id': String identifier
                - 'target': Tensor of shape (PredLen, 5) (if available)
        """
        # Convert inputs to float32 tensor
        input_tensor = torch.from_numpy(self.inputs[idx]).float()

        # Convert pair indices to long tensor (for indexing/embedding)
        pair_indices_tensor = torch.from_numpy(self.pair_indices[idx]).long()

        sample = {
            "input": input_tensor,
            "pair_indices": pair_indices_tensor,
            "id": self.ids[idx],
        }

        # Add targets if they exist
        if self.targets is not None:
            target_tensor = torch.from_numpy(self.targets[idx]).float()
            sample["target"] = target_tensor

        return sample
