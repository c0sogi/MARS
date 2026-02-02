import torch
from torch.utils.data import Dataset
import numpy as np
from library.config import Config
from library.data_utils import load_train_data, load_val_data, load_test_data


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.

    Loads preprocessed data using library.data_utils and serves:
    - Inputs: (Seq_Len, Channels) -> (107, 18)
    - Partner Indices: (Seq_Len,) -> (107,)
    - Targets: (Pred_Len, Num_Targets) -> (68, 5) [Train/Val only]
    - IDs: String identifiers
    """

    def __init__(self, mode="train", load_cached_data=True):
        """
        Args:
            mode (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): Whether to try loading from existing cache files.
        """
        super().__init__()
        self.mode = mode

        # Load data using the utility functions which handle caching and processing
        if mode == "train":
            data = load_train_data(load_cached_data=load_cached_data)
        elif mode == "val":
            data = load_val_data(load_cached_data=load_cached_data)
        elif mode == "test":
            data = load_test_data(load_cached_data=load_cached_data)
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'."
            )

        # Unpack data
        self.inputs = data["inputs"]
        self.partner_indices = data["partner_indices"]
        self.ids = data["ids"]

        # Targets are only available for train and val sets
        if mode in ["train", "val"]:
            self.targets = data["targets"]
        else:
            self.targets = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing the data for a single sample.
        """
        # Convert inputs to tensor
        # Shape: (Seq_Len, Input_Channels) -> (107, 18)
        input_tensor = torch.from_numpy(self.inputs[idx]).float()

        # Convert partner indices to tensor
        # Shape: (Seq_Len,) -> (107,)
        partner_index_tensor = torch.from_numpy(self.partner_indices[idx]).long()

        sample = {
            "inputs": input_tensor,
            "partner_indices": partner_index_tensor,
            "ids": self.ids[idx],
        }

        # Add targets if available
        if self.targets is not None:
            # Shape: (Pred_Len, Num_Targets) -> (68, 5)
            target_tensor = torch.from_numpy(self.targets[idx]).float()
            sample["targets"] = target_tensor

        return sample
