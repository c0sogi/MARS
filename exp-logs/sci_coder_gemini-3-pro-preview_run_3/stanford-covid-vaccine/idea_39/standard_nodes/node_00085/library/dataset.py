import torch
import numpy as np
from torch.utils.data import Dataset
from library.data_utils import load_or_process_data


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.

    This dataset wraps the preprocessed numpy arrays and provides them as
    PyTorch tensors. It handles both training/validation data (with targets)
    and test data (without targets).
    """

    def __init__(self, data, config, is_test=False):
        """
        Args:
            data (dict): Dictionary containing 'inputs', 'bpp_indices', 'bpp_mask',
                         'ids', and optionally 'targets'.
            config (Config): Configuration object.
            is_test (bool): Flag indicating if this is the test set (no targets).
        """
        self.inputs = data["inputs"]
        self.bpp_indices = data["bpp_indices"]
        self.bpp_mask = data["bpp_mask"]
        self.ids = data["ids"]
        self.is_test = is_test
        self.config = config

        if not self.is_test:
            self.targets = data["targets"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Retrieves a sample at the given index.

        Returns:
            dict: Dictionary containing:
                - 'inputs': FloatTensor (seq_len, input_dim)
                - 'bpp_indices': LongTensor (seq_len)
                - 'bpp_mask': FloatTensor (seq_len)
                - 'ids': str
                - 'targets': FloatTensor (pred_len, num_classes) [Only if not is_test]
        """
        # Fetch data arrays
        input_feat = self.inputs[idx]
        bpp_idx = self.bpp_indices[idx]
        bpp_m = self.bpp_mask[idx]

        # Convert to Tensors
        # Inputs: (Seq_Len, Channels) -> Float32
        input_tensor = torch.tensor(input_feat, dtype=torch.float32)

        # BPP Indices: (Seq_Len,) -> Long (Used for gathering neighbor states)
        bpp_idx_tensor = torch.tensor(bpp_idx, dtype=torch.long)

        # BPP Mask: (Seq_Len,) -> Float32 (Used for strict output masking)
        bpp_mask_tensor = torch.tensor(bpp_m, dtype=torch.float32)

        sample = {
            "inputs": input_tensor,
            "bpp_indices": bpp_idx_tensor,
            "bpp_mask": bpp_mask_tensor,
            "ids": str(self.ids[idx]),
        }

        if not self.is_test:
            # Targets: (Pred_Len, Num_Classes) -> Float32
            target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)
            sample["targets"] = target_tensor

        return sample


def get_dataset(data_type, config, load_cached_data=True):
    """
    Factory function to load data and create an RNADataset instance.

    This function leverages the caching mechanism in `library.data_utils`.

    Args:
        data_type (str): 'train', 'val', or 'test'.
        config (Config): Configuration object.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        RNADataset: The instantiated dataset ready for a DataLoader.
    """
    # Use the provided utility to handle caching, processing, and loading
    data = load_or_process_data(data_type, config, load_cached_data=load_cached_data)

    # Determine if it's a test set based on the data type
    is_test = data_type == "test"

    return RNADataset(data, config, is_test=is_test)
