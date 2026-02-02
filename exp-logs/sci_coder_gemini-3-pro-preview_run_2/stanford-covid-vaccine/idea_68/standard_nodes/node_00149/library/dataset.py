import torch
from torch.utils.data import Dataset
import numpy as np
from library import config, data_utils


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    Wraps the data loading logic from library.data_utils.
    """

    def __init__(self, split="train", load_cached_data=True, debug=False):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): Whether to load data from cache if available.
            debug (bool): Whether to run in debug mode (smaller subset).
        """
        self.split = split
        self.debug = debug

        # Load data using the provided utility
        # This handles caching, parsing, and preprocessing
        data_dict = data_utils.get_dataset(
            split=split, load_cached_data=load_cached_data, debug=debug
        )

        self.inputs = data_dict["inputs"]
        self.partner_indices = data_dict["partner_indices"]
        self.targets = data_dict["targets"]
        self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Returns:
            inputs (torch.FloatTensor): Shape (C_in, L) = (18, 107)
            partner_indices (torch.LongTensor): Shape (L,) = (107,)
            targets (torch.FloatTensor): Shape (C_out, L) = (5, 107)
        """
        # Fetch data
        # input shape: (107, 18) -> Needs transpose for Conv1d (Channels, Length)
        x = self.inputs[idx]

        # partner shape: (107,)
        p = self.partner_indices[idx]

        # target shape: (107, 5) -> Needs transpose to match output (Channels, Length)
        y = self.targets[idx]

        # Convert to Tensors
        # Transpose inputs to (Channels, Length)
        x_tensor = torch.from_numpy(x).float().permute(1, 0)

        # Partner indices stay as (Length,)
        p_tensor = torch.from_numpy(p).long()

        # Transpose targets to (Channels, Length)
        y_tensor = torch.from_numpy(y).float().permute(1, 0)

        return x_tensor, p_tensor, y_tensor
