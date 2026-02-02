import torch
from torch.utils.data import Dataset
import numpy as np
from library.config import Config
from library.data_utils import process_data


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.

    Loads preprocessed data containing:
    1. Static Inputs: Sequence, Structure, Loop Type, Partner Identity (One-Hot).
    2. Partner Indices: Mapping of paired bases for the Partner-Aware architecture.
    3. Targets: Ground truth degradation values.

    The dataset relies on library.data_utils.process_data for caching and loading
    .npz files to/from ./working/idea_31/.
    """

    def __init__(self, mode="train", load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load from .npz cache if available.
        """
        super().__init__()
        self.mode = mode

        # Load data using the provided utility function
        # This function handles the caching logic (checking ./working/idea_31/)
        data_dict = process_data(mode, load_cached_data=load_cached_data)

        self.inputs = data_dict["inputs"]
        self.targets = data_dict["targets"]
        self.ids = data_dict["ids"]
        self.partner_indices = data_dict["partner_indices"]

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Retrieves a single sample.

        Returns:
            inputs (torch.Tensor): Static features (SeqLen, 18).
            partner_indices (torch.Tensor): Partner map (SeqLen,).
            targets (torch.Tensor): Target values (SeqLen, 5).
            sample_id (str): The sample ID.
        """
        # Retrieve numpy arrays
        input_arr = self.inputs[idx]
        target_arr = self.targets[idx]
        partner_idx_arr = self.partner_indices[idx]
        sample_id = self.ids[idx]

        # Convert to PyTorch Tensors
        # Inputs and Targets are float32
        inputs_tensor = torch.from_numpy(input_arr).float()
        targets_tensor = torch.from_numpy(target_arr).float()

        # Partner indices are int32/int64 (indices)
        partner_indices_tensor = torch.from_numpy(partner_idx_arr).long()

        return inputs_tensor, partner_indices_tensor, targets_tensor, sample_id
