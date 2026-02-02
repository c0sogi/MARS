import torch
from torch.utils.data import Dataset
import numpy as np
from library.config import Config
from library.data_processor import DataProcessor


class RNADataset(Dataset):
    """
    PyTorch Dataset for the AHC-HIDN model.
    Wraps the DataProcessor to load processed numpy arrays and serves them as Tensors.
    """

    def __init__(self, mode="train", load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, attempts to load pre-processed .npz files.
                                     If False or missing, triggers reprocessing via DataProcessor.
        """
        self.mode = mode

        # Initialize processor and load data
        processor = DataProcessor()
        data = processor.process_data(mode=mode, load_cached_data=load_cached_data)

        # Store data in memory
        self.inputs = data["inputs"]
        self.partner_indices = data["partner_indices"]
        self.ids = data["ids"]

        # Targets are only present for train and val sets
        self.targets = data.get("targets", None)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing:
            - inputs: (Seq_Len, Channels) FloatTensor
            - partner_indices: (Seq_Len) LongTensor
            - targets: (Seq_Len, 5) FloatTensor (if available)
            - id: str
        """
        # Fetch numpy arrays
        input_arr = self.inputs[idx]
        partner_idx_arr = self.partner_indices[idx]
        sample_id = self.ids[idx]

        # Convert to PyTorch Tensors
        # Inputs are float32 (one-hot encodings)
        input_tensor = torch.from_numpy(input_arr).float()

        # Partner indices are int32, need Long for embedding/indexing
        partner_tensor = torch.from_numpy(partner_idx_arr).long()

        item = {
            "inputs": input_tensor,
            "partner_indices": partner_tensor,
            "id": sample_id,
        }

        # Add targets if they exist (train/val)
        if self.targets is not None:
            target_arr = self.targets[idx]
            target_tensor = torch.from_numpy(target_arr).float()
            item["targets"] = target_tensor

        return item
