import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config
from library.data_utils import process_data


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    Yields a dictionary containing hybrid input features, partner map, and targets.
    """

    def __init__(self, inputs, partner_map, targets=None, ids=None):
        """
        Args:
            inputs (np.ndarray): Shape (N, 107, 18). Hybrid features.
            partner_map (np.ndarray): Shape (N, 107). Indices of paired bases.
            targets (np.ndarray, optional): Shape (N, 68, 5). Ground truth values.
            ids (list, optional): List of sample IDs.
        """
        self.inputs = inputs
        self.partner_map = partner_map
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert inputs to float32 tensor
        # inputs shape: (107, 18)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # Convert partner map to long tensor
        # partner_map shape: (107,)
        p_map = torch.tensor(self.partner_map[idx], dtype=torch.long)

        sample = {"inputs": x, "partner_map": p_map}

        # Add targets if available (Training/Validation)
        if self.targets is not None:
            # targets shape: (68, 5)
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            sample["targets"] = y

        # Add ID if available (Inference)
        if self.ids is not None:
            sample["id"] = self.ids[idx]

        return sample


def get_dataloader(
    split,
    batch_size=Config.BATCH_SIZE,
    shuffle=None,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
):
    """
    Creates a DataLoader for the specified data split.

    Args:
        split (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data. Defaults to True for train, False otherwise.
        num_workers (int): Number of subprocesses for data loading.
        debug (bool): Whether to load a small subset for debugging.

    Returns:
        DataLoader: A PyTorch DataLoader instance.
    """
    # Default shuffle logic
    if shuffle is None:
        shuffle = split == "train"

    # Load processed data using the library utility (handles caching)
    inputs, partner_map, targets, ids = process_data(
        data_type=split, load_cached_data=True, debug=debug
    )

    # Instantiate the dataset
    dataset = RNADataset(inputs, partner_map, targets, ids)

    # Create the DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=(
            split == "train"
        ),  # Drop last incomplete batch during training for stability
    )

    return loader
