import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from library.config import Config
from library.preprocessing import prepare_data


class CrystalDataset(Dataset):
    """
    PyTorch Dataset for crystal structures.
    Wraps the processed data dictionaries containing atomic and global features.
    """

    def __init__(self, data_dict):
        """
        Args:
            data_dict: Dictionary containing:
                - 'atomic': List of np.ndarrays (N_atoms, D_atomic)
                - 'global': np.ndarray (N_samples, D_global)
                - 'targets': np.ndarray (N_samples, 2)
                - 'ids': np.ndarray (N_samples,)
        """
        self.atomic_features = data_dict["atomic"]
        self.global_features = data_dict["global"]
        self.targets = data_dict["targets"]
        self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Convert atomic features to FloatTensor
        # Shape: (N_atoms, 9)
        atomic = torch.from_numpy(self.atomic_features[idx]).float()

        # Global features to FloatTensor
        # Shape: (12,)
        glob = torch.from_numpy(self.global_features[idx]).float()

        # Targets to FloatTensor
        # Shape: (2,)
        target = torch.from_numpy(self.targets[idx]).float()

        # ID (keep as is, likely int or string)
        sample_id = self.ids[idx]

        return {"atomic": atomic, "global": glob, "target": target, "id": sample_id}


def collate_crystals(batch):
    """
    Collate function to handle variable-sized atomic sets in a batch.
    Pads atomic features and generates an attention mask.

    Args:
        batch: List of dictionaries from CrystalDataset.__getitem__

    Returns:
        batch_dict: Dictionary containing:
            - 'atomic': Padded atomic features (Batch, Max_N, D_atomic)
            - 'mask': Boolean mask (Batch, Max_N), True for real atoms
            - 'global': Global features (Batch, D_global)
            - 'target': Targets (Batch, 2)
            - 'id': List of IDs
    """
    atomic_list = [item["atomic"] for item in batch]
    global_list = [item["global"] for item in batch]
    target_list = [item["target"] for item in batch]
    id_list = [item["id"] for item in batch]

    # Pad atomic features
    # Result shape: (Batch, Max_N, Feature_Dim)
    atomic_padded = pad_sequence(atomic_list, batch_first=True, padding_value=0.0)

    # Create attention mask (True for real atoms, False for padding)
    # Shape: (Batch, Max_N)
    lengths = torch.tensor([len(a) for a in atomic_list])
    max_len = atomic_padded.shape[1]
    # Create indices [0, 1, ..., max_len-1] and compare with lengths
    # Broadcasting: (1, Max_N) < (Batch, 1) -> (Batch, Max_N)
    mask = torch.arange(max_len)[None, :] < lengths[:, None]

    # Stack global features and targets
    global_tensor = torch.stack(global_list)
    target_tensor = torch.stack(target_list)

    return {
        "atomic": atomic_padded,
        "mask": mask,
        "global": global_tensor,
        "target": target_tensor,
        "id": id_list,
    }


def get_dataloaders(batch_size=Config.BATCH_SIZE, load_cached=True):
    """
    Prepares data and returns DataLoaders for train, val, and test sets.
    Uses library.preprocessing.prepare_data for caching logic.

    Args:
        batch_size: Batch size for training/inference.
        load_cached: Whether to load pre-computed features from cache.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load processed data using the library function which handles caching
    train_data, val_data, test_data = prepare_data(load_cached=load_cached)

    # Instantiate Datasets
    train_dataset = CrystalDataset(train_data)
    val_dataset = CrystalDataset(val_data)
    test_dataset = CrystalDataset(test_data)

    # Instantiate DataLoaders
    # Using num_workers=2 for efficiency, pin_memory=True for faster GPU transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_crystals,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_crystals,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_crystals,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
