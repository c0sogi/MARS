import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from library.config import BATCH_SIZE, DEBUG_SAMPLE_SIZE, SEED
from library.feature_engineering import prepare_data
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(SEED)


class CrystalDataset(Dataset):
    """
    PyTorch Dataset for crystal structures.

    Attributes:
        atomic_features (list of np.ndarray): Variable length atomic feature matrices.
        global_features (np.ndarray): Fixed length global feature vectors.
        targets (np.ndarray): Target values (formation energy, bandgap).
        ids (np.ndarray): Sample identifiers.
    """

    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): Dictionary containing 'atomic', 'global', 'targets', and 'ids'.
        """
        self.atomic_features = data_dict["atomic"]
        self.global_features = data_dict["global"]
        self.targets = data_dict["targets"]
        self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Convert numpy arrays to torch tensors
        # Atomic features: (N_atoms, 8) -> FloatTensor
        atomic = torch.from_numpy(self.atomic_features[idx]).float()

        # Global features: (22,) -> FloatTensor
        glob = torch.from_numpy(self.global_features[idx]).float()

        # Targets: (2,) -> FloatTensor
        target = torch.from_numpy(self.targets[idx]).float()

        # ID: keep as is (int or str)
        sample_id = self.ids[idx]

        return atomic, glob, target, sample_id


def collate_fn(batch):
    """
    Custom collate function to handle variable number of atoms in a batch.

    Args:
        batch: List of tuples (atomic, global, target, id)

    Returns:
        dict: Batch dictionary with padded atomic features and masks.
    """
    atomic_list, global_list, target_list, id_list = zip(*batch)

    # 1. Pad atomic features to the max number of atoms in this batch
    # batch_first=True results in (Batch, Max_Atoms, Feat_Dim)
    atomic_padded = pad_sequence(atomic_list, batch_first=True, padding_value=0.0)

    # 2. Create a mask for the padded atoms (1 for real atom, 0 for padding)
    # Shape: (Batch, Max_Atoms)
    lengths = torch.tensor([a.shape[0] for a in atomic_list])
    max_len = atomic_padded.shape[1]
    mask = torch.arange(max_len)[None, :] < lengths[:, None]
    mask = mask.float()  # Convert boolean to float for multiplication if needed

    # 3. Stack other features
    global_tensor = torch.stack(global_list)
    target_tensor = torch.stack(target_list)

    # ids are usually kept as a list or numpy array for tracking
    ids = np.array(id_list)

    return {
        "atomic": atomic_padded,
        "atomic_mask": mask,
        "global": global_tensor,
        "targets": target_tensor,
        "ids": ids,
    }


def get_dataloaders(load_cached_data=True, batch_size=BATCH_SIZE):
    """
    Loads data, creates Datasets, and returns DataLoaders.

    Args:
        load_cached_data (bool): Whether to load pre-computed features from disk.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load processed data (features + targets)
    # This uses the caching logic implemented in feature_engineering.py
    data = prepare_data(load_cached_data=load_cached_data)

    # Subsample for debugging if configured
    if DEBUG_SAMPLE_SIZE is not None:
        print(f"DEBUG: Subsampling datasets to {DEBUG_SAMPLE_SIZE} samples.")
        for split in ["train", "val"]:
            n = min(len(data[split]["ids"]), DEBUG_SAMPLE_SIZE)
            data[split]["atomic"] = data[split]["atomic"][:n]
            data[split]["global"] = data[split]["global"][:n]
            data[split]["targets"] = data[split]["targets"][:n]
            data[split]["ids"] = data[split]["ids"][:n]
        # Test set is usually small, but we can slice it too if needed
        n_test = min(len(data["test"]["ids"]), DEBUG_SAMPLE_SIZE)
        data["test"]["atomic"] = data["test"]["atomic"][:n_test]
        data["test"]["global"] = data["test"]["global"][:n_test]
        data["test"]["targets"] = data["test"]["targets"][:n_test]
        data["test"]["ids"] = data["test"]["ids"][:n_test]

    # Create Datasets
    train_dataset = CrystalDataset(data["train"])
    val_dataset = CrystalDataset(data["val"])
    test_dataset = CrystalDataset(data["test"])

    # Create DataLoaders
    # Shuffle training data, but not validation/test
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    print(
        f"DataLoaders created. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
