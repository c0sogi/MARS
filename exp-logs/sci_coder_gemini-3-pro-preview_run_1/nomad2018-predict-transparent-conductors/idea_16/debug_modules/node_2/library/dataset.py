import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
import os
from library import config
from library.features import MaterialFeatureExtractor, DataScaler


class MaterialsDataset(Dataset):
    """
    PyTorch Dataset for materials data.
    Stores preprocessed atomic and global features along with targets.
    """

    def __init__(self, atomic_features, global_features, targets, ids):
        """
        Args:
            atomic_features (list of np.ndarray): List of (N_atoms, Feature_Dim) arrays.
            global_features (np.ndarray): Array of (N_samples, Global_Dim).
            targets (np.ndarray): Array of (N_samples, 2) targets.
            ids (np.ndarray): Array of sample IDs.
        """
        self.atomic_features = atomic_features
        self.global_features = global_features
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Convert to torch tensors
        # Atomic features are variable length, so we keep them as 2D tensors here
        # and pad them in the collate function.
        atom_feat = torch.from_numpy(self.atomic_features[idx]).float()
        glob_feat = torch.from_numpy(self.global_features[idx]).float()
        target = torch.from_numpy(self.targets[idx]).float()
        sample_id = self.ids[idx]

        return {
            "atomic_features": atom_feat,
            "global_features": glob_feat,
            "target": target,
            "id": sample_id,
        }


def collate_batch(batch):
    """
    Custom collate function to handle variable number of atoms.
    Pads atomic features and creates a mask.
    """
    atomic_features_list = [item["atomic_features"] for item in batch]
    global_features_list = [item["global_features"] for item in batch]
    targets_list = [item["target"] for item in batch]
    ids_list = [item["id"] for item in batch]

    # Pad atomic features
    # batch_first=True -> (Batch, Max_N_Atoms, Feature_Dim)
    padded_atomic_features = pad_sequence(
        atomic_features_list, batch_first=True, padding_value=0.0
    )

    # Create mask (1 for real atom, 0 for padding)
    # Shape: (Batch, Max_N_Atoms)
    lengths = torch.tensor([feat.shape[0] for feat in atomic_features_list])
    max_len = padded_atomic_features.shape[1]
    mask = torch.arange(max_len)[None, :] < lengths[:, None]
    mask = mask.float()  # Convert to float for multiplication if needed

    # Stack other features
    stacked_global_features = torch.stack(global_features_list)
    stacked_targets = torch.stack(targets_list)
    # IDs are usually just passed as a list or tensor of ints
    stacked_ids = torch.tensor(ids_list, dtype=torch.int32)

    return {
        "atomic_features": padded_atomic_features,
        "global_features": stacked_global_features,
        "mask": mask,
        "targets": stacked_targets,
        "ids": stacked_ids,
    }


def get_dataloader(
    split_name, batch_size=config.BATCH_SIZE, shuffle=True, load_cached_data=True
):
    """
    Creates a DataLoader for the specified split.

    Args:
        split_name (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    # Determine metadata path
    if split_name == "train":
        meta_path = config.TRAIN_CSV
    elif split_name == "val":
        meta_path = config.VAL_CSV
    elif split_name == "test":
        meta_path = config.TEST_CSV
    else:
        raise ValueError(f"Unknown split_name: {split_name}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path)

    # Initialize Extractor and Scaler
    extractor = MaterialFeatureExtractor()
    scaler = DataScaler()

    # Process Data
    # The extractor handles fitting the scaler if split is 'train',
    # or loading the scaler if split is 'val'/'test'.
    data_dict = extractor.process_data(
        df, split_name=split_name, load_cached_data=load_cached_data, scaler=scaler
    )

    # Create Dataset
    dataset = MaterialsDataset(
        atomic_features=data_dict["atomic_features"],
        global_features=data_dict["global_features"],
        targets=data_dict["targets"],
        ids=data_dict["ids"],
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
        num_workers=2,  # Use a few workers for data loading
        pin_memory=True,
    )

    return dataloader
