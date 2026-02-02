import os
import torch
import numpy as np
from torch.utils.data import Dataset
from library.config import Config
from library.data_processing import process_dataset, PreprocessPipeline


class MaterialsDataset(Dataset):
    """
    PyTorch Dataset for material formation and bandgap energy prediction.
    Handles loading, scaling, and batching of atomic and global features.
    """

    def __init__(
        self, metadata_path, mode="train", max_samples=None, load_cached_data=True
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): One of 'train', 'val', 'test'. Determines scaling behavior.
            max_samples (int, optional): Limit the number of samples for debugging.
            load_cached_data (bool): Whether to load pre-processed data from cache.
        """
        self.mode = mode
        self.is_test = mode == "test"

        # 1. Load Data (Atomic features, Global features, Indices, Targets)
        # process_dataset handles caching of the raw extracted features.
        data_dict = process_dataset(
            metadata_path, load_cached_data=load_cached_data, is_test=self.is_test
        )

        raw_atomic = data_dict["atomic_features"]
        raw_global = data_dict["global_features"]
        raw_indices = data_dict["sample_indices"]
        ids = data_dict["ids"]
        targets = data_dict.get("targets")

        # 2. Handle max_samples (Slicing)
        if max_samples is not None and max_samples < len(ids):
            # Slice sample-level arrays
            ids = ids[:max_samples]
            raw_global = raw_global[:max_samples]
            if targets is not None:
                targets = targets[:max_samples]

            # Slice atom-level arrays
            # raw_indices are sorted (0,0,..,1,1.., etc.), so we find the cut point
            # Find the last atom index that belongs to sample (max_samples - 1)
            # np.searchsorted finds the first index where value would be inserted to maintain order.
            # We want the start of sample `max_samples`.
            atom_cut_idx = np.searchsorted(raw_indices, max_samples)

            raw_atomic = raw_atomic[:atom_cut_idx]
            raw_indices = raw_indices[:atom_cut_idx]

        self.ids = ids
        self.num_samples = len(self.ids)

        # 3. Scaling and Transformation
        self.scaler = PreprocessPipeline()
        scaler_path = os.path.join(Config.WORKING_DIR, Config.SCALERS_CACHE_FILE)

        if self.mode == "train":
            # Fit scalers on training data
            self.scaler.fit(raw_atomic, raw_global)
            # Save scalers for val/test usage
            self.scaler.save_scalers(scaler_path)
        else:
            # Load fitted scalers
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(
                    f"Scaler file not found at {scaler_path}. "
                    "Please run with mode='train' first to generate scalers."
                )
            self.scaler.load_scalers(scaler_path)

        # Apply transformations
        self.atomic_features, self.global_features = self.scaler.transform(
            raw_atomic, raw_global
        )

        if not self.is_test and targets is not None:
            self.targets = self.scaler.transform_targets(targets)
        else:
            self.targets = None

        # 4. Pre-calculate indices for fast __getitem__
        # raw_indices maps atom_k -> sample_i.
        # We need to know start_index and count for each sample_i.
        # Since raw_indices is sorted, we can use unique with counts.
        # Note: If max_samples sliced the data, raw_indices only contains up to max_samples-1.

        # Ensure we account for all samples even if some have 0 atoms (unlikely but safe)
        # np.bincount is faster for non-negative integers
        if len(raw_indices) > 0:
            counts = np.bincount(raw_indices, minlength=self.num_samples)
        else:
            counts = np.zeros(self.num_samples, dtype=int)

        self.atom_counts = counts
        self.atom_starts = np.zeros_like(counts)
        # Cumulative sum to get start indices
        if len(counts) > 0:
            self.atom_starts[1:] = np.cumsum(counts)[:-1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Retrieve atom slice
        start = self.atom_starts[idx]
        count = self.atom_counts[idx]
        end = start + count

        # Extract features
        # Convert to torch tensors here
        atomic_feats = torch.from_numpy(self.atomic_features[start:end])
        global_feats = torch.from_numpy(self.global_features[idx])
        sample_id = int(self.ids[idx])

        item = {
            "atomic_features": atomic_feats,
            "global_features": global_feats,
            "id": sample_id,
        }

        if self.targets is not None:
            item["targets"] = torch.from_numpy(self.targets[idx])

        return item


def collate_fn(batch):
    """
    Custom collate function for the MaterialsDataset.

    Args:
        batch: List of sample dictionaries from __getitem__.

    Returns:
        Dictionary containing batched tensors:
            - atomic_features: (Total_Atoms_In_Batch, D_atomic)
            - global_features: (Batch_Size, D_global)
            - batch_indices: (Total_Atoms_In_Batch,) - maps atoms to batch index
            - ids: (Batch_Size,)
            - targets: (Batch_Size, 2) [Optional]
    """
    atomic_feats_list = []
    global_feats_list = []
    batch_indices_list = []
    ids_list = []
    targets_list = []

    for i, sample in enumerate(batch):
        # Atomic features
        af = sample["atomic_features"]
        atomic_feats_list.append(af)

        # Create batch index vector for these atoms (all have index i)
        num_atoms = af.shape[0]
        batch_indices_list.append(torch.full((num_atoms,), i, dtype=torch.long))

        # Global features
        global_feats_list.append(sample["global_features"])

        # IDs
        ids_list.append(sample["id"])

        # Targets
        if "targets" in sample:
            targets_list.append(sample["targets"])

    # Concatenate / Stack
    batch_out = {
        "atomic_features": torch.cat(atomic_feats_list, dim=0),
        "global_features": torch.stack(global_feats_list, dim=0),
        "batch_indices": torch.cat(batch_indices_list, dim=0),
        "ids": torch.tensor(ids_list, dtype=torch.long),
    }

    if targets_list:
        batch_out["targets"] = torch.stack(targets_list, dim=0)

    return batch_out
