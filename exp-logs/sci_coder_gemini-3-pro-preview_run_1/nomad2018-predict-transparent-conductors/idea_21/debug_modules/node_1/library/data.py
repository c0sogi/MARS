import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config
from library.features import process_dataset, get_scalers, load_scalers, scale_features


class CrystalDataset(Dataset):
    """
    PyTorch Dataset for crystal structures.
    """

    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): Dictionary containing processed data arrays.
                              Keys: 'atomic_inputs', 'global_inputs', 'batch_indices', 'targets', 'ids'
        """
        # Convert numpy arrays to torch tensors
        self.atomic_inputs = torch.from_numpy(data_dict["atomic_inputs"]).float()
        self.global_inputs = torch.from_numpy(data_dict["global_inputs"]).float()
        self.targets = torch.from_numpy(data_dict["targets"]).float()
        self.ids = torch.from_numpy(data_dict["ids"]).long()

        # Pre-compute atom slices for each crystal
        # The batch_indices array contains the sample index for each atom (0, 0, ..., 1, 1, ...)
        # We assume the data is sorted by sample index, which is guaranteed by process_dataset
        batch_indices = data_dict["batch_indices"]

        # Get counts of atoms for each crystal
        # np.unique returns sorted unique elements and their counts
        # Since indices are 0..N-1, counts[i] corresponds to sample i
        _, counts = np.unique(batch_indices, return_counts=True)

        self.atom_counts = counts

        # Compute start indices for slicing
        # starts[0] = 0, starts[1] = counts[0], starts[2] = counts[0] + counts[1], ...
        self.atom_starts = np.zeros(len(counts) + 1, dtype=np.int64)
        self.atom_starts[1:] = np.cumsum(counts)

    def __len__(self):
        return len(self.global_inputs)

    def __getitem__(self, idx):
        # Slice atomic features for this crystal
        start_idx = self.atom_starts[idx]
        end_idx = self.atom_starts[idx + 1]

        atomic_feats = self.atomic_inputs[start_idx:end_idx]
        global_feats = self.global_inputs[idx]
        target = self.targets[idx]
        sample_id = self.ids[idx]

        return atomic_feats, global_feats, target, sample_id


class CollateFn:
    """
    Custom collate function to batch variable-sized crystal graphs (point clouds).
    """

    def __call__(self, batch):
        """
        Args:
            batch: List of tuples (atomic_feats, global_feats, target, sample_id)
        """
        # Unzip the batch
        atomic_feats_list, global_feats_list, targets_list, ids_list = zip(*batch)

        # 1. Concatenate atomic features into a single large tensor (N_total_atoms, Feature_Dim)
        batch_atomic_feats = torch.cat(atomic_feats_list, dim=0)

        # 2. Create batch indices vector for scatter operations
        # This tells the model which crystal each atom belongs to in the batch
        batch_indices_list = []
        for i, feats in enumerate(atomic_feats_list):
            n_atoms = feats.shape[0]
            # Create a tensor of size n_atoms filled with index i
            batch_indices_list.append(torch.full((n_atoms,), i, dtype=torch.long))

        batch_indices = torch.cat(batch_indices_list, dim=0)

        # 3. Stack global features (Batch_Size, Global_Dim)
        batch_global_feats = torch.stack(global_feats_list, dim=0)

        # 4. Stack targets (Batch_Size, 2)
        batch_targets = torch.stack(targets_list, dim=0)

        # 5. Stack IDs (Batch_Size,)
        batch_ids = torch.stack(ids_list, dim=0)

        return {
            "atomic_feats": batch_atomic_feats,
            "batch_indices": batch_indices,
            "global_feats": batch_global_feats,
            "targets": batch_targets,
            "ids": batch_ids,
        }


def get_train_val_loaders(
    load_cached_data=True, batch_size=Config.BATCH_SIZE, num_workers=2
):
    """
    Loads training and validation data, fits scalers, and returns DataLoaders.
    """
    # 1. Process or Load Data
    # process_dataset handles caching internally
    train_data = process_dataset(
        Config.METADATA_TRAIN_PATH, load_cached_data=load_cached_data, mode="train"
    )
    val_data = process_dataset(
        Config.METADATA_VAL_PATH, load_cached_data=load_cached_data, mode="val"
    )

    # 2. Fit Scalers on Training Data
    # This also saves them to disk for inference
    atomic_scaler, global_scaler = get_scalers(train_data)

    # 3. Scale Features
    train_data_scaled = scale_features(train_data, atomic_scaler, global_scaler)
    val_data_scaled = scale_features(val_data, atomic_scaler, global_scaler)

    # 4. Create Datasets
    train_dataset = CrystalDataset(train_data_scaled)
    val_dataset = CrystalDataset(val_data_scaled)

    # 5. Create DataLoaders
    collate_fn = CollateFn()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True, batch_size=Config.BATCH_SIZE, num_workers=2):
    """
    Loads test data, loads pre-fitted scalers, and returns DataLoader.
    """
    # 1. Process or Load Data
    test_data = process_dataset(
        Config.METADATA_TEST_PATH, load_cached_data=load_cached_data, mode="test"
    )

    # 2. Load Scalers (Must have been fit during training)
    try:
        atomic_scaler, global_scaler = load_scalers()
    except FileNotFoundError:
        print(
            "Warning: Scalers not found. Ensure training has been run before inference."
        )
        # Fallback or re-raise depending on strictness. Here we re-raise to be safe.
        raise

    # 3. Scale Features
    test_data_scaled = scale_features(test_data, atomic_scaler, global_scaler)

    # 4. Create Dataset
    test_dataset = CrystalDataset(test_data_scaled)

    # 5. Create DataLoader
    collate_fn = CollateFn()

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader
