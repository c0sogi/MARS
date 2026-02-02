import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.geometry_utils import (
    parse_xyz,
    center_coordinates,
    get_pbc_distances,
    compute_local_context,
)
from library.feature_utils import compute_global_features
from library.utils import StandardScaler, transform_targets


class CrystalDataset(Dataset):
    """
    Dataset class for crystal structures implementing Multi-Scale Context extraction.
    Handles loading, caching, and preprocessing of atomic and global features.
    """

    def __init__(self, split, load_cached_data=True, sample_size=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load processed data from cache.
            sample_size (int, optional): Limit dataset size for debugging.
        """
        self.split = split
        self.sample_size = sample_size

        # Determine metadata path
        if split == "train":
            self.metadata_path = Config.TRAIN_METADATA
            self.cache_path = Config.TRAIN_DATA_CACHE
        elif split == "val":
            self.metadata_path = Config.VAL_METADATA
            self.cache_path = Config.VAL_DATA_CACHE
        elif split == "test":
            self.metadata_path = Config.TEST_METADATA
            self.cache_path = Config.TEST_DATA_CACHE
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load data
        self.atomic_features_list = []  # List of (N_atoms, 16) arrays
        self.global_features = None  # (N_samples, 12) array
        self.targets = None  # (N_samples, 2) array
        self.ids = None  # (N_samples,) array

        self._load_or_compute_data(load_cached_data)

        # Handle Scaling
        self.atomic_scaler = StandardScaler()
        self.global_scaler = StandardScaler()

        # Scaler paths
        self.atomic_scaler_path = os.path.join(Config.WORKING_DIR, "scalers_atomic.npz")
        self.global_scaler_path = os.path.join(Config.WORKING_DIR, "scalers_global.npz")

        if split == "train":
            self._fit_scalers()
        else:
            self._load_scalers()

        self._transform_data()

    def _load_or_compute_data(self, load_cached_data):
        """Loads data from cache or computes it from scratch."""
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading cached {self.split} data from {self.cache_path}...")
            try:
                data = np.load(self.cache_path, allow_pickle=True)
                # Reconstruct list of atomic features from flat array and counts
                flat_atomic = data["atomic_features_flat"]
                counts = data["atomic_counts"]
                self.global_features = data["global_features"]
                self.targets = data["targets"]
                self.ids = data["ids"]

                # Split flat array back into list
                cursor = 0
                self.atomic_features_list = []
                for count in counts:
                    self.atomic_features_list.append(
                        flat_atomic[cursor : cursor + count]
                    )
                    cursor += count

                print(f"Successfully loaded {len(self.ids)} samples.")
                return
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # Compute from scratch
        print(f"Computing {self.split} data from scratch...")
        df = pd.read_csv(self.metadata_path)

        if self.sample_size is not None:
            df = df.iloc[: self.sample_size]
            print(f"Debug: Limited to {self.sample_size} samples.")

        # 1. Compute Global Features
        self.global_features = compute_global_features(df)

        # 2. Extract Targets
        if self.split != "test":
            self.targets = df[Config.TARGET_COLS].values.astype(np.float32)
            # Apply log transformation
            self.targets = transform_targets(self.targets)
        else:
            # Placeholder for test set
            self.targets = np.zeros((len(df), Config.NUM_TARGETS), dtype=np.float32)

        self.ids = df["id"].values

        # 3. Compute Atomic Features
        self.atomic_features_list = []
        atomic_counts = []

        # Species mapping
        spec_map = {s: i for i, s in enumerate(Config.ATOMIC_SPECIES)}

        for idx, row in df.iterrows():
            file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

            # Parse geometry
            lattice, species, coords = parse_xyz(file_path)
            n_atoms = len(species)
            atomic_counts.append(n_atoms)

            # Center coordinates
            centered_coords = center_coordinates(coords)

            # Compute PBC distances
            distances = get_pbc_distances(coords, lattice)

            # Compute Local Contexts
            short_ctx, med_ctx, nn_dist = compute_local_context(distances, species)

            # One-hot encoding
            one_hot = np.zeros((n_atoms, Config.NUM_SPECIES), dtype=np.float32)
            for i, s in enumerate(species):
                if s in spec_map:
                    one_hot[i, spec_map[s]] = 1.0

            # Construct feature vector (16 dims)
            # [One-Hot (4) | Centered Coords (3) | NN Dist (1) | Short Ctx (4) | Med Ctx (4)]
            features = np.hstack(
                [one_hot, centered_coords, nn_dist, short_ctx, med_ctx]
            ).astype(np.float32)

            self.atomic_features_list.append(features)

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(df)} crystals...")

        # Save to cache
        Config.ensure_directories()

        # Flatten atomic features for storage
        if self.atomic_features_list:
            flat_atomic = np.vstack(self.atomic_features_list)
        else:
            flat_atomic = np.array([])

        np.savez(
            self.cache_path,
            atomic_features_flat=flat_atomic,
            atomic_counts=np.array(atomic_counts),
            global_features=self.global_features,
            targets=self.targets,
            ids=self.ids,
        )
        print(f"Saved processed data to {self.cache_path}")

    def _fit_scalers(self):
        """Fits StandardScalers on training data."""
        # Flatten all atomic features to fit scaler on continuous parts
        # Continuous features are indices 4 to 15 (12 dims)
        # 0-3 are one-hot (categorical)
        all_atomic = np.vstack(self.atomic_features_list)
        continuous_atomic = all_atomic[:, 4:]

        self.atomic_scaler.fit(continuous_atomic)
        self.atomic_scaler.save(self.atomic_scaler_path)

        self.global_scaler.fit(self.global_features)
        self.global_scaler.save(self.global_scaler_path)
        print("Scalers fitted and saved.")

    def _load_scalers(self):
        """Loads fitted scalers."""
        if os.path.exists(self.atomic_scaler_path) and os.path.exists(
            self.global_scaler_path
        ):
            self.atomic_scaler.load(self.atomic_scaler_path)
            self.global_scaler.load(self.global_scaler_path)
            print("Scalers loaded.")
        else:
            # Fallback if scalers don't exist (e.g. running test without train first)
            # This should ideally not happen in a proper pipeline
            print(
                "Warning: Scalers not found. Fitting on current data (suboptimal for test/val)."
            )
            self._fit_scalers()

    def _transform_data(self):
        """Applies scaling to continuous features."""
        # Transform Global Features
        self.global_features = self.global_scaler.transform(self.global_features)

        # Transform Atomic Features (only continuous parts)
        for i in range(len(self.atomic_features_list)):
            feats = self.atomic_features_list[i]
            # Copy categorical part
            cat_part = feats[:, :4]
            # Scale continuous part
            cont_part = self.atomic_scaler.transform(feats[:, 4:])
            # Reassemble
            self.atomic_features_list[i] = np.hstack([cat_part, cont_part]).astype(
                np.float32
            )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Returns:
            atomic_features (Tensor): [N_atoms, 16]
            global_features (Tensor): [12]
            target (Tensor): [2]
            crystal_id (int)
        """
        return (
            torch.tensor(self.atomic_features_list[idx], dtype=torch.float32),
            torch.tensor(self.global_features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            self.ids[idx],
        )


def sparse_collate(batch):
    """
    Custom collate function for Sparse Batching.
    Flattens atomic features from multiple crystals into a single tensor
    and creates a batch index vector.
    """
    atomic_features_list, global_features_list, targets_list, ids_list = zip(*batch)

    # 1. Flatten Atomic Features
    # Concatenate all atomic feature matrices along dimension 0
    batch_atomic_features = torch.cat(atomic_features_list, dim=0)

    # 2. Create Batch Index
    # Create a vector [0, 0, ..., 1, 1, ..., B-1, B-1] indicating crystal index
    batch_index_list = []
    for i, feats in enumerate(atomic_features_list):
        n_atoms = feats.shape[0]
        batch_index_list.append(torch.full((n_atoms,), i, dtype=torch.long))
    batch_index = torch.cat(batch_index_list, dim=0)

    # 3. Stack Global Features and Targets
    batch_global_features = torch.stack(global_features_list, dim=0)
    batch_targets = torch.stack(targets_list, dim=0)

    return {
        "atomic_features": batch_atomic_features,  # [Total_Batch_Atoms, 16]
        "batch_index": batch_index,  # [Total_Batch_Atoms]
        "global_features": batch_global_features,  # [Batch_Size, 12]
        "targets": batch_targets,  # [Batch_Size, 2]
        "ids": ids_list,  # List of IDs
    }


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
    sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    # Create Datasets
    train_dataset = CrystalDataset(
        "train", load_cached_data=load_cached_data, sample_size=sample_size
    )
    val_dataset = CrystalDataset(
        "val", load_cached_data=load_cached_data, sample_size=sample_size
    )
    test_dataset = CrystalDataset(
        "test", load_cached_data=load_cached_data, sample_size=sample_size
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sparse_collate,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sparse_collate,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sparse_collate,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
