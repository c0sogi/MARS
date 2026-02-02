import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import ase.io

import library.config as lib_config
from library.config import (
    INPUT_DIR,
    ATOM_FEATURES_DIM,
    GLOBAL_FEATURES_DIM,
    SEED,
)
from library.features import extract_atomic_features, extract_global_features

# Set seeds for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)


def process_data(df, cache_path, load_cached_data=True):
    """
    Processes raw data into features and targets, with caching.

    Args:
        df: Pandas DataFrame containing metadata (id, file_path, targets).
        cache_path: Path to save/load the .npz cache file.
        load_cached_data: Boolean, whether to attempt loading from cache.

    Returns:
        atomic_features: List of np.arrays (N_atoms, Atom_Dim)
        global_features: np.array (N_samples, Global_Dim)
        targets: np.array (N_samples, 2)
        ids: np.array (N_samples,)
    """
    # Adjust for debug mode
    if lib_config.DEBUG_SAMPLE_SIZE is not None:
        cache_path = cache_path.replace(
            ".npz", f"_debug_{lib_config.DEBUG_SAMPLE_SIZE}.npz"
        )
        if len(df) > lib_config.DEBUG_SAMPLE_SIZE:
            df = df.iloc[: lib_config.DEBUG_SAMPLE_SIZE]

    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached data from {cache_path}...")
            data = np.load(cache_path, allow_pickle=True)
            atomic_features = list(data["atomic_features"])
            global_features = data["global_features"]
            targets = data["targets"]
            ids = data["ids"]
            print(f"Loaded {len(ids)} samples from cache.")
            return atomic_features, global_features, targets, ids
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Compute from scratch
    print(f"Processing {len(df)} samples...")

    atomic_features_list = []
    global_features_list = []
    targets_list = []
    ids_list = []

    for idx, row in df.iterrows():
        # Construct full file path
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Load atomic structure using ASE
        try:
            atoms = ase.io.read(full_path, format="aims")
        except Exception as e:
            print(f"Error reading {full_path}: {e}")
            continue

        # Extract features using library functions
        af = extract_atomic_features(atoms)
        gf = extract_global_features(atoms)

        atomic_features_list.append(af)
        global_features_list.append(gf)
        ids_list.append(row["id"])

        # Extract targets if available (NaN for test set)
        if "formation_energy_ev_natom" in row and "bandgap_energy_ev" in row:
            t = [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
        else:
            t = [np.nan, np.nan]
        targets_list.append(t)

    # Convert to numpy arrays
    # Atomic features are variable length -> object array
    atomic_features_arr = np.array(atomic_features_list, dtype=object)
    global_features_arr = np.array(global_features_list, dtype=np.float32)
    targets_arr = np.array(targets_list, dtype=np.float32)
    ids_arr = np.array(ids_list, dtype=np.int32)

    # Save to cache
    print(f"Saving processed data to {cache_path}...")
    np.savez(
        cache_path,
        atomic_features=atomic_features_arr,
        global_features=global_features_arr,
        targets=targets_arr,
        ids=ids_arr,
    )

    return atomic_features_list, global_features_arr, targets_arr, ids_arr


def get_scalers(atomic_features, global_features):
    """
    Fits StandardScalers on atomic and global features.

    Args:
        atomic_features: List of np.arrays of atomic features.
        global_features: np.array of global features.

    Returns:
        scaler_atomic, scaler_global
    """
    print("Fitting scalers...")

    # Atomic: Flatten all atoms from all crystals to fit scaler
    # atomic_features is a list of (N_atoms, Atom_Dim) arrays
    all_atomic = np.vstack(atomic_features)
    scaler_atomic = StandardScaler()
    scaler_atomic.fit(all_atomic)

    # Global: Standard scaling across samples
    scaler_global = StandardScaler()
    scaler_global.fit(global_features)

    return scaler_atomic, scaler_global


class CrystalDataset(Dataset):
    def __init__(
        self,
        atomic_features,
        global_features,
        targets,
        ids,
        scaler_atomic=None,
        scaler_global=None,
        mode="train",
    ):
        """
        PyTorch Dataset for crystal structures.

        Args:
            atomic_features: List of np.arrays (N_atoms, F_atomic)
            global_features: np.array (N_samples, F_global)
            targets: np.array (N_samples, 2)
            ids: np.array (N_samples,)
            scaler_atomic: Fitted StandardScaler for atomic features
            scaler_global: Fitted StandardScaler for global features
            mode: 'train', 'val', or 'test'
        """
        self.atomic_features = atomic_features
        self.global_features = global_features
        self.targets = targets
        self.ids = ids
        self.mode = mode

        # Apply scaling immediately to store scaled data in memory
        if scaler_atomic:
            # Transform each crystal's atomic features individually
            self.atomic_features = [
                scaler_atomic.transform(af).astype(np.float32)
                for af in self.atomic_features
            ]

        if scaler_global:
            self.global_features = scaler_global.transform(self.global_features).astype(
                np.float32
            )

        # Apply Log(1+y) transformation to targets for training and validation
        # This aligns the training loss (MSE) with the evaluation metric (RMSLE)
        if mode in ["train", "val"]:
            self.targets = np.log1p(self.targets)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Returns raw tensors. Collate will handle batching.
        af = torch.tensor(self.atomic_features[idx], dtype=torch.float32)
        gf = torch.tensor(self.global_features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        sample_id = int(self.ids[idx])

        return af, gf, target, sample_id


def collate_sparse(batch):
    """
    Collates samples into a sparse batch (flattened atomic features).

    Args:
        batch: List of tuples (atomic_feats, global_feats, target, sample_id)

    Returns:
        Dictionary containing:
            - atomic_features: (Total_Atoms, Atom_Dim)
            - batch_index: (Total_Atoms,) indicating which crystal each atom belongs to
            - global_features: (Batch_Size, Global_Dim)
            - targets: (Batch_Size, 2)
            - ids: (Batch_Size,)
    """
    atomic_feats_list = []
    batch_indices_list = []
    global_feats_list = []
    targets_list = []
    ids_list = []

    for i, (af, gf, target, sample_id) in enumerate(batch):
        n_atoms = af.shape[0]

        # Atomic features
        atomic_feats_list.append(af)

        # Batch index (repeats 'i' for each atom in this crystal)
        # This allows scatter operations to pool atoms back to their crystal
        batch_indices_list.append(torch.full((n_atoms,), i, dtype=torch.long))

        # Global features
        global_feats_list.append(gf)

        # Targets and IDs
        targets_list.append(target)
        ids_list.append(sample_id)

    # Concatenate/Stack
    atomic_batch = torch.cat(atomic_feats_list, dim=0)
    batch_index = torch.cat(batch_indices_list, dim=0)
    global_batch = torch.stack(global_feats_list, dim=0)
    targets_batch = torch.stack(targets_list, dim=0)
    ids_batch = torch.tensor(ids_list, dtype=torch.int32)

    return {
        "atomic_features": atomic_batch,
        "batch_index": batch_index,
        "global_features": global_batch,
        "targets": targets_batch,
        "ids": ids_batch,
    }
