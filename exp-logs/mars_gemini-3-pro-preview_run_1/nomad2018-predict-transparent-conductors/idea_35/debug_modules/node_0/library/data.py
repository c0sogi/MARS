import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.utils import (
    parse_xyz,
    compute_pbc_distances,
    get_multi_order_neighbors,
    calculate_apf,
)

# Element mapping for one-hot encoding
ELEMENTS = ["Al", "Ga", "In", "O"]
ELEMENT_TO_IDX = {el: i for i, el in enumerate(ELEMENTS)}


class MaterialDataset(Dataset):
    def __init__(self, atomic_features, global_features, targets=None, ids=None):
        """
        PyTorch Dataset for materials.

        Args:
            atomic_features (list of np.ndarray): List of (N_atoms, 10) arrays.
            global_features (np.ndarray): Array of shape (N_samples, 13).
            targets (np.ndarray, optional): Array of shape (N_samples, 2).
            ids (np.ndarray, optional): Array of IDs.
        """
        self.atomic_features = atomic_features
        self.global_features = torch.tensor(global_features, dtype=torch.float32)

        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

        self.ids = ids

    def __len__(self):
        return len(self.global_features)

    def __getitem__(self, idx):
        atomic_feat = torch.tensor(self.atomic_features[idx], dtype=torch.float32)
        global_feat = self.global_features[idx]

        item = {
            "atomic_features": atomic_feat,
            "global_features": global_feat,
            "id": self.ids[idx] if self.ids is not None else -1,
        }

        if self.targets is not None:
            item["targets"] = self.targets[idx]

        return item


def collate_fn(batch):
    """
    Collate function to pad atomic features and stack other data.
    """
    atomic_features_list = [item["atomic_features"] for item in batch]
    global_features = torch.stack([item["global_features"] for item in batch])
    ids = [item["id"] for item in batch]

    # Pad atomic features (batch_first=True)
    # Result shape: (Batch, Max_Atoms, Feature_Dim)
    padded_atomic_features = pad_sequence(
        atomic_features_list, batch_first=True, padding_value=0.0
    )

    # Create mask (Batch, Max_Atoms) - 1 for real atom, 0 for padding
    # We can determine length from the unpadded list
    lengths = torch.tensor([len(x) for x in atomic_features_list], dtype=torch.long)
    max_len = padded_atomic_features.size(1)
    mask = torch.arange(max_len)[None, :] < lengths[:, None]

    batch_dict = {
        "atomic_features": padded_atomic_features,
        "global_features": global_features,
        "mask": mask,
        "ids": ids,
    }

    if "targets" in batch[0]:
        targets = torch.stack([item["targets"] for item in batch])
        batch_dict["targets"] = targets

    return batch_dict


def process_dataframe(df, input_dir):
    """
    Process a dataframe to extract atomic and global features.
    """
    atomic_feats_list = []
    global_feats_list = []
    targets_list = []
    ids_list = []

    # Columns for global features from CSV
    # 1. Lattice lengths (3)
    # 2. Lattice angles (3)
    # 3. Stoichiometry (3)
    # 4. Total Atoms (1)
    # We will compute Volume, Density, APF from geometry/derived.

    lat_len_cols = [
        "lattice_vector_1_ang",
        "lattice_vector_2_ang",
        "lattice_vector_3_ang",
    ]
    lat_ang_cols = [
        "lattice_angle_alpha_degree",
        "lattice_angle_beta_degree",
        "lattice_angle_gamma_degree",
    ]
    stoich_cols = ["percent_atom_al", "percent_atom_ga", "percent_atom_in"]
    total_atoms_col = "number_of_total_atoms"

    target_cols = Config.TARGET_COLS

    for _, row in df.iterrows():
        # 1. Load Geometry
        file_path = os.path.join(input_dir, row["file_path"])
        atoms = parse_xyz(file_path)

        # 2. Atomic Features
        # One-hot
        symbols = atoms.get_chemical_symbols()
        n_atoms = len(symbols)
        one_hot = np.zeros((n_atoms, 4))
        for i, sym in enumerate(symbols):
            if sym in ELEMENT_TO_IDX:
                one_hot[i, ELEMENT_TO_IDX[sym]] = 1.0

        # Centered Coordinates
        positions = atoms.get_positions()
        centroid = np.mean(positions, axis=0)
        centered_pos = positions - centroid

        # Neighbors (d1, d2, d3)
        sorted_dists = compute_pbc_distances(atoms)
        neighbor_dists = get_multi_order_neighbors(sorted_dists, k=3)

        # Combine Atomic Features (N, 4+3+3=10)
        atom_f = np.hstack([one_hot, centered_pos, neighbor_dists])
        atomic_feats_list.append(atom_f)

        # 3. Global Features
        # From CSV
        l_lengths = row[lat_len_cols].values.astype(float)
        l_angles = row[lat_ang_cols].values.astype(float)
        stoich = row[stoich_cols].values.astype(float)
        n_total = float(row[total_atoms_col])

        # Derived
        vol = atoms.get_volume()
        density = n_total / vol if vol > 1e-6 else 0.0
        apf = calculate_apf(atoms)

        # Combine Global (3 + 3 + 1 + 1 + 3 + 1 + 1 = 13)
        # Order: lengths, angles, volume, density, stoich, total_atoms, apf
        glob_f = np.concatenate(
            [l_lengths, l_angles, [vol], [density], stoich, [n_total], [apf]]
        )
        global_feats_list.append(glob_f)

        # 4. Targets (Log transform)
        if all(col in row for col in target_cols):
            t = row[target_cols].values.astype(float)
            # Apply log(1+y)
            t = np.log1p(t)
            targets_list.append(t)

        ids_list.append(row["id"])

    # Convert to numpy arrays where appropriate
    # atomic_feats_list stays as list of arrays because lengths vary
    global_feats_arr = np.array(global_feats_list)
    targets_arr = np.array(targets_list) if targets_list else None
    ids_arr = np.array(ids_list)

    return atomic_feats_list, global_feats_arr, targets_arr, ids_arr


def flatten_atomic_features(atomic_features_list):
    """Flatten list of atomic features into a single array and a counts array."""
    counts = np.array([len(x) for x in atomic_features_list])
    flat = np.vstack(atomic_features_list)
    return flat, counts


def unflatten_atomic_features(flat_features, counts):
    """Reconstruct list of atomic features from flattened array and counts."""
    features_list = []
    idx = 0
    for c in counts:
        features_list.append(flat_features[idx : idx + c])
        idx += c
    return features_list


def fit_and_save_scalers(train_atomic_flat, train_global, save_path):
    """
    Compute mean and std for atomic and global features on training set.
    Save to npz.
    """
    # Atomic Scaler
    atomic_mean = np.mean(train_atomic_flat, axis=0)
    atomic_std = np.std(train_atomic_flat, axis=0)
    # Avoid division by zero
    atomic_std[atomic_std < 1e-8] = 1.0

    # Global Scaler
    global_mean = np.mean(train_global, axis=0)
    global_std = np.std(train_global, axis=0)
    global_std[global_std < 1e-8] = 1.0

    np.savez(
        save_path,
        atomic_mean=atomic_mean,
        atomic_std=atomic_std,
        global_mean=global_mean,
        global_std=global_std,
    )

    return atomic_mean, atomic_std, global_mean, global_std


def load_scalers(load_path):
    data = np.load(load_path)
    return (
        data["atomic_mean"],
        data["atomic_std"],
        data["global_mean"],
        data["global_std"],
    )


def apply_scaling(
    features_list,
    features_flat,
    global_arr,
    atomic_mean,
    atomic_std,
    global_mean,
    global_std,
):
    """
    Apply standard scaling to features.
    """
    # Scale flattened atomic features
    scaled_flat = (features_flat - atomic_mean) / atomic_std

    # Reconstruct list structure
    counts = np.array([len(x) for x in features_list])
    scaled_list = unflatten_atomic_features(scaled_flat, counts)

    # Scale global features
    scaled_global = (global_arr - global_mean) / global_std

    return scaled_list, scaled_global


def get_data_loaders(load_cached_data=True):
    """
    Main function to prepare data and return DataLoaders.

    Args:
        load_cached_data (bool): If True, try to load processed data from disk.

    Returns:
        train_loader, val_loader, test_loader (DataLoader)
    """

    # Check if cache exists
    cache_exists = (
        os.path.exists(Config.TRAIN_CACHE_PATH)
        and os.path.exists(Config.VAL_CACHE_PATH)
        and os.path.exists(Config.TEST_CACHE_PATH)
        and os.path.exists(Config.SCALERS_CACHE_PATH)
    )

    if load_cached_data and cache_exists:
        print("Loading cached data...")
        # Load Train
        train_data = np.load(Config.TRAIN_CACHE_PATH)
        train_atomic = unflatten_atomic_features(
            train_data["atomic_flat"], train_data["counts"]
        )
        train_global = train_data["global_feat"]
        train_targets = train_data["targets"]
        train_ids = train_data["ids"]

        # Load Val
        val_data = np.load(Config.VAL_CACHE_PATH)
        val_atomic = unflatten_atomic_features(
            val_data["atomic_flat"], val_data["counts"]
        )
        val_global = val_data["global_feat"]
        val_targets = val_data["targets"]
        val_ids = val_data["ids"]

        # Load Test
        test_data = np.load(Config.TEST_CACHE_PATH)
        test_atomic = unflatten_atomic_features(
            test_data["atomic_flat"], test_data["counts"]
        )
        test_global = test_data["global_feat"]
        test_ids = test_data["ids"]

    else:
        print("Processing data from scratch...")
        # Load Metadata
        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        val_df = pd.read_csv(Config.VAL_METADATA_PATH)
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)

        if Config.DEBUG:
            print(f"Debug mode: using {Config.DEBUG_SIZE} samples.")
            train_df = train_df.head(Config.DEBUG_SIZE)
            val_df = val_df.head(Config.DEBUG_SIZE)
            test_df = test_df.head(Config.DEBUG_SIZE)

        # Process splits
        print("Processing Train set...")
        tr_atom, tr_glob, tr_targ, tr_ids = process_dataframe(
            train_df, Config.INPUT_DIR
        )
        print("Processing Val set...")
        va_atom, va_glob, va_targ, va_ids = process_dataframe(val_df, Config.INPUT_DIR)
        print("Processing Test set...")
        te_atom, te_glob, _, te_ids = process_dataframe(test_df, Config.INPUT_DIR)

        # Flatten for scaling and storage
        tr_atom_flat, tr_counts = flatten_atomic_features(tr_atom)
        va_atom_flat, va_counts = flatten_atomic_features(va_atom)
        te_atom_flat, te_counts = flatten_atomic_features(te_atom)

        # Fit Scalers on Train
        print("Fitting scalers...")
        a_mean, a_std, g_mean, g_std = fit_and_save_scalers(
            tr_atom_flat, tr_glob, Config.SCALERS_CACHE_PATH
        )

        # Apply Scaling
        print("Applying scaling...")
        tr_atom, tr_glob = apply_scaling(
            tr_atom, tr_atom_flat, tr_glob, a_mean, a_std, g_mean, g_std
        )
        va_atom, va_glob = apply_scaling(
            va_atom, va_atom_flat, va_glob, a_mean, a_std, g_mean, g_std
        )
        te_atom, te_glob = apply_scaling(
            te_atom, te_atom_flat, te_glob, a_mean, a_std, g_mean, g_std
        )

        # Re-flatten scaled data for saving
        tr_atom_flat_s, _ = flatten_atomic_features(tr_atom)
        va_atom_flat_s, _ = flatten_atomic_features(va_atom)
        te_atom_flat_s, _ = flatten_atomic_features(te_atom)

        # Save to Cache
        print("Saving to cache...")
        np.savez_compressed(
            Config.TRAIN_CACHE_PATH,
            atomic_flat=tr_atom_flat_s,
            counts=tr_counts,
            global_feat=tr_glob,
            targets=tr_targ,
            ids=tr_ids,
        )
        np.savez_compressed(
            Config.VAL_CACHE_PATH,
            atomic_flat=va_atom_flat_s,
            counts=va_counts,
            global_feat=va_glob,
            targets=va_targ,
            ids=va_ids,
        )
        np.savez_compressed(
            Config.TEST_CACHE_PATH,
            atomic_flat=te_atom_flat_s,
            counts=te_counts,
            global_feat=te_glob,
            ids=te_ids,
        )

        # Assign to local variables for dataset creation
        train_atomic, train_global, train_targets, train_ids = (
            tr_atom,
            tr_glob,
            tr_targ,
            tr_ids,
        )
        val_atomic, val_global, val_targets, val_ids = va_atom, va_glob, va_targ, va_ids
        test_atomic, test_global, test_ids = te_atom, te_glob, te_ids

    # Create Datasets
    train_dataset = MaterialDataset(
        train_atomic, train_global, train_targets, train_ids
    )
    val_dataset = MaterialDataset(val_atomic, val_global, val_targets, val_ids)
    test_dataset = MaterialDataset(test_atomic, test_global, None, test_ids)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
