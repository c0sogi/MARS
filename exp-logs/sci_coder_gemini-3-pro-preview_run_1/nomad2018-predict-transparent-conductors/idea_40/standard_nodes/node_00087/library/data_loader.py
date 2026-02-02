import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.geometry_utils import parse_xyz, compute_atomic_features


class MaterialDataset(Dataset):
    """
    PyTorch Dataset for Material Science data.
    Stores flattened atomic features and slices for efficient memory usage.
    """

    def __init__(self, atomic_features_flat, slices, global_features, targets, ids):
        self.atomic_features_flat = atomic_features_flat
        self.slices = slices
        self.global_features = global_features
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        start, length = self.slices[idx]
        # Extract atomic features for this specific crystal
        atomic_feats = self.atomic_features_flat[start : start + length]
        global_feats = self.global_features[idx]
        target = self.targets[idx]
        id_ = self.ids[idx]

        return {
            "atomic_feats": torch.tensor(atomic_feats, dtype=torch.float32),
            "global_feats": torch.tensor(global_feats, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "id": torch.tensor(id_, dtype=torch.long),
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable number of atoms.
    Pads atomic feature tensors to the maximum size in the batch.
    """
    ids = torch.stack([b["id"] for b in batch])
    targets = torch.stack([b["target"] for b in batch])
    global_feats = torch.stack([b["global_feats"] for b in batch])

    atomic_feats_list = [b["atomic_feats"] for b in batch]

    # Find max number of atoms in this batch for padding
    max_atoms = max([af.shape[0] for af in atomic_feats_list])
    feat_dim = atomic_feats_list[0].shape[1]

    batch_size = len(batch)

    # Create padded tensor and mask
    padded_atomic = torch.zeros((batch_size, max_atoms, feat_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool)

    for i, af in enumerate(atomic_feats_list):
        n = af.shape[0]
        padded_atomic[i, :n, :] = af
        mask[i, :n] = True

    return {
        "atomic_feats": padded_atomic,
        "global_feats": global_feats,
        "mask": mask,
        "target": targets,
        "id": ids,
    }


def compute_global_features(lattice, atom_types, df_row):
    """
    Computes global thermodynamic and geometric features.
    """
    # Lattice vector lengths
    lengths = np.linalg.norm(lattice, axis=1)

    # Calculate angles between lattice vectors
    def angle(v1, v2):
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # Clip to handle numerical errors
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    alpha = angle(lattice[1], lattice[2])
    beta = angle(lattice[0], lattice[2])
    gamma = angle(lattice[0], lattice[1])

    # Unit Cell Volume
    volume = np.abs(np.linalg.det(lattice))

    # Total atoms
    n_atoms = len(atom_types)

    # Atomic Density
    density = n_atoms / (volume + 1e-6)

    # Stoichiometry from dataframe
    pct_al = df_row["percent_atom_al"]
    pct_ga = df_row["percent_atom_ga"]
    pct_in = df_row["percent_atom_in"]

    # Feature vector: [a, b, c, alpha, beta, gamma, vol, dens, al, ga, in, n_atoms]
    return np.array(
        [
            lengths[0],
            lengths[1],
            lengths[2],
            alpha,
            beta,
            gamma,
            volume,
            density,
            pct_al,
            pct_ga,
            pct_in,
            float(n_atoms),
        ],
        dtype=np.float32,
    )


def process_subset(df, subset_name, load_cached_data=True):
    """
    Processes a subset of data (train/val/test), computing features and caching them.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{subset_name}_data.npz")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {subset_name} data from {cache_path}...")
        try:
            data = np.load(cache_path)
            return (
                data["atomic_features_flat"],
                data["slices"],
                data["global_features"],
                data["targets"],
                data["ids"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing {subset_name} data...")

    atomic_feats_list = []
    global_feats_list = []
    targets_list = []
    ids_list = []
    slices_list = []

    current_idx = 0

    # Debugging option to run on small subset
    if Config.DEBUG:
        print(f"DEBUG MODE: Processing only {Config.DEBUG_SAMPLE_SIZE} samples.")
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    for _, row in df.iterrows():
        id_ = row["id"]
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Parse geometry file
        lattice, atom_types, coords = parse_xyz(full_path)

        # Compute Atomic features (N, 13)
        af = compute_atomic_features(
            atom_types, coords, lattice, k_neighbors=Config.K_NEIGHBORS
        )

        # Compute Global features (12,)
        gf = compute_global_features(lattice, atom_types, row)

        # Handle Targets
        if "formation_energy_ev_natom" in row and pd.notna(
            row["formation_energy_ev_natom"]
        ):
            t1 = row["formation_energy_ev_natom"]
            t2 = row["bandgap_energy_ev"]
            # Log transform: log(1 + y) to align with RMSLE metric
            target = np.array([np.log1p(t1), np.log1p(t2)], dtype=np.float32)
        else:
            # Placeholder for test set
            target = np.array([0.0, 0.0], dtype=np.float32)

        n_atoms = af.shape[0]

        atomic_feats_list.append(af)
        global_feats_list.append(gf)
        targets_list.append(target)
        ids_list.append(id_)
        slices_list.append([current_idx, n_atoms])

        current_idx += n_atoms

    # Concatenate all data
    if len(atomic_feats_list) > 0:
        atomic_features_flat = np.vstack(atomic_feats_list)
        global_features = np.vstack(global_feats_list)
        targets = np.vstack(targets_list)
        ids = np.array(ids_list, dtype=np.int64)
        slices = np.array(slices_list, dtype=np.int64)
    else:
        # Handle empty case (should not happen in normal flow)
        atomic_features_flat = np.zeros((0, Config.ATOMIC_FEATURE_DIM))
        global_features = np.zeros((0, Config.GLOBAL_FEATURE_DIM))
        targets = np.zeros((0, Config.NUM_TARGETS))
        ids = np.zeros((0,), dtype=np.int64)
        slices = np.zeros((0, 2), dtype=np.int64)

    # Save to cache
    np.savez_compressed(
        cache_path,
        atomic_features_flat=atomic_features_flat,
        slices=slices,
        global_features=global_features,
        targets=targets,
        ids=ids,
    )

    return atomic_features_flat, slices, global_features, targets, ids


def get_loaders(load_cached_data=True):
    """
    Main function to load data, fit scalers, and return PyTorch DataLoaders.
    """
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Process data (load from cache or compute)
    train_data = process_subset(train_df, "train", load_cached_data)
    val_data = process_subset(val_df, "val", load_cached_data)
    test_data = process_subset(test_df, "test", load_cached_data)

    # Unpack train data for scaler fitting
    train_af_flat, _, train_gf, _, _ = train_data

    # Fit scalers on TRAINING data only
    # Atomic features: Scale indices 4 to 13 (coords, d_min, d_mean, context)
    # Indices 0-3 are One-Hot encoded and should not be scaled.
    atomic_scaler = StandardScaler()
    if train_af_flat.shape[0] > 0:
        atomic_scaler.fit(train_af_flat[:, 4:])

    # Global features: Scale all
    global_scaler = StandardScaler()
    if train_gf.shape[0] > 0:
        global_scaler.fit(train_gf)

    # Helper to apply scaling and create Dataset
    def transform_and_create_dataset(data_tuple):
        af_flat, slices, gf, targets, ids = data_tuple

        # Scale atomic features (copy to avoid modifying cache in memory)
        af_scaled = af_flat.copy()
        if af_flat.shape[0] > 0:
            af_scaled[:, 4:] = atomic_scaler.transform(af_flat[:, 4:])

        # Scale global features
        gf_scaled = gf.copy()
        if gf.shape[0] > 0:
            gf_scaled = global_scaler.transform(gf)

        return MaterialDataset(af_scaled, slices, gf_scaled, targets, ids)

    train_dataset = transform_and_create_dataset(train_data)
    val_dataset = transform_and_create_dataset(val_data)
    test_dataset = transform_and_create_dataset(test_data)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
