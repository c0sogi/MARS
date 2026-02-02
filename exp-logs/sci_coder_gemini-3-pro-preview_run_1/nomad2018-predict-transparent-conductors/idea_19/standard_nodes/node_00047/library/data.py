import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import compute_pbc_distance_matrix, RobustScaler, log_transform


class GeometryParser:
    """
    Parses geometry.xyz files and computes atomic-level features.
    """

    @staticmethod
    def parse_and_process(file_path, lattice_angles_deg=None):
        """
        Reads xyz file, computes centered coords and proximity features.

        Args:
            file_path: Path to geometry.xyz
            lattice_angles_deg: Tuple of (alpha, beta, gamma) from metadata,
                                used if lattice vectors need verification (optional).

        Returns:
            atomic_features: (N, 11) numpy array
                             [One-Hot(4) | Centered Coords(3) | Proximity(4)]
            num_atoms: int
            volume: float
        """
        full_path = os.path.join(Config.INPUT_DIR, file_path)
        with open(full_path, "r") as f:
            lines = f.readlines()

        # Parse Lattice Vectors (lines 3-5 usually, but based on file format description)
        # Format:
        # lattice_vector x y z
        lattice_vectors = []
        atoms = []
        coords = []

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "lattice_vector":
                lattice_vectors.append([float(x) for x in parts[1:4]])
            elif parts[0] == "atom":
                # atom x y z symbol
                coords.append([float(x) for x in parts[1:4]])
                atoms.append(parts[4])

        lattice_matrix = np.array(lattice_vectors, dtype=np.float32)
        coords = np.array(coords, dtype=np.float32)
        num_atoms = len(atoms)

        # 1. Centering
        centroid = np.mean(coords, axis=0)
        centered_coords = coords - centroid

        # 2. One-Hot Encoding
        one_hot = np.zeros((num_atoms, Config.NUM_ATOM_TYPES), dtype=np.float32)
        atom_indices = [Config.ATOM_TO_IDX[atom] for atom in atoms]
        one_hot[np.arange(num_atoms), atom_indices] = 1.0

        # 3. PBC Distance Matrix
        # Shape: (N, N)
        dist_matrix = compute_pbc_distance_matrix(coords, lattice_matrix)

        # 4. Chemically-Resolved Proximity
        # For each atom i, find min dist to any atom of type T (excluding self if same type)
        proximity_feats = np.zeros((num_atoms, Config.NUM_ATOM_TYPES), dtype=np.float32)

        # Create masks for each atom type
        type_masks = []
        for t_idx in range(Config.NUM_ATOM_TYPES):
            type_masks.append(np.array(atom_indices) == t_idx)

        for i in range(num_atoms):
            for t_idx in range(Config.NUM_ATOM_TYPES):
                mask = type_masks[t_idx]
                if not np.any(mask):
                    # Element not present in crystal
                    min_dist = np.inf
                else:
                    # Distances to all atoms of type t_idx
                    dists = dist_matrix[i, mask]

                    # If the atom itself is of type t_idx, the distance is 0.
                    # We want the nearest *neighbor*.
                    # However, standard proximity usually implies "distance to nearest X".
                    # If i is X, distance is 0. Let's keep it 0 to indicate "I am X".
                    # But wait, if I am Al, my proximity to Al is 0.
                    # If I am O, my proximity to Al is bond length.
                    # This differentiates identity from environment.
                    # Let's stick to raw min distance including self.
                    # If we want neighbor, we'd filter > 1e-6.
                    # Given the prompt implies "proximity to nearest neighbor", usually self-loops are excluded
                    # in graph contexts, but here it's a point cloud feature.
                    # Let's exclude self-distance to make it a true "neighbor" feature.

                    if mask[i]:  # If atom i is of this type
                        # Filter out the 0.0 distance (self)
                        valid_dists = dists[dists > 1e-5]
                        if len(valid_dists) > 0:
                            min_dist = np.min(valid_dists)
                        else:
                            # Only one atom of this type exists
                            min_dist = np.inf
                    else:
                        min_dist = np.min(dists)

                if np.isinf(min_dist):
                    proximity_feats[i, t_idx] = 0.0
                else:
                    proximity_feats[i, t_idx] = np.exp(
                        -Config.PROXIMITY_GAMMA * min_dist
                    )

        # Concatenate all atomic features
        # [One-Hot (4) | Centered Coords (3) | Proximity (4)]
        atomic_features = np.hstack([one_hot, centered_coords, proximity_feats])

        # Calculate Volume (scalar triple product)
        volume = np.abs(
            np.dot(lattice_matrix[0], np.cross(lattice_matrix[1], lattice_matrix[2]))
        )

        return atomic_features, num_atoms, volume


class MaterialDataset(Dataset):
    def __init__(
        self, atomic_features_flat, lengths, global_features, targets=None, ids=None
    ):
        """
        Args:
            atomic_features_flat: (Sum_N, Atomic_Dim) numpy array
            lengths: (Num_Samples,) numpy array containing number of atoms per sample
            global_features: (Num_Samples, Global_Dim) numpy array
            targets: (Num_Samples, Num_Targets) numpy array (optional)
            ids: (Num_Samples,) numpy array of IDs
        """
        self.atomic_features_flat = torch.FloatTensor(atomic_features_flat)
        self.lengths = torch.LongTensor(lengths)
        self.global_features = torch.FloatTensor(global_features)

        if targets is not None:
            self.targets = torch.FloatTensor(targets)
        else:
            self.targets = None

        self.ids = ids

        # Pre-calculate start indices for fast slicing
        self.starts = torch.cumsum(
            torch.cat([torch.LongTensor([0]), self.lengths[:-1]]), dim=0
        )

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = start + self.lengths[idx]

        atom_feats = self.atomic_features_flat[start:end]
        glob_feats = self.global_features[idx]

        sample = {
            "atomic_features": atom_feats,
            "global_features": glob_feats,
            "length": self.lengths[idx],
            "id": self.ids[idx] if self.ids is not None else -1,
        }

        if self.targets is not None:
            sample["target"] = self.targets[idx]

        return sample


def collate_fn(batch):
    """
    Pads atomic features to the max length in the batch.
    Returns dictionary with padded features and masks.
    """
    batch_size = len(batch)

    # Global features and targets are simple stacks
    global_features = torch.stack([b["global_features"] for b in batch])
    ids = [b["id"] for b in batch]

    targets = None
    if "target" in batch[0]:
        targets = torch.stack([b["target"] for b in batch])

    # Atomic features need padding
    lengths = [b["length"] for b in batch]
    max_len = max(lengths)
    feat_dim = batch[0]["atomic_features"].shape[1]

    padded_atomic = torch.zeros(batch_size, max_len, feat_dim)
    mask = torch.zeros(
        batch_size, max_len, dtype=torch.bool
    )  # 1 for valid, 0 for padding

    for i, b in enumerate(batch):
        l = b["length"]
        padded_atomic[i, :l, :] = b["atomic_features"]
        mask[i, :l] = True

    return {
        "atomic_features": padded_atomic,  # (B, Max_L, 11)
        "mask": mask,  # (B, Max_L)
        "global_features": global_features,  # (B, 12)
        "targets": targets,  # (B, 2) or None
        "ids": ids,
    }


def process_split(df, scaler_atomic=None, scaler_global=None, is_train=False):
    """
    Process a dataframe (train/val/test) into arrays.
    Fits scalers if is_train is True.
    """
    all_atomic_feats = []
    all_global_feats = []
    all_lengths = []
    all_targets = []
    all_ids = []

    # Features to extract from CSV directly
    global_cols = [
        "lattice_vector_1_ang",
        "lattice_vector_2_ang",
        "lattice_vector_3_ang",
        "lattice_angle_alpha_degree",
        "lattice_angle_beta_degree",
        "lattice_angle_gamma_degree",
        "percent_atom_al",
        "percent_atom_ga",
        "percent_atom_in",
        "number_of_total_atoms",
    ]

    for _, row in df.iterrows():
        # 1. Geometry Processing
        atomic_feats, num_atoms, volume = GeometryParser.parse_and_process(
            row["file_path"]
        )

        # 2. Global Features
        # Base from CSV
        glob_base = row[global_cols].values.astype(np.float32)
        # Derived
        density = row["number_of_total_atoms"] / volume

        # Combine: [Lattice(6) | Volume(1) | Density(1) | Stoich(3) | Total(1)]
        # Note: glob_base has [L(3), Ang(3), Stoich(3), Total(1)]
        # Reorder to match Config: L(3)+Ang(3) + Vol(1) + Dens(1) + Stoich(3) + Total(1)

        l_params = glob_base[0:6]
        stoich = glob_base[6:9]
        total_atoms = glob_base[9]

        glob_combined = np.concatenate(
            [l_params, [volume, density], stoich, [total_atoms]]
        )

        all_atomic_feats.append(atomic_feats)
        all_global_feats.append(glob_combined)
        all_lengths.append(num_atoms)
        all_ids.append(row["id"])

        if "formation_energy_ev_natom" in row:
            t = [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
            all_targets.append(t)

    # Flatten atomic features for scaling and storage
    flat_atomic = np.vstack(all_atomic_feats)
    stacked_global = np.vstack(all_global_feats)

    # Scaling
    # Atomic: Scale coords (cols 4-6) and proximity (cols 7-10). Leave one-hot (0-3) alone.
    # Global: Scale all.

    if is_train:
        scaler_atomic = RobustScaler()
        # Fit on continuous parts: indices 4 to 11
        scaler_atomic.fit(flat_atomic[:, 4:])

        scaler_global = RobustScaler()
        scaler_global.fit(stacked_global)

    # Transform
    if scaler_atomic is not None:
        flat_atomic[:, 4:] = scaler_atomic.transform(flat_atomic[:, 4:])

    if scaler_global is not None:
        stacked_global = scaler_global.transform(stacked_global)

    # Targets
    if all_targets:
        targets_arr = np.array(all_targets, dtype=np.float32)
        # Log transform
        targets_arr = log_transform(targets_arr)
    else:
        targets_arr = None

    return (
        {
            "atomic_features_flat": flat_atomic,
            "lengths": np.array(all_lengths, dtype=np.int64),
            "global_features": stacked_global,
            "targets": targets_arr,
            "ids": np.array(all_ids, dtype=np.int64),
        },
        scaler_atomic,
        scaler_global,
    )


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
    debug_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Main function to load data, process/cache it, and return DataLoaders.
    """

    # 1. Check Cache
    cache_exists = (
        os.path.exists(Config.TRAIN_DATA_CACHE)
        and os.path.exists(Config.VAL_DATA_CACHE)
        and os.path.exists(Config.TEST_DATA_CACHE)
        and os.path.exists(Config.SCALERS_CACHE)
    )

    if load_cached_data and cache_exists:
        print("Loading cached data...")
        train_data = np.load(Config.TRAIN_DATA_CACHE)
        val_data = np.load(Config.VAL_DATA_CACHE)
        test_data = np.load(Config.TEST_DATA_CACHE)

        # Reconstruct dictionaries
        train_dict = {k: train_data[k] for k in train_data}
        val_dict = {k: val_data[k] for k in val_data}
        test_dict = {k: test_data[k] for k in test_data}

        # Targets might be None in test, np.load handles None as None object if saved correctly,
        # but usually npz saves arrays. For test targets, we expect it to be missing or dummy.
        # Our save logic below handles it.
        if (
            "targets" not in test_dict or test_dict["targets"].ndim == 0
        ):  # Handle None saved as 0-d array
            test_dict["targets"] = None

    else:
        print("Processing data from scratch...")
        # Load Metadata
        train_df = pd.read_csv(Config.TRAIN_METADATA)
        val_df = pd.read_csv(Config.VAL_METADATA)
        test_df = pd.read_csv(Config.TEST_METADATA)

        # Debug subset
        if debug_size:
            train_df = train_df.head(debug_size)
            val_df = val_df.head(debug_size)
            test_df = test_df.head(debug_size)
            print(f"Debug mode: using {debug_size} samples per split.")

        # Process Train (fits scalers)
        print("Processing Train...")
        train_dict, scaler_atomic, scaler_global = process_split(
            train_df, is_train=True
        )

        # Process Val
        print("Processing Val...")
        val_dict, _, _ = process_split(
            val_df, scaler_atomic=scaler_atomic, scaler_global=scaler_global
        )

        # Process Test
        print("Processing Test...")
        test_dict, _, _ = process_split(
            test_df, scaler_atomic=scaler_atomic, scaler_global=scaler_global
        )

        # Save Cache
        print("Saving cache...")
        np.savez(Config.TRAIN_DATA_CACHE, **train_dict)
        np.savez(Config.VAL_DATA_CACHE, **val_dict)
        # Handle None targets for test
        test_save_dict = test_dict.copy()
        if test_save_dict["targets"] is None:
            del test_save_dict["targets"]  # np.savez doesn't like None
        np.savez(Config.TEST_DATA_CACHE, **test_save_dict)

        # Save Scalers (we can't easily save objects to npz, but we can save their params if needed)
        # For simplicity in this script, we assume re-running training fits scalers again if cache is invalid.
        # To strictly follow "save result to cache", we save dummy file to indicate completion or save params.
        # Here we just save a marker file or params.
        np.savez(
            Config.SCALERS_CACHE,
            atomic_mean=scaler_atomic.mean_,
            atomic_scale=scaler_atomic.scale_,
            global_mean=scaler_global.mean_,
            global_scale=scaler_global.scale_,
        )

    # 2. Create Datasets
    train_dataset = MaterialDataset(**train_dict)
    val_dataset = MaterialDataset(**val_dict)

    # Handle test targets potentially being missing in dict if loaded from cache where it was deleted
    test_targets = test_dict.get("targets", None)
    test_dataset = MaterialDataset(
        atomic_features_flat=test_dict["atomic_features_flat"],
        lengths=test_dict["lengths"],
        global_features=test_dict["global_features"],
        targets=test_targets,
        ids=test_dict["ids"],
    )

    # 3. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
