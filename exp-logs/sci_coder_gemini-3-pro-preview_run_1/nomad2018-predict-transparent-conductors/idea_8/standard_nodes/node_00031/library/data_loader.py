import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import calculate_pbc_distance, get_logger

# Initialize logger
logger = get_logger("data_loader")


class MaterialDataset(Dataset):
    """
    PyTorch Dataset for material science data.

    Attributes:
        ids (np.ndarray): List of material IDs.
        atomic_features (list of np.ndarray): List of atomic feature matrices (N_atoms, 8).
        global_features (np.ndarray): Global feature matrix (N_samples, 11).
        symmetry_features (np.ndarray): Symmetry feature vector (N_samples,).
        targets (np.ndarray): Target values (N_samples, 2).
    """

    def __init__(
        self, ids, atomic_features, global_features, symmetry_features, targets=None
    ):
        self.ids = ids
        self.atomic_features = atomic_features
        self.global_features = global_features
        self.symmetry_features = symmetry_features
        self.targets = targets

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample = {
            "id": self.ids[idx],
            "atomic_x": torch.FloatTensor(self.atomic_features[idx]),
            "global_x": torch.FloatTensor(self.global_features[idx]),
            "symmetry_x": torch.LongTensor([self.symmetry_features[idx]]),
        }

        if self.targets is not None:
            sample["y"] = torch.FloatTensor(self.targets[idx])

        return sample


def parse_xyz(file_path):
    """
    Parses a geometry.xyz file to extract lattice vectors and atomic information.

    Args:
        file_path (str): Path to the .xyz file.

    Returns:
        tuple: (lattice_matrix, atomic_coords, atomic_symbols)
    """
    lattice_vectors = []
    atomic_coords = []
    atomic_symbols = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == "lattice_vector":
                lattice_vectors.append([float(x) for x in parts[1:4]])
            elif parts[0] == "atom":
                atomic_coords.append([float(x) for x in parts[1:4]])
                atomic_symbols.append(parts[4])

    return np.array(lattice_vectors), np.array(atomic_coords), atomic_symbols


def process_data(df, input_dir, mode="train"):
    """
    Processes raw data from the dataframe and xyz files into features.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        input_dir (str): Root directory containing geometry files.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        dict: Dictionary containing processed arrays.
    """
    logger.info(f"Processing {mode} data ({len(df)} samples)...")

    ids = []
    atomic_features_list = []
    global_features_list = []
    symmetry_features_list = []
    targets_list = []

    # Element mapping for one-hot encoding
    element_map = {"Al": 0, "Ga": 1, "In": 2, "O": 3}

    for _, row in df.iterrows():
        mat_id = row["id"]
        rel_path = row["file_path"]
        full_path = os.path.join(input_dir, rel_path)

        # 1. Parse Geometry
        try:
            lattice, coords, symbols = parse_xyz(full_path)
        except FileNotFoundError:
            logger.warning(f"File not found: {full_path}. Skipping.")
            continue

        # 2. Atomic Features
        # One-hot encoding
        n_atoms = len(symbols)
        one_hot = np.zeros((n_atoms, 4))
        for i, sym in enumerate(symbols):
            if sym in element_map:
                one_hot[i, element_map[sym]] = 1.0

        # Centered Coordinates
        centroid = np.mean(coords, axis=0)
        centered_coords = coords - centroid

        # Nearest Neighbor Distance (PBC corrected)
        nn_dists = calculate_pbc_distance(coords, lattice)

        # Combine Atomic Features: [One-hot(4), Coords(3), NN_Dist(1)]
        # Shape: (N_atoms, 8)
        # Note: We will standardize Coords and NN_Dist later
        atom_feats = np.hstack([one_hot, centered_coords, nn_dists.reshape(-1, 1)])

        # 3. Global Features
        # From CSV: Lattice lengths (3), Angles (3), Stoichiometry (3)
        # Derived: Volume, Density

        # Calculate Volume from lattice matrix (more precise than lengths/angles)
        # Handle case where lattice might be empty or invalid shape
        if lattice.shape == (3, 3):
            volume = np.abs(np.linalg.det(lattice))
        else:
            volume = 0.0

        density = n_atoms / volume if volume > 1e-6 else 0.0

        glob_feat = np.array(
            [
                row["lattice_vector_1_ang"],
                row["lattice_vector_2_ang"],
                row["lattice_vector_3_ang"],
                row["lattice_angle_alpha_degree"],
                row["lattice_angle_beta_degree"],
                row["lattice_angle_gamma_degree"],
                volume,
                density,
                row["percent_atom_al"],
                row["percent_atom_ga"],
                row["percent_atom_in"],
            ]
        )

        # 4. Symmetry Features
        sym_feat = row["spacegroup"]

        # 5. Targets (if available)
        if mode != "test":
            # Log transform targets: log(1 + y)
            # Targets are formation_energy and bandgap
            t1 = np.log1p(row["formation_energy_ev_natom"])
            t2 = np.log1p(row["bandgap_energy_ev"])
            targets_list.append([t1, t2])

        ids.append(mat_id)
        atomic_features_list.append(atom_feats)
        global_features_list.append(glob_feat)
        symmetry_features_list.append(sym_feat)

    # Convert lists to arrays where appropriate
    # atomic_features_list remains a list of arrays due to variable N_atoms
    global_features = np.array(global_features_list)
    symmetry_features = np.array(symmetry_features_list)
    targets = np.array(targets_list) if targets_list else None
    ids = np.array(ids)

    return {
        "ids": ids,
        "atomic_features": np.array(atomic_features_list, dtype=object),
        "global_features": global_features,
        "symmetry_features": symmetry_features,
        "targets": targets,
    }


def load_or_process_data(
    metadata_path, cache_path, input_dir, mode, load_cached_data=True
):
    """
    Loads data from cache or processes it from scratch.
    """
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached {mode} data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {
                "ids": data["ids"],
                "atomic_features": data["atomic_features"],
                "global_features": data["global_features"],
                "symmetry_features": data["symmetry_features"],
                "targets": data["targets"] if "targets" in data else None,
            }
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Reprocessing.")

    # Process from scratch
    df = pd.read_csv(metadata_path)
    data_dict = process_data(df, input_dir, mode)

    # Save to cache
    logger.info(f"Saving {mode} data to {cache_path}")
    save_dict = {k: v for k, v in data_dict.items() if v is not None}
    np.savez(cache_path, **save_dict)

    return data_dict


def collate_materials(batch):
    """
    Custom collate function to handle variable number of atoms.
    Pads atomic features to the maximum number of atoms in the batch.
    """
    ids = [item["id"] for item in batch]
    global_x = torch.stack([item["global_x"] for item in batch])
    symmetry_x = torch.cat([item["symmetry_x"] for item in batch])

    # Handle atomic features (variable length)
    atomic_list = [item["atomic_x"] for item in batch]
    lengths = [x.shape[0] for x in atomic_list]
    max_len = max(lengths)
    feature_dim = atomic_list[0].shape[1]

    # Create padded tensor and mask
    batch_size = len(batch)
    padded_atomic = torch.zeros(batch_size, max_len, feature_dim)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, (x, length) in enumerate(zip(atomic_list, lengths)):
        padded_atomic[i, :length, :] = x
        mask[i, :length] = True

    result = {
        "id": ids,
        "atomic_x": padded_atomic,
        "atomic_mask": mask,
        "global_x": global_x,
        "symmetry_x": symmetry_x,
    }

    if "y" in batch[0]:
        result["y"] = torch.stack([item["y"] for item in batch])

    return result


def get_dataloaders(batch_size=Config.BATCH_SIZE, load_cached_data=True, num_workers=2):
    """
    Main function to get DataLoaders. Handles scaling.
    """
    # 1. Load Data
    train_data = load_or_process_data(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_DATA_CACHE,
        Config.INPUT_DIR,
        "train",
        load_cached_data,
    )
    val_data = load_or_process_data(
        Config.VAL_METADATA_PATH,
        Config.VAL_DATA_CACHE,
        Config.INPUT_DIR,
        "val",
        load_cached_data,
    )
    test_data = load_or_process_data(
        Config.TEST_METADATA_PATH,
        Config.TEST_DATA_CACHE,
        Config.INPUT_DIR,
        "test",
        load_cached_data,
    )

    # 2. Fit Scalers on Training Data
    logger.info("Fitting scalers on training data...")

    # Atomic Scaler: Scale Centered Coords (cols 4-6) and NN Dist (col 7)
    # Flatten all atomic features from training set
    all_atomic_train = np.vstack(train_data["atomic_features"])
    atomic_scaler = StandardScaler()
    # Fit only on cols 4: (coords + dist)
    atomic_scaler.fit(all_atomic_train[:, 4:])

    # Global Scaler: Scale all global features
    global_scaler = StandardScaler()
    global_scaler.fit(train_data["global_features"])

    # 3. Apply Scaling
    def apply_scaling(data_dict):
        # Scale atomic features
        scaled_atomic = []
        for feat in data_dict["atomic_features"]:
            feat_copy = feat.copy()
            feat_copy[:, 4:] = atomic_scaler.transform(feat[:, 4:])
            scaled_atomic.append(feat_copy)
        data_dict["atomic_features"] = scaled_atomic

        # Scale global features
        data_dict["global_features"] = global_scaler.transform(
            data_dict["global_features"]
        )
        return data_dict

    train_data = apply_scaling(train_data)
    val_data = apply_scaling(val_data)
    test_data = apply_scaling(test_data)

    # 4. Create Datasets
    train_dataset = MaterialDataset(
        train_data["ids"],
        train_data["atomic_features"],
        train_data["global_features"],
        train_data["symmetry_features"],
        train_data["targets"],
    )
    val_dataset = MaterialDataset(
        val_data["ids"],
        val_data["atomic_features"],
        val_data["global_features"],
        val_data["symmetry_features"],
        val_data["targets"],
    )
    test_dataset = MaterialDataset(
        test_data["ids"],
        test_data["atomic_features"],
        test_data["global_features"],
        test_data["symmetry_features"],
        None,
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_materials,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_materials,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_materials,
        pin_memory=True,
    )

    logger.info(
        f"DataLoaders created. Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}"
    )
    return train_loader, val_loader, test_loader
