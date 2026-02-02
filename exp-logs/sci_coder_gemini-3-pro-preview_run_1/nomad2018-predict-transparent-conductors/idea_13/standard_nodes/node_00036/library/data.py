import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import (
    parse_xyz,
    get_lattice_params,
    get_cell_volume,
    center_coordinates,
    cartesian_to_fractional,
    compute_pbc_distances,
    compute_local_potential,
)


class MaterialDataset(Dataset):
    """
    PyTorch Dataset for material science data.
    """

    def __init__(self, atomic_features, global_features, targets=None, ids=None):
        """
        Args:
            atomic_features (list of np.ndarray): List of (N_atoms, Feature_Dim) arrays.
            global_features (np.ndarray): Array of (N_samples, Global_Dim) global features.
            targets (np.ndarray, optional): Array of (N_samples, 2) targets.
            ids (np.ndarray, optional): Array of (N_samples,) IDs.
        """
        self.atomic_features = atomic_features
        self.global_features = global_features
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.atomic_features)

    def __getitem__(self, idx):
        sample = {
            "atomic_features": torch.tensor(
                self.atomic_features[idx], dtype=torch.float32
            ),
            "global_features": torch.tensor(
                self.global_features[idx], dtype=torch.float32
            ),
        }

        if self.targets is not None:
            sample["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.ids is not None:
            sample["id"] = torch.tensor(self.ids[idx], dtype=torch.long)

        return sample


def collate_fn(batch):
    """
    Collate function to handle variable number of atoms via padding.
    """
    max_atoms = Config.MAX_ATOMS
    atomic_dim = Config.ATOMIC_INPUT_DIM

    batch_size = len(batch)

    # Initialize tensors
    padded_atomic = torch.zeros(batch_size, max_atoms, atomic_dim)
    mask = torch.zeros(batch_size, max_atoms)
    global_feats = []
    targets = []
    ids = []

    for i, item in enumerate(batch):
        # Atomic features
        atoms = item["atomic_features"]
        num_atoms = atoms.shape[0]
        # Truncate if necessary (though MAX_ATOMS should cover it)
        num_atoms = min(num_atoms, max_atoms)

        padded_atomic[i, :num_atoms, :] = atoms[:num_atoms, :]
        mask[i, :num_atoms] = 1.0

        # Global features
        global_feats.append(item["global_features"])

        # Targets and IDs
        if "targets" in item:
            targets.append(item["targets"])
        if "id" in item:
            ids.append(item["id"])

    global_feats = torch.stack(global_feats)

    batch_out = {
        "atomic_features": padded_atomic,
        "global_features": global_feats,
        "mask": mask,
    }

    if targets:
        batch_out["targets"] = torch.stack(targets)
    if ids:
        batch_out["ids"] = torch.stack(ids)

    return batch_out


def process_dataset(metadata_path, cache_path, load_cached_data=True):
    """
    Processes raw data into features and targets, with caching.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {
                "atomic_features": list(
                    data["atomic_features"]
                ),  # Convert back to list of arrays
                "global_features": data["global_features"],
                "targets": data["targets"] if "targets" in data else None,
                "ids": data["ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    atomic_features_list = []
    global_features_list = []
    targets_list = []
    ids_list = []

    atom_type_map = {sym: i for i, sym in enumerate(Config.ATOM_TYPES)}

    for _, row in df.iterrows():
        # --- Parse Geometry ---
        xyz_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        lattice, types, coords = parse_xyz(xyz_path)
        n_atoms = len(types)

        # Validation Check (Cite debug_lesson_11)
        # Skip if lattice is invalid or no atoms found
        if lattice.shape != (3, 3) or n_atoms == 0:
            # print(f"Skipping malformed file: {xyz_path}")
            continue

        # --- Atomic Features ---
        # 1. One-hot encoding
        one_hot = np.zeros((n_atoms, len(Config.ATOM_TYPES)))
        for i, t in enumerate(types):
            if t in atom_type_map:
                one_hot[i, atom_type_map[t]] = 1.0

        # 2. Centered Cartesian Coordinates
        centered_coords = center_coordinates(coords)

        # 3. Fractional Coordinates
        frac_coords = cartesian_to_fractional(coords, lattice)

        # 4. Derived Spatial Features
        dist_matrix = compute_pbc_distances(coords, lattice)

        # Nearest Neighbor Distance
        # Set diagonal to infinity to ignore self-distance
        dist_matrix_inf = dist_matrix.copy()
        np.fill_diagonal(dist_matrix_inf, np.inf)
        nn_dist = np.min(dist_matrix_inf, axis=1).reshape(-1, 1)

        # Potential Proxy
        potential = compute_local_potential(dist_matrix).reshape(-1, 1)

        # Concatenate Atomic Features
        # [OneHot(4), Centered(3), Frac(3), NN(1), Pot(1)] -> Dim 12
        atom_feats = np.hstack(
            [one_hot, centered_coords, frac_coords, nn_dist, potential]
        )
        atomic_features_list.append(atom_feats.astype(np.float32))

        # --- Global Features ---
        # 1. Lattice Parameters
        lengths, angles = get_lattice_params(lattice)

        # 2. Volume & Density
        vol = get_cell_volume(lattice)
        density = n_atoms / vol

        # 3. Stoichiometry & Total Atoms
        # From CSV: percent_atom_al, percent_atom_ga, percent_atom_in
        stoich = np.array(
            [row["percent_atom_al"], row["percent_atom_ga"], row["percent_atom_in"]]
        )
        total_atoms = row["number_of_total_atoms"]

        # Concatenate Global Features
        # [Lengths(3), Angles(3), Vol(1), Density(1), Stoich(3), Total(1)] -> Dim 12
        glob_feats = np.concatenate(
            [lengths, angles, [vol], [density], stoich, [total_atoms]]
        )
        global_features_list.append(glob_feats.astype(np.float32))

        # --- Targets & ID ---
        if "formation_energy_ev_natom" in row and "bandgap_energy_ev" in row:
            targets_list.append(
                [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
            )

        ids_list.append(row["id"])

    # Convert lists to arrays for saving
    # atomic_features_list is a list of variable-length arrays, so we keep it as object array or list
    # but np.savez handles arrays. We will use object array for atomic features.
    atomic_features_arr = np.array(atomic_features_list, dtype=object)
    global_features_arr = np.array(global_features_list, dtype=np.float32)
    ids_arr = np.array(ids_list, dtype=np.int64)

    if targets_list:
        targets_arr = np.array(targets_list, dtype=np.float32)
    else:
        targets_arr = None

    # 3. Save to Cache
    save_dict = {
        "atomic_features": atomic_features_arr,
        "global_features": global_features_arr,
        "ids": ids_arr,
    }
    if targets_arr is not None:
        save_dict["targets"] = targets_arr

    np.savez(cache_path, **save_dict)

    # Return in expected format
    return {
        "atomic_features": atomic_features_list,
        "global_features": global_features_arr,
        "targets": targets_arr,
        "ids": ids_arr,
    }


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders for Train, Val, and Test.
    Handles scaling and target transformation.
    """
    # 1. Load Data
    train_data = process_dataset(
        os.path.join(Config.METADATA_DIR, "train.csv"),
        Config.TRAIN_CACHE,
        load_cached_data,
    )
    val_data = process_dataset(
        os.path.join(Config.METADATA_DIR, "val.csv"), Config.VAL_CACHE, load_cached_data
    )
    test_data = process_dataset(
        os.path.join(Config.METADATA_DIR, "test.csv"),
        Config.TEST_CACHE,
        load_cached_data,
    )

    # 2. Fit Scalers on Training Data
    # Atomic Scaler: Scale indices 4:12 (Continuous features)
    # Exclude One-Hot (0:4)
    atomic_scaler = StandardScaler()

    # Flatten all atomic features from training set to fit scaler
    # Note: Only include samples that actually have atoms
    valid_train_atoms = [a for a in train_data["atomic_features"] if a.shape[0] > 0]

    if valid_train_atoms:
        all_train_atoms = np.vstack(valid_train_atoms)
        atomic_scaler.fit(all_train_atoms[:, 4:])
    else:
        # Fallback if training set is empty (unlikely)
        print("Warning: No valid atomic features found for scaling.")

    # Global Scaler: Scale all global features
    global_scaler = StandardScaler()
    if len(train_data["global_features"]) > 0:
        global_scaler.fit(train_data["global_features"])

    # 3. Transform Data (Helper Function)
    def transform_data(data_dict):
        # Atomic Scaling
        transformed_atomic = []
        for atoms in data_dict["atomic_features"]:
            # Copy to avoid modifying original if cached
            a = atoms.copy()
            if a.shape[0] > 0:
                a[:, 4:] = atomic_scaler.transform(a[:, 4:])
            transformed_atomic.append(a)

        # Global Scaling
        transformed_global = global_scaler.transform(data_dict["global_features"])

        # Target Log Transformation log(1+y)
        transformed_targets = None
        if data_dict["targets"] is not None:
            transformed_targets = np.log1p(data_dict["targets"])

        return (
            transformed_atomic,
            transformed_global,
            transformed_targets,
            data_dict["ids"],
        )

    train_atomic, train_global, train_targets, train_ids = transform_data(train_data)
    val_atomic, val_global, val_targets, val_ids = transform_data(val_data)
    test_atomic, test_global, _, test_ids = transform_data(test_data)

    # 4. Create Datasets
    train_dataset = MaterialDataset(
        train_atomic, train_global, train_targets, train_ids
    )
    val_dataset = MaterialDataset(val_atomic, val_global, val_targets, val_ids)
    test_dataset = MaterialDataset(test_atomic, test_global, None, test_ids)

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader
