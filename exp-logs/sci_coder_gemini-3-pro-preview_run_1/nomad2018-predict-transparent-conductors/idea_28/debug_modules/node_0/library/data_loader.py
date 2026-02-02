import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.geometry_utils import parse_xyz, calculate_atomic_features


class MaterialDataset(Dataset):
    def __init__(self, atomic_features_list, global_features, targets=None, ids=None):
        """
        Args:
            atomic_features_list (list of np.ndarray): List where each element is (N_atoms, D_atomic)
            global_features (np.ndarray): Array of shape (N_samples, D_global)
            targets (np.ndarray, optional): Array of shape (N_samples, 2)
            ids (np.ndarray, optional): Array of shape (N_samples,)
        """
        self.atomic_features_list = atomic_features_list
        self.global_features = torch.tensor(global_features, dtype=torch.float32)

        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

        if ids is not None:
            self.ids = torch.tensor(ids, dtype=torch.long)
        else:
            self.ids = None

    def __len__(self):
        return len(self.atomic_features_list)

    def __getitem__(self, idx):
        atomic_feats = torch.tensor(self.atomic_features_list[idx], dtype=torch.float32)
        global_feats = self.global_features[idx]

        item = {
            "atomic_features": atomic_feats,
            "global_features": global_feats,
        }

        if self.targets is not None:
            item["targets"] = self.targets[idx]

        if self.ids is not None:
            item["id"] = self.ids[idx]

        return item


def collate_fn(batch):
    """
    Custom collate function to handle variable number of atoms.
    """
    batch_atomic_features = []
    batch_indices = []
    batch_global_features = []
    batch_targets = []
    batch_ids = []

    for i, item in enumerate(batch):
        # Atomic features
        atoms = item["atomic_features"]
        batch_atomic_features.append(atoms)

        # Create batch index for these atoms (e.g., [0, 0, 0, 1, 1, ...])
        num_atoms = atoms.shape[0]
        batch_indices.append(torch.full((num_atoms,), i, dtype=torch.long))

        # Global features
        batch_global_features.append(item["global_features"])

        # Targets
        if "targets" in item:
            batch_targets.append(item["targets"])

        # IDs
        if "id" in item:
            batch_ids.append(item["id"])

    # Concatenate everything
    batch_atomic_features = torch.cat(batch_atomic_features, dim=0)
    batch_indices = torch.cat(batch_indices, dim=0)
    batch_global_features = torch.stack(batch_global_features, dim=0)

    result = {
        "atomic_features": batch_atomic_features,
        "batch_indices": batch_indices,
        "global_features": batch_global_features,
    }

    if batch_targets:
        result["targets"] = torch.stack(batch_targets, dim=0)

    if batch_ids:
        result["ids"] = torch.stack(batch_ids, dim=0)

    return result


def calculate_cell_volume(lattice_vectors):
    return np.abs(
        np.dot(lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2]))
    )


def calculate_lattice_params(lattice_vectors):
    # Lengths
    a = np.linalg.norm(lattice_vectors[0])
    b = np.linalg.norm(lattice_vectors[1])
    c = np.linalg.norm(lattice_vectors[2])

    # Angles
    alpha = np.degrees(
        np.arccos(np.dot(lattice_vectors[1], lattice_vectors[2]) / (b * c))
    )
    beta = np.degrees(
        np.arccos(np.dot(lattice_vectors[0], lattice_vectors[2]) / (a * c))
    )
    gamma = np.degrees(
        np.arccos(np.dot(lattice_vectors[0], lattice_vectors[1]) / (a * b))
    )

    return [a, b, c, alpha, beta, gamma]


def process_data(load_cached_data=True, sample_size=None):
    """
    Main function to load, process, and return DataLoaders.
    """

    # Check if cache exists
    cache_files_exist = (
        os.path.exists(Config.TRAIN_DATA_CACHE)
        and os.path.exists(Config.VAL_DATA_CACHE)
        and os.path.exists(Config.TEST_DATA_CACHE)
        and os.path.exists(Config.SCALERS_PATH)
    )

    if load_cached_data and cache_files_exist:
        print("Loading cached data...")
        try:
            train_data = np.load(Config.TRAIN_DATA_CACHE, allow_pickle=True)
            val_data = np.load(Config.VAL_DATA_CACHE, allow_pickle=True)
            test_data = np.load(Config.TEST_DATA_CACHE, allow_pickle=True)

            # Reconstruct lists of arrays for atomic features
            # Train
            train_atomic_flat = train_data["atomic_features_flat"]
            train_counts = train_data["atom_counts"]
            train_atomic_list = []
            idx = 0
            for count in train_counts:
                train_atomic_list.append(train_atomic_flat[idx : idx + count])
                idx += count

            # Val
            val_atomic_flat = val_data["atomic_features_flat"]
            val_counts = val_data["atom_counts"]
            val_atomic_list = []
            idx = 0
            for count in val_counts:
                val_atomic_list.append(val_atomic_flat[idx : idx + count])
                idx += count

            # Test
            test_atomic_flat = test_data["atomic_features_flat"]
            test_counts = test_data["atom_counts"]
            test_atomic_list = []
            idx = 0
            for count in test_counts:
                test_atomic_list.append(test_atomic_flat[idx : idx + count])
                idx += count

            train_dataset = MaterialDataset(
                train_atomic_list,
                train_data["global_features"],
                train_data["targets"],
                train_data["ids"],
            )
            val_dataset = MaterialDataset(
                val_atomic_list,
                val_data["global_features"],
                val_data["targets"],
                val_data["ids"],
            )
            test_dataset = MaterialDataset(
                test_atomic_list, test_data["global_features"], None, test_data["ids"]
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=Config.NUM_WORKERS,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=Config.NUM_WORKERS,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=Config.NUM_WORKERS,
            )

            return train_loader, val_loader, test_loader

        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Compute from scratch
    print("Processing data from scratch...")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if sample_size is not None:
        train_df = train_df.iloc[:sample_size]
        val_df = val_df.iloc[: min(sample_size, len(val_df))]
        test_df = test_df.iloc[: min(sample_size, len(test_df))]

    def extract_features(df, is_test=False):
        atomic_feats_list = []
        global_feats_list = []
        targets_list = []
        ids_list = []

        for _, row in df.iterrows():
            # 1. Geometry Features
            file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
            lattice_vectors, atom_types, atom_coords = parse_xyz(file_path)

            # Atomic features (12 dims)
            # One-hot(4) + Coords(3) + RecipProx(4) + Packing(1)
            af = calculate_atomic_features(atom_types, atom_coords, lattice_vectors)
            atomic_feats_list.append(af)

            # 2. Global Features
            # Lattice params (6)
            lat_params = calculate_lattice_params(lattice_vectors)
            # Volume (1)
            vol = calculate_cell_volume(lattice_vectors)
            # Total atoms (1)
            n_atoms = len(atom_types)
            # Density (1)
            density = n_atoms / vol
            # Composition (3) - Al, Ga, In (O is implicit as 1 - sum, but usually we just use the 3 cations)
            comp = [
                row["percent_atom_al"],
                row["percent_atom_ga"],
                row["percent_atom_in"],
            ]

            # Total Global: 6 + 1 + 1 + 1 + 3 = 12 dims
            gf = lat_params + [vol, density, n_atoms] + comp
            global_feats_list.append(gf)

            # 3. Targets & IDs
            ids_list.append(row["id"])
            if not is_test:
                # Log transform targets: log(1 + y)
                t = [
                    np.log1p(row["formation_energy_ev_natom"]),
                    np.log1p(row["bandgap_energy_ev"]),
                ]
                targets_list.append(t)

        return (
            atomic_feats_list,
            np.array(global_feats_list),
            np.array(targets_list) if not is_test else None,
            np.array(ids_list),
        )

    print("Extracting training features...")
    train_atomic, train_global, train_targets, train_ids = extract_features(train_df)
    print("Extracting validation features...")
    val_atomic, val_global, val_targets, val_ids = extract_features(val_df)
    print("Extracting test features...")
    test_atomic, test_global, _, test_ids = extract_features(test_df, is_test=True)

    # Scaling
    # Atomic features: Indices 4-12 (Coords: 4,5,6; RecipProx: 7,8,9,10; Packing: 11)
    # Global features: All 0-11

    # Flatten train atomic features to compute stats
    train_atomic_flat = np.vstack(train_atomic)

    # Atomic Scaler
    atomic_mean = np.mean(train_atomic_flat[:, 4:], axis=0)
    atomic_std = np.std(train_atomic_flat[:, 4:], axis=0)
    # Avoid division by zero
    atomic_std[atomic_std == 0] = 1.0

    # Global Scaler
    global_mean = np.mean(train_global, axis=0)
    global_std = np.std(train_global, axis=0)
    global_std[global_std == 0] = 1.0

    # Apply Scaling
    def scale_atomic(atomic_list, mean, std):
        scaled_list = []
        for af in atomic_list:
            af_new = af.copy()
            af_new[:, 4:] = (af[:, 4:] - mean) / std
            scaled_list.append(af_new)
        return scaled_list

    def scale_global(gf, mean, std):
        return (gf - mean) / std

    train_atomic = scale_atomic(train_atomic, atomic_mean, atomic_std)
    val_atomic = scale_atomic(val_atomic, atomic_mean, atomic_std)
    test_atomic = scale_atomic(test_atomic, atomic_mean, atomic_std)

    train_global = scale_global(train_global, global_mean, global_std)
    val_global = scale_global(val_global, global_mean, global_std)
    test_global = scale_global(test_global, global_mean, global_std)

    # Save to cache
    print("Saving data to cache...")

    # Helper to save atomic lists
    def save_dataset(path, atomic_list, global_arr, targets_arr, ids_arr):
        # Flatten atomic list and store counts
        counts = np.array([len(a) for a in atomic_list])
        flat = np.vstack(atomic_list)

        save_dict = {
            "atomic_features_flat": flat,
            "atom_counts": counts,
            "global_features": global_arr,
            "ids": ids_arr,
        }
        if targets_arr is not None:
            save_dict["targets"] = targets_arr

        np.savez(path, **save_dict)

    save_dataset(
        Config.TRAIN_DATA_CACHE, train_atomic, train_global, train_targets, train_ids
    )
    save_dataset(Config.VAL_DATA_CACHE, val_atomic, val_global, val_targets, val_ids)
    save_dataset(Config.TEST_DATA_CACHE, test_atomic, test_global, None, test_ids)

    np.savez(
        Config.SCALERS_PATH,
        atomic_mean=atomic_mean,
        atomic_std=atomic_std,
        global_mean=global_mean,
        global_std=global_std,
    )

    # Create DataLoaders
    train_dataset = MaterialDataset(
        train_atomic, train_global, train_targets, train_ids
    )
    val_dataset = MaterialDataset(val_atomic, val_global, val_targets, val_ids)
    test_dataset = MaterialDataset(test_atomic, test_global, None, test_ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
    )

    return train_loader, val_loader, test_loader
