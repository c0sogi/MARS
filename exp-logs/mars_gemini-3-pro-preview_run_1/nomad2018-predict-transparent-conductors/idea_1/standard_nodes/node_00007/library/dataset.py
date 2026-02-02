import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from library.config import Config


def parse_xyz(file_path):
    """
    Parses an XYZ file to extract lattice vectors, atomic positions, and types.
    """
    lattice_vectors = []
    atom_positions = []
    atom_types = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "lattice_vector":
            lattice_vectors.append([float(x) for x in parts[1:4]])
        elif parts[0] == "atom":
            atom_positions.append([float(x) for x in parts[1:4]])
            atom_types.append(parts[4])

    return np.array(lattice_vectors), np.array(atom_positions), atom_types


def process_split(metadata_path, split_name, load_cached_data=True):
    """
    Loads and processes data for a specific split, with caching.
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"{split_name}_data.npz")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {split_name} data from {cache_file}...")
        try:
            data = np.load(cache_file, allow_pickle=False)
            return dict(data)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing {split_name} data from scratch...")
    df = pd.read_csv(metadata_path)

    all_atom_types = []
    all_positions = []
    sample_indices = []  # Maps atoms to samples
    lattice_features_list = []
    targets_list = []
    ids_list = []

    # Pre-compute indices for atom mapping
    atom_map = Config.ATOM_MAP

    for idx, row in df.iterrows():
        sample_id = row["id"]
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Parse Geometry
        _, positions, types = parse_xyz(full_path)

        # Center positions
        if len(positions) > 0:
            centroid = np.mean(positions, axis=0)
            centered_positions = positions - centroid
        else:
            centered_positions = positions

        # Map types to integers
        type_indices = [atom_map[t] for t in types]

        # Append to lists
        n_atoms = len(positions)
        all_atom_types.extend(type_indices)
        all_positions.extend(centered_positions)
        sample_indices.extend([idx] * n_atoms)

        # Tabular features
        feats = row[Config.TABULAR_FEATURE_COLS].values.astype(np.float32)
        lattice_features_list.append(feats)

        # Targets (if available)
        if split_name != "test":
            t = row[Config.TARGET_COLS].values.astype(np.float32)
            targets_list.append(t)

        ids_list.append(sample_id)

    # Convert to numpy arrays
    result = {
        "atom_types": np.array(all_atom_types, dtype=np.int64),
        "positions": np.array(all_positions, dtype=np.float32),
        "sample_indices": np.array(sample_indices, dtype=np.int64),
        "lattice_features": np.array(lattice_features_list, dtype=np.float32),
        "ids": np.array(ids_list, dtype=np.int64),
    }

    if split_name != "test":
        result["targets"] = np.array(targets_list, dtype=np.float32)
    else:
        # Placeholder for test targets
        result["targets"] = np.zeros((len(ids_list), 2), dtype=np.float32)

    # Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.savez(cache_file, **result)
    print(f"Saved {split_name} data to {cache_file}")

    return result


class CrystalDataset(InMemoryDataset):
    def __init__(self, data_dict, scaler=None, transform=None, pre_transform=None):
        self.data_dict = data_dict
        self.scaler = scaler
        super().__init__(".", transform, pre_transform)
        self.data, self.slices = self.process_data()

    def process_data(self):
        data_list = []

        atom_types_all = self.data_dict["atom_types"]
        positions_all = self.data_dict["positions"]
        sample_indices = self.data_dict["sample_indices"]
        lattice_features_all = self.data_dict["lattice_features"]
        targets_all = self.data_dict["targets"]
        ids_all = self.data_dict["ids"]

        # Normalize lattice features if scaler is provided
        if self.scaler:
            lattice_features_all = (
                lattice_features_all - self.scaler["mean"]
            ) / self.scaler["std"]
            # Handle potential division by zero or static columns (though unlikely for these features)
            lattice_features_all = np.nan_to_num(lattice_features_all)

        # Group atoms by sample index
        # We assume sample_indices are sorted 0, 0, ..., 1, 1, ... which they are by construction
        # Find split points
        counts = np.bincount(sample_indices, minlength=len(ids_all))

        current_atom_idx = 0
        for i in range(len(ids_all)):
            n_atoms = counts[i]

            # Extract atom data
            atom_types = atom_types_all[current_atom_idx : current_atom_idx + n_atoms]
            pos = positions_all[current_atom_idx : current_atom_idx + n_atoms]
            current_atom_idx += n_atoms

            # Create features
            # One-hot encoding for atom types
            x_one_hot = torch.zeros(
                (n_atoms, Config.NUM_ATOM_TYPES), dtype=torch.float32
            )
            x_one_hot[range(n_atoms), atom_types] = 1.0

            # Concatenate pos to x as per model requirement
            pos_tensor = torch.tensor(pos, dtype=torch.float32)
            x = torch.cat([pos_tensor, x_one_hot], dim=1)

            # Lattice features
            lattice = torch.tensor(
                lattice_features_all[i], dtype=torch.float32
            ).unsqueeze(
                0
            )  # (1, F)

            # Targets: Apply log(1+x)
            y_val = targets_all[i]
            y_log = np.log1p(y_val)
            y = torch.tensor(y_log, dtype=torch.float32).unsqueeze(0)  # (1, 2)

            sample_id = int(ids_all[i])

            data = Data(
                x=x, pos=pos_tensor, lattice_features=lattice, y=y, id=sample_id
            )
            data_list.append(data)

        return self.collate(data_list)


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Prepares DataLoaders for train, validation, and test sets.
    """
    # 1. Process all splits
    train_data = process_split(Config.TRAIN_METADATA_PATH, "train", load_cached_data)
    val_data = process_split(Config.VAL_METADATA_PATH, "val", load_cached_data)
    test_data = process_split(Config.TEST_METADATA_PATH, "test", load_cached_data)

    # 2. Compute Scaler from Train
    train_lattice = train_data["lattice_features"]
    scaler = {
        "mean": np.mean(train_lattice, axis=0),
        "std": np.std(train_lattice, axis=0) + 1e-8,  # Avoid div by zero
    }

    # 3. Create Datasets
    train_dataset = CrystalDataset(train_data, scaler=scaler)
    val_dataset = CrystalDataset(val_data, scaler=scaler)
    test_dataset = CrystalDataset(test_data, scaler=scaler)

    # 4. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
