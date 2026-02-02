import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.geometry_utils import (
    parse_xyz,
    get_cell_volume,
    get_atomic_density,
    compute_local_anisotropy,
    get_centered_coordinates,
)

# Atom mapping for one-hot encoding
ATOM_MAP = {sym: i for i, sym in enumerate(Config.ATOM_TYPES)}


def get_one_hot(atom_types):
    """
    Converts a list of atomic symbols to one-hot encoding.
    """
    one_hot = np.zeros((len(atom_types), Config.NUM_ATOM_TYPES), dtype=np.float32)
    for i, sym in enumerate(atom_types):
        if sym in ATOM_MAP:
            one_hot[i, ATOM_MAP[sym]] = 1.0
    return one_hot


def process_split(df, input_dir):
    """
    Extracts features for a given dataframe split.
    Returns flattened arrays suitable for caching.
    """
    all_atomic_feats = []
    all_global_feats = []
    all_targets = []
    all_ids = []

    # Iterate through each material in the dataframe
    for _, row in df.iterrows():
        crystal_id = row["id"]
        rel_path = row["file_path"]
        full_path = os.path.join(input_dir, rel_path)

        # 1. Parse Geometry
        lattice_vectors, atom_types, coords = parse_xyz(full_path)
        num_atoms = len(atom_types)

        # 2. Atomic Features
        # One-hot encoding
        one_hot = get_one_hot(atom_types)

        # Centered Coordinates
        centered_coords = get_centered_coordinates(coords, lattice_vectors)

        # Local Anisotropy & Nearest Neighbor Distance
        eigenvalues, nn_dists = compute_local_anisotropy(
            coords, lattice_vectors, k_neighbors=Config.K_NEIGHBORS
        )

        # Concatenate atomic features: [One-hot (4), Coords (3), NN (1), Eigs (3)]
        atomic_feats = np.hstack([one_hot, centered_coords, nn_dists, eigenvalues])
        all_atomic_feats.append(atomic_feats)

        # 3. Global Features
        # Extract lattice parameters from metadata
        l1 = row["lattice_vector_1_ang"]
        l2 = row["lattice_vector_2_ang"]
        l3 = row["lattice_vector_3_ang"]
        alpha = row["lattice_angle_alpha_degree"]
        beta = row["lattice_angle_beta_degree"]
        gamma = row["lattice_angle_gamma_degree"]

        # Derived physical properties
        vol = get_cell_volume(lattice_vectors)
        density = get_atomic_density(num_atoms, vol)

        # Stoichiometry (normalized counts)
        counts = {t: 0 for t in Config.ATOM_TYPES}
        for t in atom_types:
            if t in counts:
                counts[t] += 1
        stoich = [counts[t] / num_atoms for t in Config.ATOM_TYPES]

        # Construct global feature vector (13 dims)
        global_feat = np.array(
            [l1, l2, l3, alpha, beta, gamma, vol, density] + stoich + [num_atoms],
            dtype=np.float32,
        )
        all_global_feats.append(global_feat)

        # 4. Targets
        if "formation_energy_ev_natom" in row:
            t1 = row["formation_energy_ev_natom"]
            t2 = row["bandgap_energy_ev"]
            all_targets.append([t1, t2])
        else:
            # Placeholder for test set
            all_targets.append([0.0, 0.0])

        all_ids.append(crystal_id)

    # Flatten atomic features for efficient storage
    # We maintain an 'indices' array to reconstruct the ragged structure
    flat_atomic = np.vstack(all_atomic_feats).astype(np.float32)
    lengths = np.array([len(a) for a in all_atomic_feats], dtype=np.int32)
    indices = np.concatenate(([0], np.cumsum(lengths)))

    global_feats = np.vstack(all_global_feats).astype(np.float32)
    targets = np.vstack(all_targets).astype(np.float32)
    ids = np.array(all_ids, dtype=np.int32)

    return {
        "flat_atomic": flat_atomic,
        "indices": indices,
        "global_feats": global_feats,
        "targets": targets,
        "ids": ids,
    }


class MaterialsDataset(Dataset):
    def __init__(
        self, data_dict, atomic_scaler=None, global_scaler=None, is_train=False
    ):
        self.flat_atomic = data_dict["flat_atomic"]
        self.indices = data_dict["indices"]
        self.global_feats = data_dict["global_feats"]
        self.targets = data_dict["targets"]
        self.ids = data_dict["ids"]
        self.is_train = is_train

        # Apply Scaling
        if atomic_scaler:
            self.flat_atomic = atomic_scaler.transform(self.flat_atomic)
        if global_scaler:
            self.global_feats = global_scaler.transform(self.global_feats)

        # Log transform targets to align MSE loss with RMSLE metric
        # log(1 + y) ensures non-negative energies are handled correctly
        self.targets = np.log1p(self.targets)

        # Convert to PyTorch tensors
        self.flat_atomic = torch.from_numpy(self.flat_atomic).float()
        self.global_feats = torch.from_numpy(self.global_feats).float()
        self.targets = torch.from_numpy(self.targets).float()
        self.indices = torch.from_numpy(self.indices).long()
        self.ids = torch.from_numpy(self.ids).long()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Retrieve atomic features for the specific crystal using pre-calculated indices
        start = self.indices[idx]
        end = self.indices[idx + 1]
        atom_x = self.flat_atomic[start:end]

        global_x = self.global_feats[idx]
        y = self.targets[idx]
        crystal_id = self.ids[idx]

        return atom_x, global_x, y, crystal_id


def collate_fn(batch):
    """
    Custom collate function to handle variable number of atoms per crystal.
    It concatenates atomic features and creates a batch index vector.
    """
    atom_x_list, global_x_list, y_list, id_list = zip(*batch)

    # Concatenate all atomic features from the batch into a single tensor
    batch_atom_x = torch.cat(atom_x_list, dim=0)

    # Create batch indices vector (e.g., [0, 0, 0, 1, 1, 1, ...])
    # This maps each atom to its corresponding crystal in the batch
    batch_indices = []
    for i, x in enumerate(atom_x_list):
        batch_indices.append(torch.full((x.shape[0],), i, dtype=torch.long))
    batch_indices = torch.cat(batch_indices, dim=0)

    # Stack global features and targets normally
    batch_global_x = torch.stack(global_x_list, dim=0)
    batch_y = torch.stack(y_list, dim=0)
    batch_ids = torch.stack(id_list, dim=0)

    return batch_atom_x, batch_indices, batch_global_x, batch_y, batch_ids


def get_dataloaders(
    load_cached_data=True, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Main entry point to get dataloaders.
    Handles caching, preprocessing, and scaling.
    """
    Config.prepare_directories()

    splits = ["train", "val", "test"]
    paths = {
        "train": Config.TRAIN_META_PATH,
        "val": Config.VAL_META_PATH,
        "test": Config.TEST_META_PATH,
    }

    data_dicts = {}

    # 1. Load or Compute Data
    for split in splits:
        cache_path = os.path.join(Config.CACHE_DIR, f"{split}_data.npz")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached {split} data from {cache_path}...")
            loaded = np.load(cache_path)
            data_dicts[split] = {k: loaded[k] for k in loaded.files}
        else:
            print(f"Processing {split} data...")
            df = pd.read_csv(paths[split])
            data = process_split(df, Config.INPUT_DIR)
            np.savez(cache_path, **data)
            data_dicts[split] = data

    # 2. Fit Scalers on Training Data
    print("Fitting scalers on training data...")
    train_data = data_dicts["train"]

    atomic_scaler = StandardScaler()
    atomic_scaler.fit(train_data["flat_atomic"])

    global_scaler = StandardScaler()
    global_scaler.fit(train_data["global_feats"])

    # 3. Create Datasets
    train_dataset = MaterialsDataset(
        data_dicts["train"], atomic_scaler, global_scaler, is_train=True
    )
    val_dataset = MaterialsDataset(
        data_dicts["val"], atomic_scaler, global_scaler, is_train=False
    )
    test_dataset = MaterialsDataset(
        data_dicts["test"], atomic_scaler, global_scaler, is_train=False
    )

    # 4. Create Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
