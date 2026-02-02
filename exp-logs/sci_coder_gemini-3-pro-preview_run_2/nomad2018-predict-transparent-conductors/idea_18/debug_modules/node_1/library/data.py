import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from ase import Atoms
from ase.neighborlist import neighbor_list
from library.config import Config
from library.utils import get_scaler

# Map chemical symbols to atomic numbers for embedding
SYMBOL_TO_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
}


def parse_custom_xyz(file_path):
    """
    Parses the custom XYZ format provided in the dataset.
    Extracts the unit cell vectors and atomic positions.
    """
    cell = []
    positions = []
    symbols = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if line.startswith("lattice_vector"):
                vec = [float(x) for x in parts[1:4]]
                cell.append(vec)
            elif line.startswith("atom"):
                pos = [float(x) for x in parts[1:4]]
                sym = parts[4]
                positions.append(pos)
                symbols.append(sym)

    return np.array(cell), symbols, np.array(positions)


def build_crystal_graph(cell, symbols, positions, cutoff):
    """
    Constructs a graph from crystal structure using ASE neighbor list.
    """
    # Create ASE Atoms object
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

    # Compute neighbors
    # i: source index, j: target index, d: distance
    i_indices, j_indices, distances = neighbor_list("ijd", atoms, cutoff)

    # Construct PyG data components
    edge_index = torch.tensor(np.vstack((i_indices, j_indices)), dtype=torch.long)
    edge_attr = torch.tensor(distances, dtype=torch.float).unsqueeze(1)

    # Node features: Atomic numbers
    z_values = [SYMBOL_TO_Z[s] for s in symbols]
    x = torch.tensor(z_values, dtype=torch.long)

    return x, edge_index, edge_attr


def save_data_to_cache(data_list, path):
    """
    Saves a list of PyG Data objects to a compressed .npz file.
    Flattens the variable-size tensors and stores split indices.
    """
    if not data_list:
        return

    # Concatenate all features
    x_list = []
    edge_index_list = []
    edge_attr_list = []
    lattice_list = []
    y_list = []

    # Track sizes for reconstruction
    num_nodes = []
    num_edges = []

    for data in data_list:
        x_list.append(data.x.numpy())
        edge_index_list.append(data.edge_index.numpy())
        edge_attr_list.append(data.edge_attr.numpy())
        lattice_list.append(data.lattice_params.numpy())

        # Handle y which might be None for test set, but we usually initialize it to nan or 0
        if data.y is not None:
            y_list.append(data.y.numpy())
        else:
            y_list.append(np.full((1, 2), np.nan))

        num_nodes.append(data.num_nodes)
        num_edges.append(data.num_edges)

    # Stack/Concatenate
    x_all = np.concatenate(x_list, axis=0)
    edge_index_all = np.concatenate(edge_index_list, axis=1)
    edge_attr_all = np.concatenate(edge_attr_list, axis=0)
    lattice_all = np.concatenate(lattice_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    node_splits = np.cumsum([0] + num_nodes)
    edge_splits = np.cumsum([0] + num_edges)

    np.savez_compressed(
        path,
        x=x_all,
        edge_index=edge_index_all,
        edge_attr=edge_attr_all,
        lattice=lattice_all,
        y=y_all,
        node_splits=node_splits,
        edge_splits=edge_splits,
    )
    print(f"Saved {len(data_list)} graphs to cache: {path}")


def load_data_from_cache(path):
    """
    Loads a list of PyG Data objects from a compressed .npz file.
    """
    if not os.path.exists(path):
        return None

    try:
        data_dict = np.load(path)
        x_all = data_dict["x"]
        edge_index_all = data_dict["edge_index"]
        edge_attr_all = data_dict["edge_attr"]
        lattice_all = data_dict["lattice"]
        y_all = data_dict["y"]
        node_splits = data_dict["node_splits"]
        edge_splits = data_dict["edge_splits"]

        data_list = []
        num_graphs = len(node_splits) - 1

        for i in range(num_graphs):
            # Slicing indices
            n_start, n_end = node_splits[i], node_splits[i + 1]
            e_start, e_end = edge_splits[i], edge_splits[i + 1]

            # Reconstruct tensors
            x = torch.tensor(x_all[n_start:n_end], dtype=torch.long)
            edge_index = torch.tensor(
                edge_index_all[:, e_start:e_end], dtype=torch.long
            )
            edge_attr = torch.tensor(edge_attr_all[e_start:e_end], dtype=torch.float)
            lattice = torch.tensor(lattice_all[i : i + 1], dtype=torch.float)
            y = torch.tensor(y_all[i : i + 1], dtype=torch.float)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                lattice_params=lattice,
                y=y,
            )
            data_list.append(data)

        print(f"Loaded {len(data_list)} graphs from cache: {path}")
        return data_list
    except Exception as e:
        print(f"Failed to load cache from {path}: {e}")
        return None


class CrystalGraphDataset(Dataset):
    def __init__(self, data_list, transform=None):
        super().__init__(root=None, transform=transform)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def process_subset(metadata_path, cache_path, load_cached_data=True):
    """
    Process a subset of data (train/val/test) defined by the metadata CSV.
    Handles caching logic.
    """
    # 1. Try loading from cache
    if load_cached_data:
        cached_data = load_data_from_cache(cache_path)
        if cached_data is not None:
            return cached_data

    # 2. Compute from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    data_list = []

    # Lattice parameter columns
    lattice_cols = [
        "lattice_vector_1_ang",
        "lattice_vector_2_ang",
        "lattice_vector_3_ang",
        "lattice_angle_alpha_degree",
        "lattice_angle_beta_degree",
        "lattice_angle_gamma_degree",
    ]

    # Target columns
    target_cols = Config.TARGET_COLS

    for _, row in df.iterrows():
        # Geometry file path
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            print(f"Warning: File not found {full_path}")
            continue

        # Parse geometry
        cell, symbols, positions = parse_custom_xyz(full_path)

        # Build graph
        x, edge_index, edge_attr = build_crystal_graph(
            cell, symbols, positions, Config.CUTOFF_RADIUS
        )

        # Extract lattice features
        lattice_params = torch.tensor(
            [row[col] for col in lattice_cols], dtype=torch.float
        ).unsqueeze(0)

        # Extract targets (if available)
        if all(col in row for col in target_cols):
            y = torch.tensor(
                [row[col] for col in target_cols], dtype=torch.float
            ).unsqueeze(0)
        else:
            # Placeholder for test set
            y = torch.full((1, len(target_cols)), float("nan"), dtype=torch.float)

        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            lattice_params=lattice_params,
            y=y,
        )
        data_list.append(data)

    # 3. Save to cache
    save_data_to_cache(data_list, cache_path)

    return data_list


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get dataloaders for train, val, and test sets.
    Handles scaling of lattice parameters and targets.
    """
    Config.setup_directories()

    # 1. Load Data Lists (Train, Val, Test)
    train_list = process_subset(
        Config.TRAIN_METADATA_PATH, Config.TRAIN_GRAPHS_CACHE, load_cached_data
    )
    val_list = process_subset(
        Config.VAL_METADATA_PATH, Config.VAL_GRAPHS_CACHE, load_cached_data
    )
    test_list = process_subset(
        Config.TEST_METADATA_PATH, Config.TEST_GRAPHS_CACHE, load_cached_data
    )

    # 2. Fit Scalers on Training Data
    print("Fitting scalers...")

    # Collect training data for fitting
    train_lattice = np.concatenate(
        [d.lattice_params.numpy() for d in train_list], axis=0
    )
    train_targets = np.concatenate([d.y.numpy() for d in train_list], axis=0)

    # Fit and save Lattice Scaler
    lattice_scaler = get_scaler(
        train_lattice, Config.LATTICE_SCALER_PATH, load_cached_data
    )

    # Fit and save Target Scaler
    target_scaler = get_scaler(
        train_targets, Config.TARGET_SCALER_PATH, load_cached_data
    )

    # 3. Apply Scaling to All Datasets
    print("Applying scaling...")

    def apply_scaling(data_list, lat_scaler, tgt_scaler=None):
        for data in data_list:
            # Scale lattice parameters
            lat_scaled = lat_scaler.transform(data.lattice_params)
            data.lattice_params = torch.tensor(lat_scaled, dtype=torch.float)

            # Scale targets (if they exist and are not NaN)
            if tgt_scaler is not None and not torch.isnan(data.y).any():
                y_scaled = tgt_scaler.transform(data.y)
                data.y = torch.tensor(y_scaled, dtype=torch.float)

    apply_scaling(train_list, lattice_scaler, target_scaler)
    apply_scaling(val_list, lattice_scaler, target_scaler)
    apply_scaling(
        test_list, lattice_scaler, None
    )  # Don't scale test targets (they are NaN)

    # 4. Create Datasets
    train_dataset = CrystalGraphDataset(train_list)
    val_dataset = CrystalGraphDataset(val_list)
    test_dataset = CrystalGraphDataset(test_list)

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(
        f"DataLoaders created. Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}"
    )

    return train_loader, val_loader, test_loader
