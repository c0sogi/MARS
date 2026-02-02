import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from ase import Atoms
from ase.neighborlist import neighbor_list
from library.config import Config
from library.utils import TargetScaler


def parse_xyz(file_path):
    """
    Parses an XYZ file with lattice vectors into an ASE Atoms object.
    """
    lattice_vectors = []
    positions = []
    symbols = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "lattice_vector":
            lattice_vectors.append([float(x) for x in parts[1:4]])
        elif parts[0] == "atom":
            positions.append([float(x) for x in parts[1:4]])
            symbols.append(parts[4])

    # Create ASE Atoms object with Periodic Boundary Conditions
    atoms = Atoms(symbols=symbols, positions=positions, cell=lattice_vectors, pbc=True)
    return atoms


def build_knn_graph(atoms, k=12, cutoff=8.0):
    """
    Constructs a k-NN graph from an ASE Atoms object.

    Args:
        atoms: ASE Atoms object.
        k: Number of nearest neighbors.
        cutoff: Initial radius to search for neighbors (should be large enough to find k neighbors).

    Returns:
        torch_geometric.data.Data object.
    """
    # Get all neighbors within the cutoff
    # 'i': index of central atom, 'j': index of neighbor, 'd': distance
    i_indices, j_indices, distances = neighbor_list("ijd", atoms, cutoff)

    num_atoms = len(atoms)

    new_i = []
    new_j = []
    new_d = []

    # For each atom, select the k nearest neighbors
    for atom_idx in range(num_atoms):
        # Mask for neighbors of the current atom
        mask = i_indices == atom_idx

        if not np.any(mask):
            continue

        current_neighbors = j_indices[mask]
        current_dists = distances[mask]

        # Sort by distance
        sorted_arg = np.argsort(current_dists)

        # Select top k
        k_nearest_arg = sorted_arg[:k]

        for arg in k_nearest_arg:
            new_i.append(atom_idx)
            new_j.append(current_neighbors[arg])
            new_d.append(current_dists[arg])

    # Convert to PyTorch tensors
    edge_index = torch.tensor([new_i, new_j], dtype=torch.long)

    # Edge attributes: Distances (will be expanded by RBF in the model)
    edge_attr = torch.tensor(new_d, dtype=torch.float).unsqueeze(1)

    # Node features: Atomic numbers
    z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)

    return Data(x=z, edge_index=edge_index, edge_attr=edge_attr)


class CrystalGraphDataset(InMemoryDataset):
    def __init__(self, data_list, root=None, transform=None, pre_transform=None):
        self.data_list = data_list
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass


def process_dataset(metadata_path, k_neighbors, rbf_cutoff, sample_size=None):
    """
    Reads metadata, loads XYZ files, and constructs graphs.
    """
    df = pd.read_csv(metadata_path)

    if sample_size is not None:
        df = df.head(sample_size)

    data_list = []

    for _, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            continue

        atoms = parse_xyz(full_path)
        data = build_knn_graph(atoms, k=k_neighbors, cutoff=rbf_cutoff)

        # Add targets if they exist (train/val)
        if "formation_energy_ev_natom" in row and "bandgap_energy_ev" in row:
            y = torch.tensor(
                [[row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]],
                dtype=torch.float,
            )
            data.y = y
        else:
            # Placeholder for test
            data.y = None

        # Add ID
        data.id = torch.tensor([row["id"]], dtype=torch.long)

        data_list.append(data)

    return data_list


def save_graphs(data_list, path):
    """
    Saves a list of Data objects to a .npz file using numpy arrays.
    Avoids pickle.
    """
    all_x = []
    all_edge_index = []
    all_edge_attr = []
    all_y = []
    all_ids = []

    num_nodes_list = []
    num_edges_list = []

    for data in data_list:
        all_x.append(data.x.numpy())
        all_edge_index.append(data.edge_index.numpy())
        all_edge_attr.append(data.edge_attr.numpy())

        if data.y is not None:
            all_y.append(data.y.numpy())
        else:
            # Use NaN for missing targets
            all_y.append(np.array([[np.nan, np.nan]]))

        all_ids.append(data.id.numpy())

        num_nodes_list.append(data.num_nodes)
        num_edges_list.append(data.num_edges)

    # Concatenate arrays
    x_concat = np.concatenate(all_x)
    edge_index_concat = np.concatenate(all_edge_index, axis=1)
    edge_attr_concat = np.concatenate(all_edge_attr)
    y_concat = np.concatenate(all_y)
    ids_concat = np.concatenate(all_ids)

    num_nodes_arr = np.array(num_nodes_list)
    num_edges_arr = np.array(num_edges_list)

    np.savez(
        path,
        x=x_concat,
        edge_index=edge_index_concat,
        edge_attr=edge_attr_concat,
        y=y_concat,
        ids=ids_concat,
        num_nodes=num_nodes_arr,
        num_edges=num_edges_arr,
    )


def load_graphs(path):
    """
    Loads Data objects from a .npz file.
    """
    data = np.load(path)
    x_concat = data["x"]
    edge_index_concat = data["edge_index"]
    edge_attr_concat = data["edge_attr"]
    y_concat = data["y"]
    ids_concat = data["ids"]
    num_nodes_arr = data["num_nodes"]
    num_edges_arr = data["num_edges"]

    data_list = []

    node_offset = 0
    edge_offset = 0

    for i in range(len(num_nodes_arr)):
        n_nodes = num_nodes_arr[i]
        n_edges = num_edges_arr[i]

        # Slice node features
        x = torch.tensor(
            x_concat[node_offset : node_offset + n_nodes], dtype=torch.long
        )

        # Slice edge indices
        ei = torch.tensor(
            edge_index_concat[:, edge_offset : edge_offset + n_edges], dtype=torch.long
        )

        # Slice edge attributes
        ea = torch.tensor(
            edge_attr_concat[edge_offset : edge_offset + n_edges], dtype=torch.float
        )

        # Slice target
        y_val = y_concat[i : i + 1]
        if np.isnan(y_val).any():
            y = None
        else:
            y = torch.tensor(y_val, dtype=torch.float)

        # Slice ID
        id_val = torch.tensor(ids_concat[i : i + 1], dtype=torch.long)

        d = Data(x=x, edge_index=ei, edge_attr=ea, y=y)
        d.id = id_val

        data_list.append(d)

        node_offset += n_nodes
        edge_offset += n_edges

    return data_list


def get_dataset(split, load_cached_data=True):
    """
    Factory function to retrieve the dataset for a given split.
    Implements caching logic.
    """
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
        cache_path = Config.TRAIN_GRAPHS_PATH
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
        cache_path = Config.VAL_GRAPHS_PATH
    elif split == "test":
        meta_path = Config.TEST_METADATA_PATH
        cache_path = Config.TEST_GRAPHS_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Check cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split} graphs from {cache_path}...")
        try:
            data_list = load_graphs(cache_path)
            if Config.DEBUG_SAMPLE_SIZE and len(data_list) > Config.DEBUG_SAMPLE_SIZE:
                data_list = data_list[: Config.DEBUG_SAMPLE_SIZE]
            return CrystalGraphDataset(data_list)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Compute
    print(f"Processing {split} data from {meta_path}...")
    data_list = process_dataset(
        meta_path,
        k_neighbors=Config.K_NEIGHBORS,
        rbf_cutoff=Config.RBF_CUTOFF,
        sample_size=Config.DEBUG_SAMPLE_SIZE,
    )

    # Save cache
    print(f"Saving {split} graphs to {cache_path}...")
    save_graphs(data_list, cache_path)

    return CrystalGraphDataset(data_list)


def get_dataloaders(load_cached_data=True):
    """
    Constructs DataLoaders for train, val, and test sets.
    Fits and saves the TargetScaler on the training data.
    """
    train_dataset = get_dataset("train", load_cached_data)
    val_dataset = get_dataset("val", load_cached_data)
    test_dataset = get_dataset("test", load_cached_data)

    # Fit scaler
    scaler = TargetScaler()
    y_train_list = [d.y for d in train_dataset]
    y_train = torch.cat(y_train_list, dim=0).numpy()
    scaler.fit(y_train)
    scaler.save(Config.TARGET_SCALER_PATH)

    print(f"Target Scaler saved. Mean: {scaler.mean}, Std: {scaler.std}")

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader, scaler
