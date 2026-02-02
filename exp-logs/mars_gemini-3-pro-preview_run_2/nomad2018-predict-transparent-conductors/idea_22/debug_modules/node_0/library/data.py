import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from ase import Atoms
from ase.neighborlist import neighbor_list
from library.config import Config


def parse_custom_xyz(file_path):
    """
    Parses the custom XYZ format provided in the dataset.
    Extracts lattice vectors and atom positions.
    """
    lattice_vectors = []
    symbols = []
    positions = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        if parts[0] == "lattice_vector":
            # Format: lattice_vector x y z
            vec = [float(x) for x in parts[1:4]]
            lattice_vectors.append(vec)
        elif parts[0] == "atom":
            # Format: atom x y z symbol
            pos = [float(x) for x in parts[1:4]]
            sym = parts[4]
            positions.append(pos)
            symbols.append(sym)

    if len(lattice_vectors) != 3:
        # Fallback or error handling if lattice is missing,
        # though dataset description guarantees it.
        # Assuming cubic box if missing (unlikely based on description)
        lattice_vectors = np.eye(3)

    atoms = Atoms(symbols=symbols, positions=positions, cell=lattice_vectors, pbc=True)
    return atoms


def get_pbc_neighbors(atoms, cutoff=Config.CUTOFF_RADIUS):
    """
    Computes neighbor list respecting periodic boundary conditions.
    Returns edge_index and edge_distances.
    """
    # i: source indices, j: target indices, d: distances
    # We use 'ijD' to get indices and distances
    i, j, d = neighbor_list("ijD", atoms, cutoff)

    # Filter out self-loops if any (though usually neighbor_list handles cutoff > 0)
    mask = d > 0
    i = i[mask]
    j = j[mask]
    d = d[mask]

    edge_index = np.vstack((i, j))
    return edge_index, d


def process_dataset(metadata_path, cache_path, load_cached_data=True):
    """
    Processes the dataset: parses geometry, builds graphs, and caches/loads from disk.
    Uses a CSR-like format to store variable-length graph data in .npz without pickle.
    """
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path)

            # Reconstruct list of Data objects
            ids = data["ids"]
            targets = data["targets"]

            node_ptr = data["node_ptr"]
            all_nodes = data["all_nodes"]

            edge_ptr = data["edge_ptr"]
            all_edge_index = data["all_edge_index"]
            all_edge_attr = data["all_edge_attr"]

            data_list = []
            for k in range(len(ids)):
                # Extract nodes
                n_start, n_end = node_ptr[k], node_ptr[k + 1]
                x = torch.tensor(all_nodes[n_start:n_end], dtype=torch.long)

                # Extract edges
                e_start, e_end = edge_ptr[k], edge_ptr[k + 1]
                edge_index = torch.tensor(
                    all_edge_index[:, e_start:e_end], dtype=torch.long
                )
                edge_attr = torch.tensor(
                    all_edge_attr[e_start:e_end], dtype=torch.float
                )

                # Extract target
                y = torch.tensor(targets[k], dtype=torch.float).view(1, -1)

                # Create Data object
                # Note: pos is not strictly needed for the proposed graph network
                # as we use precomputed edge distances, but can be added if needed.
                # For this implementation, we focus on x, edge_index, edge_attr.
                data_obj = Data(
                    x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, id=int(ids[k])
                )
                data_list.append(data_obj)

            return data_list
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Compute from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Containers for flattened data
    all_ids = []
    all_targets = []

    flat_nodes = []
    node_ptr = [0]

    flat_edge_indices_src = []
    flat_edge_indices_dst = []
    flat_edge_attrs = []
    edge_ptr = [0]

    for idx, row in df.iterrows():
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Parse Atoms
        atoms = parse_custom_xyz(file_path)

        # Get Neighbors
        edge_index, distances = get_pbc_neighbors(atoms, Config.CUTOFF_RADIUS)

        # Node features (Atomic Numbers)
        atomic_numbers = atoms.get_atomic_numbers()

        # Targets (Handle test set where targets might be missing/NaN)
        if "formation_energy_ev_natom" in row and "bandgap_energy_ev" in row:
            y = [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
        else:
            y = [np.nan, np.nan]  # Placeholder for test

        # Append to lists
        all_ids.append(row["id"])
        all_targets.append(y)

        flat_nodes.extend(atomic_numbers)
        node_ptr.append(len(flat_nodes))

        if edge_index.shape[1] > 0:
            flat_edge_indices_src.extend(edge_index[0])
            flat_edge_indices_dst.extend(edge_index[1])
            flat_edge_attrs.extend(distances)
        edge_ptr.append(len(flat_edge_attrs))

    # Convert to numpy arrays
    all_ids = np.array(all_ids, dtype=np.int32)
    all_targets = np.array(all_targets, dtype=np.float32)
    all_nodes = np.array(flat_nodes, dtype=np.int32)
    node_ptr = np.array(node_ptr, dtype=np.int32)

    if len(flat_edge_indices_src) > 0:
        all_edge_index = np.vstack(
            (flat_edge_indices_src, flat_edge_indices_dst)
        ).astype(np.int32)
    else:
        all_edge_index = np.empty((2, 0), dtype=np.int32)

    all_edge_attr = np.array(flat_edge_attrs, dtype=np.float32)
    edge_ptr = np.array(edge_ptr, dtype=np.int32)

    # Save to cache
    print(f"Saving processed data to {cache_path}...")
    np.savez_compressed(
        cache_path,
        ids=all_ids,
        targets=all_targets,
        node_ptr=node_ptr,
        all_nodes=all_nodes,
        edge_ptr=edge_ptr,
        all_edge_index=all_edge_index,
        all_edge_attr=all_edge_attr,
    )

    # Reconstruct Data objects to return
    data_list = []
    for k in range(len(all_ids)):
        n_start, n_end = node_ptr[k], node_ptr[k + 1]
        x = torch.tensor(all_nodes[n_start:n_end], dtype=torch.long)

        e_start, e_end = edge_ptr[k], edge_ptr[k + 1]
        edge_index = torch.tensor(all_edge_index[:, e_start:e_end], dtype=torch.long)
        edge_attr = torch.tensor(all_edge_attr[e_start:e_end], dtype=torch.float)

        y = torch.tensor(all_targets[k], dtype=torch.float).view(1, -1)

        data_obj = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, id=int(all_ids[k])
        )
        data_list.append(data_obj)

    return data_list


class CrystalDataset(Dataset):
    def __init__(self, data_list, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def get_dataloaders(subset_size=None, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    # Define cache paths
    train_cache = os.path.join(Config.CACHE_DIR, "train_graphs.npz")
    val_cache = os.path.join(Config.CACHE_DIR, "val_graphs.npz")
    test_cache = os.path.join(Config.CACHE_DIR, "test_graphs.npz")

    # Process or load data
    train_list = process_dataset(
        Config.TRAIN_METADATA_PATH, train_cache, load_cached_data
    )
    val_list = process_dataset(Config.VAL_METADATA_PATH, val_cache, load_cached_data)
    test_list = process_dataset(Config.TEST_METADATA_PATH, test_cache, load_cached_data)

    # Subset for debugging if requested
    if subset_size is not None:
        train_list = train_list[:subset_size]
        val_list = val_list[:subset_size]
        # We usually don't subset test unless debugging inference flow specifically

    # Create Datasets
    train_dataset = CrystalDataset(train_list)
    val_dataset = CrystalDataset(val_list)
    test_dataset = CrystalDataset(test_list)

    # Create Loaders
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

    return train_loader, val_loader, test_loader
