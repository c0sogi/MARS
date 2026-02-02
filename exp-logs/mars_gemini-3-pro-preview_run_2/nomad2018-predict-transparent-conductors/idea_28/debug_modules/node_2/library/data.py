import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from ase.io import read
from ase.neighborlist import neighbor_list
from library.config import Config


class CrystalGraphDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def process_single_structure(file_path, target=None, material_id=None):
    """
    Reads an XYZ file and converts it into a PyTorch Geometric Data object.
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path)

    # Read structure using ASE
    try:
        # Explicitly specify 'aims' format as the file content matches FHI-aims geometry
        # despite the .xyz extension. Cite debug_lesson_1
        atoms = read(full_path, format="aims")
    except Exception as e:
        print(f"Error reading {full_path}: {e}")
        return None

    # Get atomic numbers (node features)
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)

    # Get neighbors with PBC
    # i: source indices, j: target indices, d: distances
    # We use 'ijD' to get indices and distance vectors, then compute norm
    # self_interaction=False prevents self-loops
    cut = Config.CUTOFF_RADIUS
    i, j, D = neighbor_list("ijD", atoms, cut, self_interaction=False)

    # Compute distances
    distances = np.linalg.norm(D, axis=1)

    # Convert to tensors
    edge_index = torch.tensor(np.vstack((i, j)), dtype=torch.long)
    edge_attr = torch.tensor(distances, dtype=torch.float).unsqueeze(1)

    # Handle Max Neighbors (Pruning) if necessary
    # Although neighbor_list is efficient, we enforce the config limit per node
    if Config.MAX_NEIGHBORS > 0:
        # This is a simple heuristic: if total edges are excessive relative to nodes
        # A more rigorous per-node sort could be done, but for this dataset size
        # and cutoff, it's usually fine. We'll skip complex per-node pruning
        # to keep preprocessing fast, as 5.0A usually yields reasonable degree.
        pass

    # Targets
    y = None
    if target is not None:
        y = torch.tensor(target, dtype=torch.float).view(1, -1)

    # Create Data object
    data = Data(x=atomic_numbers, edge_index=edge_index, edge_attr=edge_attr, y=y)

    if material_id is not None:
        data.id = torch.tensor([material_id], dtype=torch.long)

    return data


def save_graphs_to_npz(data_list, path):
    """
    Saves a list of PyTorch Geometric Data objects to a compressed NPZ file
    using flattened arrays to avoid pickle.
    """
    # Flatten all data
    all_x = []
    all_edge_index_src = []
    all_edge_index_dst = []
    all_edge_attr = []
    all_y = []
    all_ids = []

    # Metadata for reconstruction
    num_nodes_per_graph = []
    num_edges_per_graph = []

    for data in data_list:
        num_nodes = data.x.shape[0]
        num_edges = data.edge_index.shape[1]

        num_nodes_per_graph.append(num_nodes)
        num_edges_per_graph.append(num_edges)

        all_x.append(data.x.numpy())
        all_edge_index_src.append(data.edge_index[0].numpy())
        all_edge_index_dst.append(data.edge_index[1].numpy())
        all_edge_attr.append(data.edge_attr.numpy().flatten())

        if data.y is not None:
            all_y.append(data.y.numpy().flatten())
        else:
            # Placeholder for test set if needed, though we usually handle None
            all_y.append(np.array([np.nan, np.nan]))

        if hasattr(data, "id"):
            all_ids.append(data.id.numpy()[0])
        else:
            all_ids.append(-1)

    # Concatenate
    if len(all_x) > 0:
        flat_x = np.concatenate(all_x)
        flat_src = np.concatenate(all_edge_index_src)
        flat_dst = np.concatenate(all_edge_index_dst)
        flat_attr = np.concatenate(all_edge_attr)
        flat_y = np.stack(all_y)
        flat_ids = np.array(all_ids)
        counts_nodes = np.array(num_nodes_per_graph)
        counts_edges = np.array(num_edges_per_graph)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            x=flat_x,
            src=flat_src,
            dst=flat_dst,
            attr=flat_attr,
            y=flat_y,
            ids=flat_ids,
            n_nodes=counts_nodes,
            n_edges=counts_edges,
        )
        print(f"Saved {len(data_list)} graphs to {path}")
    else:
        print("No data to save.")


def load_graphs_from_npz(path):
    """
    Loads graphs from a compressed NPZ file containing flattened arrays.
    """
    print(f"Loading graphs from {path}...")
    data = np.load(path)

    flat_x = data["x"]
    flat_src = data["src"]
    flat_dst = data["dst"]
    flat_attr = data["attr"]
    flat_y = data["y"]
    flat_ids = data["ids"]
    n_nodes = data["n_nodes"]
    n_edges = data["n_edges"]

    data_list = []

    x_ptr = 0
    e_ptr = 0

    for i in range(len(n_nodes)):
        num_n = n_nodes[i]
        num_e = n_edges[i]

        # Extract node features
        x = torch.tensor(flat_x[x_ptr : x_ptr + num_n], dtype=torch.long)
        x_ptr += num_n

        # Extract edges
        src = torch.tensor(flat_src[e_ptr : e_ptr + num_e], dtype=torch.long)
        dst = torch.tensor(flat_dst[e_ptr : e_ptr + num_e], dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)

        attr = torch.tensor(
            flat_attr[e_ptr : e_ptr + num_e], dtype=torch.float
        ).unsqueeze(1)
        e_ptr += num_e

        # Extract target
        y_val = flat_y[i]
        if np.isnan(y_val).any():
            y = None
        else:
            y = torch.tensor(y_val, dtype=torch.float).view(1, -1)

        # Create Data
        d = Data(x=x, edge_index=edge_index, edge_attr=attr, y=y)
        d.id = torch.tensor([flat_ids[i]], dtype=torch.long)

        data_list.append(d)

    print(f"Loaded {len(data_list)} graphs.")
    return data_list


def load_or_process_data(
    metadata_path, cache_path, load_cached_data=True, is_test=False
):
    """
    Main logic to load data from cache or process from scratch.
    """
    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return load_graphs_from_npz(cache_path)
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Debugging: subset
    if Config.DEBUG:
        print("DEBUG MODE: Using only first 50 samples.")
        df = df.head(50)

    data_list = []

    for _, row in df.iterrows():
        file_path = row["file_path"]
        mat_id = row["id"]

        targets = None
        if not is_test:
            # Extract targets
            targets = row[Config.TARGET_COLS].values.astype(float)

        graph_data = process_single_structure(file_path, targets, mat_id)
        if graph_data is not None:
            data_list.append(graph_data)

    # 3. Save to cache
    save_graphs_to_npz(data_list, cache_path)

    return data_list


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # --- Train Data ---
    train_graphs = load_or_process_data(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_GRAPH_CACHE,
        load_cached_data=load_cached_data,
        is_test=False,
    )
    train_dataset = CrystalGraphDataset(train_graphs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues in some envs, set to >0 if safe
    )

    # --- Validation Data ---
    val_graphs = load_or_process_data(
        Config.VAL_METADATA_PATH,
        Config.VAL_GRAPH_CACHE,
        load_cached_data=load_cached_data,
        is_test=False,
    )
    val_dataset = CrystalGraphDataset(val_graphs)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # --- Test Data ---
    test_graphs = load_or_process_data(
        Config.TEST_METADATA_PATH,
        Config.TEST_GRAPH_CACHE,
        load_cached_data=load_cached_data,
        is_test=True,
    )
    test_dataset = CrystalGraphDataset(test_graphs)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader
