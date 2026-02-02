import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from ase.io import read
from library.config import (
    INPUT_DIR,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    CACHE_DIR,
    COMPOSITION_COLS,
    TARGET_COLS,
    BATCH_SIZE,
    NUM_WORKERS,
    DEBUG_SAMPLE_SIZE,
    SEED,
)
from library.utils import (
    build_pbc_graph,
    CompositionScaler,
    LogStandardScaler,
)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class CrystalDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def save_graphs_to_npz(data_list, path):
    """
    Saves a list of PyG Data objects to an NPZ file without using pickle.
    """
    # Arrays to collect
    all_x = []
    all_edge_index = []
    all_edge_attr = []
    all_y = []
    all_composition = []
    all_ids = []

    nodes_per_graph = []
    edges_per_graph = []

    for data in data_list:
        all_x.append(data.x.numpy())
        all_edge_index.append(data.edge_index.numpy())
        all_edge_attr.append(data.edge_attr.numpy())

        # Handle targets (might be None for test set, but we usually initialize with NaNs or 0s)
        if data.y is not None:
            all_y.append(data.y.numpy())
        else:
            all_y.append(np.zeros((1, 2), dtype=np.float32))

        all_composition.append(data.composition.numpy())
        all_ids.append(data.id)

        nodes_per_graph.append(data.num_nodes)
        edges_per_graph.append(data.num_edges)

    # Concatenate
    # Note: edge_index is (2, E), need to transpose or handle carefully.
    # np.concatenate on axis 1 for edge_index

    np.savez(
        path,
        x=np.concatenate(all_x, axis=0),
        edge_index=np.concatenate(all_edge_index, axis=1),
        edge_attr=np.concatenate(all_edge_attr, axis=0),
        y=np.concatenate(all_y, axis=0),
        composition=np.concatenate(all_composition, axis=0),
        ids=np.array(all_ids, dtype=np.int32),
        nodes_per_graph=np.array(nodes_per_graph, dtype=np.int32),
        edges_per_graph=np.array(edges_per_graph, dtype=np.int32),
    )


def load_graphs_from_npz(path):
    """
    Loads a list of PyG Data objects from an NPZ file.
    """
    data = np.load(path)

    x_all = data["x"]
    edge_index_all = data["edge_index"]
    edge_attr_all = data["edge_attr"]
    y_all = data["y"]
    comp_all = data["composition"]
    ids_all = data["ids"]
    nodes_per_graph = data["nodes_per_graph"]
    edges_per_graph = data["edges_per_graph"]

    data_list = []

    node_offset = 0
    edge_offset = 0

    for i in range(len(nodes_per_graph)):
        n_nodes = nodes_per_graph[i]
        n_edges = edges_per_graph[i]

        x = torch.tensor(x_all[node_offset : node_offset + n_nodes], dtype=torch.long)

        # edge_index is (2, Total_Edges)
        edge_index = torch.tensor(
            edge_index_all[:, edge_offset : edge_offset + n_edges], dtype=torch.long
        )

        edge_attr = torch.tensor(
            edge_attr_all[edge_offset : edge_offset + n_edges], dtype=torch.float32
        )

        y = torch.tensor(y_all[i : i + 1], dtype=torch.float32)
        composition = torch.tensor(comp_all[i : i + 1], dtype=torch.float32)
        id_val = int(ids_all[i])

        d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        d.composition = composition
        d.id = id_val

        data_list.append(d)

        node_offset += n_nodes
        edge_offset += n_edges

    return data_list


def process_data(metadata_path, cache_path, load_cached_data=True, is_test=False):
    """
    Reads metadata, loads XYZ files, builds graphs, and caches/loads them.
    """
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            return load_graphs_from_npz(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    if DEBUG_SAMPLE_SIZE is not None:
        df = df.head(DEBUG_SAMPLE_SIZE)
        print(f"Debug mode: sampled {len(df)} records.")

    data_list = []

    for idx, row in df.iterrows():
        # Path to geometry file
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Load atoms
        atoms = read(full_path)

        # Build graph
        graph_data = build_pbc_graph(atoms)

        # Composition features
        # Assuming cols are: percent_atom_al, percent_atom_ga, percent_atom_in
        # We need to ensure they are in the correct order as defined in config
        comp_vals = row[COMPOSITION_COLS].values.astype(np.float32)
        graph_data.composition = torch.tensor(comp_vals, dtype=torch.float32).unsqueeze(
            0
        )  # (1, 3)

        # Targets
        if not is_test:
            y_vals = row[TARGET_COLS].values.astype(np.float32)
            graph_data.y = torch.tensor(y_vals, dtype=torch.float32).unsqueeze(
                0
            )  # (1, 2)
        else:
            # Dummy target for test set to maintain consistency
            graph_data.y = torch.zeros((1, 2), dtype=torch.float32)

        # Store ID for submission mapping
        graph_data.id = row["id"]

        data_list.append(graph_data)

    print(f"Saving processed data to {cache_path}...")
    save_graphs_to_npz(data_list, cache_path)

    return data_list


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders.
    Handles splitting, scaling, and caching.
    """
    set_seed(SEED)

    # Define cache paths
    train_cache = os.path.join(CACHE_DIR, "train_graphs.npz")
    val_cache = os.path.join(CACHE_DIR, "val_graphs.npz")
    test_cache = os.path.join(CACHE_DIR, "test_graphs.npz")

    # Process or load datasets
    train_list = process_data(TRAIN_CSV, train_cache, load_cached_data, is_test=False)
    val_list = process_data(VAL_CSV, val_cache, load_cached_data, is_test=False)
    test_list = process_data(TEST_CSV, test_cache, load_cached_data, is_test=True)

    # Fit scalers on training data
    print("Fitting scalers on training data...")

    # Collect all composition vectors and targets from train list
    # Note: data.composition is (1, 3), data.y is (1, 2)
    all_comp = torch.cat([d.composition for d in train_list], dim=0)
    all_y = torch.cat([d.y for d in train_list], dim=0)

    comp_scaler = CompositionScaler()
    comp_scaler.fit(all_comp)

    target_scaler = LogStandardScaler()
    target_scaler.fit(all_y)

    # Save scalers
    comp_scaler.save(os.path.join(CACHE_DIR, "global_scaler.npz"))
    target_scaler.save(os.path.join(CACHE_DIR, "target_scaler.npz"))

    # Create Datasets
    train_dataset = CrystalDataset(train_list)
    val_dataset = CrystalDataset(val_list)
    test_dataset = CrystalDataset(test_list)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, comp_scaler, target_scaler
