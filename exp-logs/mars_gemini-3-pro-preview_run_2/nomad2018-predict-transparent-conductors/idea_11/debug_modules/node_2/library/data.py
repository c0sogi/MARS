import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from ase.io import read
from ase.neighborlist import neighbor_list
from tqdm import tqdm

from library.config import Config
from library.utils import StandardScaler


class CrystalGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric Dataset for Crystal Graphs.
    Holds a list of Data objects in memory.
    """

    def __init__(self, data_list, root=None, transform=None, pre_transform=None):
        self.data_list = data_list
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self.collate(data_list)


def get_pbc_neighbors(atoms, cutoff):
    """
    Computes neighbors within a cutoff considering periodic boundary conditions.
    Returns edge_index (2, E) and edge_dist (E,).
    """
    # 'i' is the index of the central atom, 'j' is the index of the neighbor
    # 'd' is the distance vector
    # 'D' is the distance value (scalar) -> We use 'd' to get vector then norm, or just 'D' if available
    # ase.neighborlist.neighbor_list returns:
    # i, j, d, D = neighbor_list('ijdD', atoms, cutoff)
    # But we only need indices and scalar distance for features usually.
    # The prompt mentions using RBF of distances.

    i, j, d = neighbor_list("ijd", atoms, cutoff)

    # Compute distances
    distances = np.linalg.norm(d, axis=1)

    return i, j, distances


def process_one_structure(
    row, input_dir, global_feature_cols, target_cols, neighbor_cutoff
):
    """
    Process a single material into a graph Data object.
    """
    # 1. Load Structure
    file_path = os.path.join(input_dir, row["file_path"])
    try:
        # Cite debug_lesson_1: Explicitly Define Parsers When File Extensions Are Misleading
        atoms = read(file_path, format="aims")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # 2. Graph Construction (Nodes & Edges)
    # Nodes: Atomic numbers
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)

    # Edges: PBC Neighbor search
    src_idx, dst_idx, distances = get_pbc_neighbors(atoms, neighbor_cutoff)

    edge_index = torch.tensor(np.stack([src_idx, dst_idx], axis=0), dtype=torch.long)
    edge_attr = torch.tensor(distances, dtype=torch.float32)

    # 3. Global Features
    # Extract from pandas row
    global_feats = row[global_feature_cols].values.astype(np.float32)
    global_feats = torch.tensor(global_feats, dtype=torch.float32).unsqueeze(
        0
    )  # (1, n_global)

    # 4. Targets
    # Check if targets exist (they might not for test set)
    y = None
    if all(col in row for col in target_cols):
        targets = row[target_cols].values.astype(np.float32)
        y = torch.tensor(targets, dtype=torch.float32).unsqueeze(0)  # (1, n_targets)
    else:
        # Placeholder for test set
        y = torch.zeros((1, len(target_cols)), dtype=torch.float32)

    # Create Data object
    # We store material_id for reference
    data = Data(
        x=atomic_numbers,
        edge_index=edge_index,
        edge_attr=edge_attr,
        global_x=global_feats,
        y=y,
        material_id=row["id"],
    )

    return data


def save_graphs_to_npz(data_list, path):
    """
    Saves a list of Data objects to a compressed npz file without using pickle.
    We flatten the arrays and store indices to reconstruct them.
    """
    # Arrays to collect
    all_x = []
    all_edge_index = []
    all_edge_attr = []
    all_global_x = []
    all_y = []
    all_ids = []

    # Pointers
    node_ptr = [0]
    edge_ptr = [0]

    for data in data_list:
        all_x.append(data.x.numpy())
        all_edge_index.append(data.edge_index.numpy())
        all_edge_attr.append(data.edge_attr.numpy())
        all_global_x.append(data.global_x.numpy())
        all_y.append(data.y.numpy())
        all_ids.append(data.material_id)

        node_ptr.append(node_ptr[-1] + data.x.shape[0])
        edge_ptr.append(edge_ptr[-1] + data.edge_index.shape[1])

    # Concatenate
    if all_x:
        x_cat = np.concatenate(all_x)
        edge_index_cat = np.concatenate(all_edge_index, axis=1)
        edge_attr_cat = np.concatenate(all_edge_attr)
        global_x_cat = np.concatenate(all_global_x, axis=0)
        y_cat = np.concatenate(all_y, axis=0)
    else:
        # Empty dataset handling
        x_cat = np.array([])
        edge_index_cat = np.array([[], []])
        edge_attr_cat = np.array([])
        global_x_cat = np.array([])
        y_cat = np.array([])

    np.savez_compressed(
        path,
        x=x_cat,
        edge_index=edge_index_cat,
        edge_attr=edge_attr_cat,
        global_x=global_x_cat,
        y=y_cat,
        node_ptr=np.array(node_ptr, dtype=np.int64),
        edge_ptr=np.array(edge_ptr, dtype=np.int64),
        ids=np.array(all_ids, dtype=np.int64),
    )
    print(f"Saved {len(data_list)} graphs to {path}")


def load_graphs_from_npz(path):
    """
    Loads Data objects from an npz file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache file {path} does not exist.")

    print(f"Loading graphs from {path}...")
    data = np.load(path)

    x = torch.from_numpy(data["x"]).long()
    edge_index = torch.from_numpy(data["edge_index"]).long()
    edge_attr = torch.from_numpy(data["edge_attr"]).float()
    global_x = torch.from_numpy(data["global_x"]).float()
    y = torch.from_numpy(data["y"]).float()
    node_ptr = data["node_ptr"]
    edge_ptr = data["edge_ptr"]
    ids = data["ids"]

    data_list = []
    num_graphs = len(node_ptr) - 1

    for i in range(num_graphs):
        # Slice nodes
        n_start, n_end = node_ptr[i], node_ptr[i + 1]
        x_i = x[n_start:n_end]

        # Slice edges
        e_start, e_end = edge_ptr[i], edge_ptr[i + 1]
        edge_index_i = edge_index[:, e_start:e_end]
        edge_attr_i = edge_attr[e_start:e_end]

        # Slice globals and targets
        global_x_i = global_x[i : i + 1]
        y_i = y[i : i + 1]

        d = Data(
            x=x_i,
            edge_index=edge_index_i,
            edge_attr=edge_attr_i,
            global_x=global_x_i,
            y=y_i,
            material_id=int(ids[i]),
        )
        data_list.append(d)

    print(f"Loaded {len(data_list)} graphs.")
    return data_list


def get_dataset(split, load_cached_data=True, debug=False):
    """
    Main function to retrieve the dataset for a specific split.
    Handles loading, processing, caching, and scaling.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, limits the dataset size.

    Returns:
        CrystalGraphDataset: The PyTorch Geometric dataset.
    """
    # Determine metadata path
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
    elif split == "test":
        meta_path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    # Determine cache path
    cache_filename = f"{split}_graphs"
    if debug:
        cache_filename += "_debug"
    cache_path = os.path.join(Config.CACHE_DIR, f"{cache_filename}.npz")

    # 1. Try to load from cache
    data_list = []
    if load_cached_data and os.path.exists(cache_path):
        try:
            data_list = load_graphs_from_npz(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")
            data_list = []

    # 2. Process if empty
    if not data_list:
        print(f"Processing {split} data from scratch...")
        df = pd.read_csv(meta_path)

        if debug:
            df = df.head(Config.DEBUG_SAMPLE_SIZE)

        # Process in parallel or loop? Loop is safer for simple scripts, tqdm for progress
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
            d = process_one_structure(
                row,
                Config.INPUT_DIR,
                Config.GLOBAL_FEATURES,
                Config.TARGET_COLS,
                Config.NEIGHBOR_CUTOFF,
            )
            if d is not None:
                data_list.append(d)

        # Save to cache
        save_graphs_to_npz(data_list, cache_path)

    # 3. Scaling
    # We need to scale 'global_x' and 'y'.
    # Scalers are fitted ONLY on the training set.

    # Initialize scalers
    global_scaler = StandardScaler()
    target_scaler = StandardScaler()

    if split == "train":
        # Fit scalers
        all_global = np.concatenate([d.global_x.numpy() for d in data_list], axis=0)
        all_targets = np.concatenate([d.y.numpy() for d in data_list], axis=0)

        global_scaler.fit(all_global)
        target_scaler.fit(all_targets)

        # Save scalers
        torch.save(global_scaler.state_dict(), Config.GLOBAL_SCALER_PATH)
        torch.save(target_scaler.state_dict(), Config.TARGET_SCALER_PATH)
        print("Scalers fitted and saved.")

    else:
        # Load scalers
        if os.path.exists(Config.GLOBAL_SCALER_PATH) and os.path.exists(
            Config.TARGET_SCALER_PATH
        ):
            global_scaler.load_state_dict(torch.load(Config.GLOBAL_SCALER_PATH))
            target_scaler.load_state_dict(torch.load(Config.TARGET_SCALER_PATH))
            print("Scalers loaded.")
        else:
            print(
                "Warning: Scalers not found. Data will not be scaled (okay if just testing pipeline structure, bad for inference)."
            )
            # If we are in test mode but haven't trained, we might proceed unscaled or error out.
            # For robustness in this script, we'll proceed but metrics will be off.

    # Apply scaling
    # Note: We modify the data objects in place or create new ones.
    # Since we loaded into memory, we can modify.
    if global_scaler.mean is not None:
        for d in data_list:
            # Scale global features
            g = d.global_x.numpy()
            d.global_x = torch.tensor(global_scaler.transform(g), dtype=torch.float32)

            # Scale targets (only if they exist/are valid, i.e., not test set placeholder)
            # For test set, we don't scale y because it's dummy zeros.
            if split != "test":
                t = d.y.numpy()
                d.y = torch.tensor(target_scaler.transform(t), dtype=torch.float32)

    return CrystalGraphDataset(data_list)
