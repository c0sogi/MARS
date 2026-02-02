import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from ase.io import read
from ase.neighborlist import neighbor_list
from library.config import Config
from library.utils import get_scaler


class CrystalGraphDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def save_graphs(path, data_list):
    """
    Saves a list of PyG Data objects to an .npz file without using pickle.
    """
    # Collect all data into arrays
    x_list = []
    edge_index_list = []
    edge_attr_list = []
    y_list = []

    n_nodes_list = []
    n_edges_list = []

    # We also need to store ids to map back if needed, though order is preserved
    ids = []

    for data in data_list:
        x_list.append(data.x.numpy())
        edge_index_list.append(data.edge_index.numpy())
        edge_attr_list.append(data.edge_attr.numpy())

        # Handle y (targets)
        if data.y is not None:
            y_list.append(data.y.numpy())
        else:
            # Placeholder for test set if y is missing
            y_list.append(np.array([[np.nan, np.nan]], dtype=np.float32))

        n_nodes_list.append(data.num_nodes)
        n_edges_list.append(data.num_edges)

        # Assuming id is stored in data object, if not we rely on list order
        if hasattr(data, "material_id"):
            ids.append(data.material_id)
        else:
            ids.append(-1)

    # Concatenate
    if x_list:
        x_all = np.concatenate(x_list, axis=0)
    else:
        x_all = np.array([])

    if edge_index_list:
        edge_index_all = np.concatenate(edge_index_list, axis=1)
    else:
        edge_index_all = np.array([[], []])

    if edge_attr_list:
        edge_attr_all = np.concatenate(edge_attr_list, axis=0)
    else:
        edge_attr_all = np.array([])

    if y_list:
        # y_list contains (1, 2) arrays, stack them to (N, 2)
        y_all = np.concatenate(y_list, axis=0)
    else:
        y_all = np.array([])

    n_nodes_arr = np.array(n_nodes_list, dtype=np.int32)
    n_edges_arr = np.array(n_edges_list, dtype=np.int32)
    ids_arr = np.array(ids, dtype=np.int32)

    np.savez(
        path,
        x=x_all,
        edge_index=edge_index_all,
        edge_attr=edge_attr_all,
        y=y_all,
        n_nodes=n_nodes_arr,
        n_edges=n_edges_arr,
        ids=ids_arr,
    )


def load_graphs(path):
    """
    Loads a list of PyG Data objects from an .npz file.
    """
    if not os.path.exists(path):
        return None

    try:
        data = np.load(path)
        x_all = data["x"]
        edge_index_all = data["edge_index"]
        edge_attr_all = data["edge_attr"]
        y_all = data["y"]
        n_nodes_arr = data["n_nodes"]
        n_edges_arr = data["n_edges"]
        ids_arr = data["ids"]

        data_list = []

        node_ptr = 0
        edge_ptr = 0

        for i in range(len(n_nodes_arr)):
            n_n = n_nodes_arr[i]
            n_e = n_edges_arr[i]

            # Slice x
            x = torch.from_numpy(x_all[node_ptr : node_ptr + n_n]).long()

            # Slice edge_index
            edge_index = torch.from_numpy(
                edge_index_all[:, edge_ptr : edge_ptr + n_e]
            ).long()

            # Slice edge_attr
            edge_attr = torch.from_numpy(
                edge_attr_all[edge_ptr : edge_ptr + n_e]
            ).float()

            # Slice y
            y_val = y_all[i]
            if np.isnan(y_val).any():
                y = None
            else:
                y = torch.from_numpy(y_val).float().view(1, -1)

            mat_id = int(ids_arr[i])

            d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            d.material_id = mat_id

            data_list.append(d)

            node_ptr += n_n
            edge_ptr += n_e

        return data_list
    except Exception as e:
        print(f"Error loading cache {path}: {e}")
        return None


def process_geometry(file_path, radius, max_neighbors):
    """
    Parses an xyz file and returns graph data components.
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path)
    atoms = read(full_path, format="aims")

    # Get neighbors
    # i: center atom indices, j: neighbor atom indices, D: distances
    i, j, D = neighbor_list("ijd", atoms, radius)

    num_atoms = len(atoms)

    # Prepare lists for filtered edges
    final_i = []
    final_j = []
    final_D = []

    for atom_idx in range(num_atoms):
        # Find neighbors for this atom
        mask = i == atom_idx

        if not np.any(mask):
            continue

        nbr_indices = j[mask]
        nbr_dists = D[mask]

        if len(nbr_indices) > max_neighbors:
            # Sort by distance
            sorted_indices = np.argsort(nbr_dists)
            selected_indices = sorted_indices[:max_neighbors]

            final_i.extend([atom_idx] * max_neighbors)
            final_j.extend(nbr_indices[selected_indices])
            final_D.extend(nbr_dists[selected_indices])
        else:
            final_i.extend([atom_idx] * len(nbr_indices))
            final_j.extend(nbr_indices)
            final_D.extend(nbr_dists)

    edge_index = torch.tensor([final_i, final_j], dtype=torch.long)
    edge_attr = torch.tensor(final_D, dtype=torch.float)
    x = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)

    return x, edge_index, edge_attr


def get_graph_data(metadata_df, radius, max_neighbors, has_targets=True):
    data_list = []
    for _, row in metadata_df.iterrows():
        x, edge_index, edge_attr = process_geometry(
            row["file_path"], radius, max_neighbors
        )

        y = None
        if has_targets:
            # Extract targets
            targets = row[Config.TARGET_COLS].values.astype(np.float32)
            y = torch.tensor(targets, dtype=torch.float).view(1, -1)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.material_id = row["id"]
        data_list.append(data)

    return data_list


def get_dataloaders(load_cached_data=True):
    """
    Main function to prepare DataLoaders.
    """

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    if Config.DEBUG:
        train_df = train_df.iloc[: Config.MAX_SAMPLES]
        val_df = val_df.iloc[: Config.MAX_SAMPLES]
        test_df = test_df.iloc[: Config.MAX_SAMPLES]
        print(
            f"DEBUG Mode: Reduced dataset sizes to {len(train_df)}, {len(val_df)}, {len(test_df)}"
        )

    # 2. Process/Load Train Data
    train_list = None
    if load_cached_data:
        train_list = load_graphs(Config.TRAIN_GRAPHS_CACHE)

    if train_list is None:
        print("Processing training graphs...")
        train_list = get_graph_data(
            train_df, Config.RADIUS, Config.MAX_NUM_NBR, has_targets=True
        )
        save_graphs(Config.TRAIN_GRAPHS_CACHE, train_list)
    else:
        print("Loaded training graphs from cache.")

    # 3. Process/Load Val Data
    val_list = None
    if load_cached_data:
        val_list = load_graphs(Config.VAL_GRAPHS_CACHE)

    if val_list is None:
        print("Processing validation graphs...")
        val_list = get_graph_data(
            val_df, Config.RADIUS, Config.MAX_NUM_NBR, has_targets=True
        )
        save_graphs(Config.VAL_GRAPHS_CACHE, val_list)
    else:
        print("Loaded validation graphs from cache.")

    # 4. Process/Load Test Data
    test_list = None
    if load_cached_data:
        test_list = load_graphs(Config.TEST_GRAPHS_CACHE)

    if test_list is None:
        print("Processing test graphs...")
        test_list = get_graph_data(
            test_df, Config.RADIUS, Config.MAX_NUM_NBR, has_targets=False
        )
        save_graphs(Config.TEST_GRAPHS_CACHE, test_list)
    else:
        print("Loaded test graphs from cache.")

    # 5. Scaling
    # Extract training targets for scaler fitting
    # We need to stack them. train_list contains Data objects with y.
    train_targets = torch.cat([d.y for d in train_list], dim=0)

    # Get scaler (fits on train_targets)
    scaler = get_scaler(
        train_targets, Config.TARGET_SCALER_CACHE, load_cached_data=load_cached_data
    )

    # Apply scaling to train and val datasets in-place (or create new list)
    # Since Data objects are mutable, we can modify them.
    # Note: We don't scale test data as it has no targets.

    for data in train_list:
        data.y = scaler.transform(data.y)

    for data in val_list:
        data.y = scaler.transform(data.y)

    # 6. Create Datasets
    train_dataset = CrystalGraphDataset(train_list)
    val_dataset = CrystalGraphDataset(val_list)
    test_dataset = CrystalGraphDataset(test_list)

    # 7. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, scaler
