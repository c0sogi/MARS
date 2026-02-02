import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from ase.io import read
from ase.neighborlist import neighbor_list
from tqdm import tqdm

from library.config import Config
from library.utils import set_seed


class TargetScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, targets):
        self.mean = np.mean(targets, axis=0)
        self.std = np.std(targets, axis=0)
        # Avoid division by zero
        self.std[self.std == 0] = 1.0

    def transform(self, targets):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler not fitted")
        return (targets - self.mean) / self.std

    def inverse_transform(self, targets_scaled):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler not fitted")
        return (targets_scaled * self.std) + self.mean

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path):
        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]


def structure_to_graph(file_path, cutoff=5.0):
    """
    Converts a crystal structure file to a graph representation.
    """
    try:
        # Read structure using ASE
        atoms = read(file_path)

        # Get atomic numbers (node features)
        atomic_numbers = atoms.get_atomic_numbers()

        # Compute neighbor list with PBC
        # i: source, j: target, d: distance
        i, j, d = neighbor_list("ijd", atoms, cutoff)

        # Create edge index (2, E)
        edge_index = np.vstack((i, j)).astype(np.int64)

        # Edge features (distances)
        edge_distances = d.astype(np.float32)

        return {
            "atom_numbers": atomic_numbers.astype(np.int64),
            "edge_index": edge_index,
            "edge_distances": edge_distances,
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Return empty graph as fallback
        return {
            "atom_numbers": np.array([], dtype=np.int64),
            "edge_index": np.empty((2, 0), dtype=np.int64),
            "edge_distances": np.array([], dtype=np.float32),
        }


def process_and_cache_graphs(
    metadata_df, cache_path, load_cached_data=True, geometry_dir=None
):
    """
    Processes structures into graphs and caches them using numpy .npz format.
    Efficiently stores variable-sized graphs using concatenated arrays and pointers.
    """

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached graphs from {cache_path}...")
        try:
            data = np.load(cache_path)
            all_atom_numbers = data["atom_numbers"]
            all_edge_indices = data["edge_indices"]
            all_edge_distances = data["edge_distances"]
            atom_ptr = data["atom_ptr"]
            edge_ptr = data["edge_ptr"]

            graphs = []
            for k in range(len(atom_ptr) - 1):
                a_start, a_end = atom_ptr[k], atom_ptr[k + 1]
                e_start, e_end = edge_ptr[k], edge_ptr[k + 1]

                graphs.append(
                    {
                        "atom_numbers": all_atom_numbers[a_start:a_end],
                        "edge_index": all_edge_indices[:, e_start:e_end],
                        "edge_distances": all_edge_distances[e_start:e_end],
                    }
                )
            return graphs
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Processing {len(metadata_df)} structures...")
    graphs = []

    # Lists to accumulate data for vectorization
    all_atom_numbers = []
    all_edge_indices = []
    all_edge_distances = []
    atom_ptr = [0]
    edge_ptr = [0]

    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
        # Construct full file path
        # The metadata file_path is relative to input dir, e.g. "train/1/geometry.xyz"
        # Config.INPUT_DIR is "./input"
        # So we join them.
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        graph = structure_to_graph(full_path, cutoff=Config.CUTOFF_RADIUS)
        graphs.append(graph)

        all_atom_numbers.append(graph["atom_numbers"])
        all_edge_indices.append(graph["edge_index"])
        all_edge_distances.append(graph["edge_distances"])

        atom_ptr.append(atom_ptr[-1] + len(graph["atom_numbers"]))
        edge_ptr.append(edge_ptr[-1] + graph["edge_index"].shape[1])

    # Concatenate
    if graphs:
        np_atom_numbers = np.concatenate(all_atom_numbers)
        np_edge_indices = np.concatenate(all_edge_indices, axis=1)
        np_edge_distances = np.concatenate(all_edge_distances)
        np_atom_ptr = np.array(atom_ptr, dtype=np.int64)
        np_edge_ptr = np.array(edge_ptr, dtype=np.int64)

        # Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            cache_path,
            atom_numbers=np_atom_numbers,
            edge_indices=np_edge_indices,
            edge_distances=np_edge_distances,
            atom_ptr=np_atom_ptr,
            edge_ptr=np_edge_ptr,
        )
        print(f"Saved processed graphs to {cache_path}")

    return graphs


class CrystalDataset(Dataset):
    def __init__(self, graphs, metadata, scaler=None, is_test=False):
        """
        Args:
            graphs: List of dictionaries containing graph data.
            metadata: Pandas DataFrame containing metadata and targets.
            scaler: TargetScaler instance for normalizing targets (optional).
            is_test: Boolean, if True, does not look for targets.
        """
        self.graphs = graphs
        self.metadata = metadata
        self.scaler = scaler
        self.is_test = is_test

        # Pre-convert targets to tensor if not test
        if not self.is_test:
            targets = self.metadata[
                ["formation_energy_ev_natom", "bandgap_energy_ev"]
            ].values.astype(np.float32)
            if self.scaler:
                targets = self.scaler.transform(targets)
            self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph_data = self.graphs[idx]

        # Create PyTorch Geometric Data object
        # x: Node features (atomic numbers). Shape: [num_nodes]
        # edge_index: Graph connectivity. Shape: [2, num_edges]
        # edge_attr: Edge features (distances). Shape: [num_edges, 1]

        x = torch.tensor(graph_data["atom_numbers"], dtype=torch.long)
        edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)
        edge_attr = torch.tensor(
            graph_data["edge_distances"], dtype=torch.float32
        ).unsqueeze(1)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Add target if available
        if not self.is_test:
            data.y = self.y[idx].unsqueeze(0)  # [1, 2]

        # Add identifier for tracking
        data.id = self.metadata.iloc[idx]["id"]

        return data


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    data_sample_size=Config.DATA_SAMPLE_SIZE,
    load_cached_data=True,
):
    """
    Prepares DataLoaders for train, validation, and test sets.
    Handles caching, scaling, and subsetting.
    """
    set_seed(Config.SEED)

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Subset for debugging if requested
    if data_sample_size is not None:
        print(f"Subsetting data to {data_sample_size} samples for debugging.")
        train_df = train_df.iloc[:data_sample_size]
        val_df = val_df.iloc[: min(len(val_df), data_sample_size)]
        # We generally keep test set full or subset it too? Let's subset to be safe for quick runs
        test_df = test_df.iloc[: min(len(test_df), data_sample_size)]

    # 2. Process/Load Graphs
    print("Preparing Train Graphs...")
    train_graphs = process_and_cache_graphs(
        train_df, Config.TRAIN_GRAPH_CACHE, load_cached_data, Config.TRAIN_GEOMETRY_DIR
    )

    print("Preparing Validation Graphs...")
    val_graphs = process_and_cache_graphs(
        val_df, Config.VAL_GRAPH_CACHE, load_cached_data, Config.TRAIN_GEOMETRY_DIR
    )

    print("Preparing Test Graphs...")
    test_graphs = process_and_cache_graphs(
        test_df, Config.TEST_GRAPH_CACHE, load_cached_data, Config.TEST_GEOMETRY_DIR
    )

    # 3. Fit Scaler
    scaler = TargetScaler()
    # Extract raw targets from train_df
    train_targets = train_df[
        ["formation_energy_ev_natom", "bandgap_energy_ev"]
    ].values.astype(np.float32)
    scaler.fit(train_targets)

    # Save scaler for inference
    scaler.save(Config.TARGET_SCALER_CACHE)
    print(f"Target Scaler fitted. Mean: {scaler.mean}, Std: {scaler.std}")

    # 4. Create Datasets
    train_dataset = CrystalDataset(train_graphs, train_df, scaler=scaler, is_test=False)
    val_dataset = CrystalDataset(val_graphs, val_df, scaler=scaler, is_test=False)
    test_dataset = CrystalDataset(test_graphs, test_df, scaler=None, is_test=True)

    # 5. Create DataLoaders
    # PyG DataLoader handles batching of graphs automatically
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

    return train_loader, val_loader, test_loader, scaler
