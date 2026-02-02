import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from ase.io import read
from ase.neighborlist import neighbor_list

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_1"
# Mapping based on dataset description: Al, Ga, In, O
# Atomic numbers: O(8), Al(13), Ga(31), In(49)
ATOM_MAP = {8: 0, 13: 1, 31: 2, 49: 3}


def get_atom_features(atomic_numbers):
    """
    Maps atomic numbers to 0-based indices for embedding lookup.
    """
    return np.array([ATOM_MAP.get(z, -1) for z in atomic_numbers], dtype=np.int64)


def process_structures(metadata_path, radius=5.0):
    """
    Reads structures listed in the metadata CSV, computes neighbor lists,
    and returns a dictionary of concatenated arrays suitable for saving to disk.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        radius (float): Cutoff radius for neighbor finding.

    Returns:
        dict: Dictionary containing concatenated atom features, edge indices,
              edge distances, targets, IDs, and split indices.
    """
    df = pd.read_csv(metadata_path)

    all_atom_feats = []
    all_edge_src = []
    all_edge_dst = []
    all_edge_dists = []
    all_targets = []
    all_ids = []

    node_counts = []
    edge_counts = []

    for _, row in df.iterrows():
        file_path = os.path.join(INPUT_DIR, row["file_path"])

        # Parse geometry file
        # The file extension is .xyz but the content is FHI-aims format
        # We must specify format='aims' to parse it correctly.
        try:
            atoms = read(file_path, format="aims")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Get atom features (node attributes)
        z = atoms.get_atomic_numbers()
        atom_feats = get_atom_features(z)

        # Get neighbors (edge attributes and connectivity)
        # i: source index, j: target index, d: distance
        # We use 'ijd' to get indices and scalar distances
        i, j, d = neighbor_list("ijd", atoms, cutoff=radius)

        # Filter out self-loops (distance > 0)
        # Although neighbor_list typically respects cutoff, self-interactions
        # might appear with d=0 if not careful, though usually not with default settings.
        # Explicitly filtering d > 0 is safer for graph convs.
        mask = d > 0
        i = i[mask]
        j = j[mask]
        d = d[mask]

        # Collect data for this crystal
        all_atom_feats.append(atom_feats)
        all_edge_src.append(i)
        all_edge_dst.append(j)
        all_edge_dists.append(d)
        all_ids.append(row["id"])

        node_counts.append(len(atom_feats))
        edge_counts.append(len(i))

        # Handle targets
        # Test set might not have targets, fill with NaN
        if "formation_energy_ev_natom" in row and pd.notna(
            row["formation_energy_ev_natom"]
        ):
            all_targets.append(
                [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
            )
        else:
            all_targets.append([np.nan, np.nan])

    # Concatenate all lists into single numpy arrays for efficient storage
    # If dataset is empty (e.g. bad paths), handle gracefully
    if not all_atom_feats:
        raise ValueError(f"No valid data processed from {metadata_path}")

    return {
        "atom_feats": np.concatenate(all_atom_feats).astype(np.int64),
        "edge_src": np.concatenate(all_edge_src).astype(np.int64),
        "edge_dst": np.concatenate(all_edge_dst).astype(np.int64),
        "edge_dists": np.concatenate(all_edge_dists).astype(np.float32),
        "targets": np.array(all_targets, dtype=np.float32),
        "ids": np.array(all_ids, dtype=np.int64),
        # Store cumulative counts to reconstruct individual graphs
        "node_splits": np.cumsum([0] + node_counts, dtype=np.int64),
        "edge_splits": np.cumsum([0] + edge_counts, dtype=np.int64),
    }


class CrystalDataset(Dataset):
    """
    PyTorch Dataset for Crystal Graphs.
    """

    def __init__(
        self, metadata_file, radius=5.0, load_cached_data=True, split_name="train"
    ):
        """
        Args:
            metadata_file (str): Filename of the metadata CSV in ./metadata.
            radius (float): Cutoff radius for graph construction.
            load_cached_data (bool): Whether to try loading from cache.
            split_name (str): 'train', 'val', or 'test' for cache naming.
        """
        self.metadata_path = os.path.join(METADATA_DIR, metadata_file)
        self.radius = radius
        self.split_name = split_name

        # Cache file path
        self.cache_path = os.path.join(CACHE_DIR, f"{split_name}_graphs_r{radius}.npz")
        os.makedirs(CACHE_DIR, exist_ok=True)

        data = None

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                print(f"Loading cached {split_name} data from {self.cache_path}...")
                # allow_pickle=False is default and safer, we only stored numeric arrays
                loaded = np.load(self.cache_path)
                # Convert NpzFile to dict to keep in memory
                data = {k: loaded[k] for k in loaded.files}
            except Exception as e:
                print(f"Failed to load cache for {split_name}: {e}")
                data = None

        # 2. Process if not loaded
        if data is None:
            print(f"Processing {split_name} data from scratch...")
            data = process_structures(self.metadata_path, radius=radius)
            print(f"Saving {split_name} data to cache...")
            np.savez(self.cache_path, **data)

        # 3. Convert to PyTorch tensors
        self.atom_feats = torch.from_numpy(data["atom_feats"])
        self.edge_src = torch.from_numpy(data["edge_src"])
        self.edge_dst = torch.from_numpy(data["edge_dst"])
        self.edge_dists = torch.from_numpy(data["edge_dists"])
        self.targets = torch.from_numpy(data["targets"])
        self.ids = torch.from_numpy(data["ids"])
        self.node_splits = torch.from_numpy(data["node_splits"])
        self.edge_splits = torch.from_numpy(data["edge_splits"])

        self.length = len(self.ids)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns a single graph object.
        """
        # Identify slice indices
        n_start = self.node_splits[idx]
        n_end = self.node_splits[idx + 1]
        e_start = self.edge_splits[idx]
        e_end = self.edge_splits[idx + 1]

        # Slice data
        atom_fea = self.atom_feats[n_start:n_end]

        # Edge indices are already local to the graph (0 to N-1)
        src = self.edge_src[e_start:e_end]
        dst = self.edge_dst[e_start:e_end]
        edge_index = torch.stack([src, dst], dim=0)  # Shape [2, n_edges]

        edge_dist = self.edge_dists[e_start:e_end]

        target = self.targets[idx]
        crystal_id = self.ids[idx]

        return atom_fea, edge_index, edge_dist, target, crystal_id


def collate_batch(batch):
    """
    Collates a list of (atom_fea, edge_index, edge_dist, target, id) tuples
    into a single batched graph.

    Args:
        batch: List of tuples from __getitem__

    Returns:
        batch_atom_fea: [Total_Nodes]
        batch_edge_index: [2, Total_Edges]
        batch_edge_dist: [Total_Edges]
        batch_index: [Total_Nodes] (maps nodes to graph index in batch)
        batch_targets: [Batch_Size, 2]
        batch_ids: [Batch_Size]
    """
    atom_feas, edge_indices, edge_dists, targets, ids = zip(*batch)

    # 1. Concatenate Node Features
    batch_atom_fea = torch.cat(atom_feas, dim=0)

    # 2. Concatenate Targets and IDs
    batch_targets = torch.stack(targets, dim=0)
    batch_ids = torch.stack(ids, dim=0)

    # 3. Create Batch Index (for global pooling)
    # e.g. [0, 0, 0, 1, 1, 2, 2, 2, 2...]
    batch_index_list = []
    for i, fea in enumerate(atom_feas):
        n_nodes = fea.shape[0]
        batch_index_list.append(torch.full((n_nodes,), i, dtype=torch.long))
    batch_index = torch.cat(batch_index_list, dim=0)

    # 4. Concatenate Edge Indices with Offset
    # We need to shift the indices of subsequent graphs so they point to the correct nodes in the big batch list
    cumulative_nodes = 0
    shifted_edge_indices = []
    for i, edge_index in enumerate(edge_indices):
        shifted_edge_indices.append(edge_index + cumulative_nodes)
        cumulative_nodes += atom_feas[i].shape[0]

    if shifted_edge_indices:
        batch_edge_index = torch.cat(shifted_edge_indices, dim=1)
        batch_edge_dist = torch.cat(edge_dists, dim=0)
    else:
        # Handle case with no edges in entire batch (unlikely but possible)
        batch_edge_index = torch.empty((2, 0), dtype=torch.long)
        batch_edge_dist = torch.empty((0,), dtype=torch.float32)

    return (
        batch_atom_fea,
        batch_edge_index,
        batch_edge_dist,
        batch_index,
        batch_targets,
        batch_ids,
    )


def get_train_val_test_loaders(
    batch_size=64, radius=5.0, num_workers=0, load_cached_data=True
):
    """
    Helper function to create DataLoaders for all splits.

    Args:
        batch_size (int): Batch size.
        radius (float): Cutoff radius.
        num_workers (int): Number of worker processes for loading.
        load_cached_data (bool): Whether to use cached .npz files.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = CrystalDataset(
        "train_metadata.csv", radius, load_cached_data, split_name="train"
    )
    val_dataset = CrystalDataset(
        "val_metadata.csv", radius, load_cached_data, split_name="val"
    )
    test_dataset = CrystalDataset(
        "test_metadata.csv", radius, load_cached_data, split_name="test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
