import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from library.config import (
    STRUCTURES_CSV,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    WORKING_DIR,
    ATOM_MAP,
    TYPE_MAP,
    RADIUS_CUTOFF,
    BATCH_SIZE,
    NUM_WORKERS,
    DEBUG,
    DEBUG_SAMPLE_SIZE,
    SEED,
)


class StructureProcessor:
    """
    Handles processing of molecular structures into graph components.
    Computes distance matrices, creates radius graphs, and caches the result
    as flattened numpy arrays for efficient random access.
    """

    def __init__(self, structures_path=STRUCTURES_CSV, cache_dir=WORKING_DIR):
        self.structures_path = structures_path
        self.cache_dir = cache_dir

        # Cache file paths
        self.node_feats_path = os.path.join(cache_dir, "node_features.npy")
        self.coords_path = os.path.join(cache_dir, "atom_coords.npy")
        self.edge_indices_path = os.path.join(cache_dir, "edge_indices.npy")
        self.edge_attrs_path = os.path.join(cache_dir, "edge_attrs.npy")
        self.index_map_path = os.path.join(cache_dir, "mol_index_map.parquet")

        # In-memory storage
        self.node_feats = None
        self.coords = None
        self.edge_indices = None
        self.edge_attrs = None
        self.mol_map = None

    def process(self, load_cached_data=True):
        """
        Loads data from cache if available and requested.
        Otherwise, processes structures.csv and creates the cache.
        """
        if load_cached_data and self._check_cache_exists():
            print("Loading processed structure data from cache...")
            self._load_cache()
        else:
            print("Processing structures and generating graph data...")
            self._compute_and_cache()

        return self

    def _check_cache_exists(self):
        return (
            os.path.exists(self.node_feats_path)
            and os.path.exists(self.coords_path)
            and os.path.exists(self.edge_indices_path)
            and os.path.exists(self.edge_attrs_path)
            and os.path.exists(self.index_map_path)
        )

    def _load_cache(self):
        # Load using mmap_mode='r' to save memory if needed,
        # but given 220GB RAM, loading into memory is faster for random access.
        self.node_feats = np.load(self.node_feats_path)
        self.coords = np.load(self.coords_path)
        self.edge_indices = np.load(self.edge_indices_path)
        self.edge_attrs = np.load(self.edge_attrs_path)
        self.mol_map = pd.read_parquet(self.index_map_path)
        # Ensure index is molecule_name for fast lookup
        if self.mol_map.index.name != "molecule_name":
            self.mol_map = self.mol_map.set_index("molecule_name")

    def _compute_and_cache(self):
        # 1. Load Structures
        df = pd.read_csv(self.structures_path)

        # Ensure sorted by molecule and atom_index for contiguous slicing
        df = df.sort_values(["molecule_name", "atom_index"]).reset_index(drop=True)

        # 2. Prepare Lists for Flattened Arrays
        all_node_feats = []
        all_coords = []
        all_edge_indices = []
        all_edge_attrs = []

        mol_map_data = []

        # Track offsets
        node_offset = 0
        edge_offset = 0

        # Group by molecule
        grouped = df.groupby("molecule_name")

        # Iterate (using simple print for progress if needed, but keeping silent as per instructions)
        for mol_name, group in grouped:
            # Extract atom types and coords
            atoms = group["atom"].map(ATOM_MAP).values.astype(np.int32)
            xyz = group[["x", "y", "z"]].values.astype(np.float32)
            num_atoms = len(atoms)

            # Compute Distance Matrix
            # Shape: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
            diff = xyz[:, np.newaxis, :] - xyz[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))

            # Create Radius Graph (d < Cutoff and d > 0 to avoid self-loops in edge index)
            # We explicitly exclude self-loops here (d > 1e-6)
            mask = (dists < RADIUS_CUTOFF) & (dists > 1e-6)
            src, dst = np.where(mask)

            # Compute Edge Weights (1/d^2)
            edge_dists = dists[src, dst]
            weights = 1.0 / (edge_dists**2)

            # Append to lists
            all_node_feats.append(atoms)
            all_coords.append(xyz)

            # Edge indices must be relative to the molecule start (0 to N-1)
            # We stack them as (2, NumEdges)
            edges = np.stack([src, dst], axis=0).astype(np.int32)
            all_edge_indices.append(edges)
            all_edge_attrs.append(weights.astype(np.float32))

            num_edges = len(weights)

            # Record map info
            mol_map_data.append(
                {
                    "molecule_name": mol_name,
                    "node_start": node_offset,
                    "node_count": num_atoms,
                    "edge_start": edge_offset,
                    "edge_count": num_edges,
                }
            )

            node_offset += num_atoms
            edge_offset += num_edges

        # 3. Concatenate and Save
        print("Concatenating arrays...")
        self.node_feats = np.concatenate(all_node_feats, axis=0)
        self.coords = np.concatenate(all_coords, axis=0)

        if len(all_edge_indices) > 0:
            self.edge_indices = np.concatenate(all_edge_indices, axis=1)
            self.edge_attrs = np.concatenate(all_edge_attrs, axis=0)
        else:
            # Handle edge case of no edges found (unlikely)
            self.edge_indices = np.empty((2, 0), dtype=np.int32)
            self.edge_attrs = np.empty((0,), dtype=np.float32)

        self.mol_map = pd.DataFrame(mol_map_data).set_index("molecule_name")

        print("Saving to cache...")
        os.makedirs(self.cache_dir, exist_ok=True)
        np.save(self.node_feats_path, self.node_feats)
        np.save(self.coords_path, self.coords)
        np.save(self.edge_indices_path, self.edge_indices)
        np.save(self.edge_attrs_path, self.edge_attrs)
        self.mol_map.to_parquet(self.index_map_path)

    def get_molecule_data(self, molecule_name):
        """
        Retrieves graph data for a specific molecule.
        Returns: (node_feats, coords, edge_index, edge_attr)
        """
        if molecule_name not in self.mol_map.index:
            raise KeyError(f"Molecule {molecule_name} not found in structure data.")

        info = self.mol_map.loc[molecule_name]

        n_start = int(info["node_start"])
        n_count = int(info["node_count"])
        e_start = int(info["edge_start"])
        e_count = int(info["edge_count"])

        # Slice arrays
        x = self.node_feats[n_start : n_start + n_count]
        pos = self.coords[n_start : n_start + n_count]

        if e_count > 0:
            edge_index = self.edge_indices[:, e_start : e_start + e_count]
            edge_attr = self.edge_attrs[e_start : e_start + e_count]
        else:
            edge_index = np.empty((2, 0), dtype=np.int32)
            edge_attr = np.empty((0,), dtype=np.float32)

        return x, pos, edge_index, edge_attr


class MoleculeGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for Scalar Coupling Prediction.
    Combines tabular coupling data with pre-computed molecular graphs.
    """

    def __init__(self, metadata_df, processor):
        super().__init__()
        self.metadata = metadata_df.reset_index(drop=True)
        self.processor = processor

        # Pre-map types to indices for speed
        self.metadata["type_idx"] = self.metadata["type"].map(TYPE_MAP).astype(np.int64)

    def len(self):
        return len(self.metadata)

    def get(self, idx):
        row = self.metadata.iloc[idx]
        mol_name = row["molecule_name"]

        # Retrieve graph structure
        x, pos, edge_index, edge_attr = self.processor.get_molecule_data(mol_name)

        # Convert to tensors
        x_tensor = torch.from_numpy(x).long()
        edge_index_tensor = torch.from_numpy(edge_index).long()
        edge_attr_tensor = torch.from_numpy(edge_attr).float()

        # Get target pair indices
        atom_0 = int(row["atom_index_0"])
        atom_1 = int(row["atom_index_1"])

        # Compute distance between the target pair
        # We use the cached coordinates
        pos_0 = pos[atom_0]
        pos_1 = pos[atom_1]
        dist = np.linalg.norm(pos_0 - pos_1)

        # Construct Data object
        data = Data(
            x=x_tensor,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor,
            num_nodes=len(x),
        )

        # Add task-specific attributes
        data.target_pair = torch.tensor([atom_0, atom_1], dtype=torch.long).unsqueeze(
            0
        )  # Shape (1, 2)
        data.type_idx = torch.tensor([row["type_idx"]], dtype=torch.long)
        data.dist = torch.tensor([dist], dtype=torch.float)
        data.id = torch.tensor([row["id"]], dtype=torch.long)

        # Add target if available
        if "scalar_coupling_constant" in row:
            data.y = torch.tensor(
                [[row["scalar_coupling_constant"]]], dtype=torch.float
            )

        return data


def get_dataloaders(
    train_path=TRAIN_META_PATH,
    val_path=VAL_META_PATH,
    test_path=TEST_META_PATH,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    debug=DEBUG,
    debug_size=DEBUG_SAMPLE_SIZE,
):
    """
    Factory function to create DataLoaders for Train, Validation, and Test sets.
    """
    # 1. Initialize Structure Processor
    processor = StructureProcessor()
    # Force reprocessing to ensure cache contains all molecules (fixes stale cache KeyError)
    processor.process(load_cached_data=False)

    # 2. Load Metadata
    print("Loading metadata...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 3. Handle Debug Mode
    if debug:
        print(f"DEBUG MODE: Subsampling datasets to {debug_size} samples.")
        df_train = df_train.iloc[:debug_size]
        df_val = df_val.iloc[:debug_size]
        df_test = df_test.iloc[:debug_size]

    # 4. Create Datasets
    print("Creating datasets...")
    train_dataset = MoleculeGraphDataset(df_train, processor)
    val_dataset = MoleculeGraphDataset(df_val, processor)
    test_dataset = MoleculeGraphDataset(df_test, processor)

    # 5. Create DataLoaders
    # PyG DataLoader handles batching of Data objects (creating block diagonal matrices)
    print("Creating dataloaders...")
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

    return train_loader, val_loader, test_loader
