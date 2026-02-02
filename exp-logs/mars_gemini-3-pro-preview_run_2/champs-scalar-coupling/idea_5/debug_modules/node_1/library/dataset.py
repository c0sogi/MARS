import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import pandas as pd
import numpy as np
import os
from library.config import Config
from library.features import process_dataset


class ChampsDataset(Dataset):
    """
    PyTorch Dataset for the molecular scalar coupling prediction task.
    Loads metadata and molecular graphs, and prepares them for the HGA-Net model.
    """

    def __init__(
        self,
        metadata_path,
        cache_path,
        split="train",
        debug=Config.DEBUG,
        debug_size=Config.DEBUG_SIZE,
    ):
        """
        Args:
            metadata_path (str): Path to the CSV file containing the dataset split.
            cache_path (str): Path to the .npz file for caching processed graphs.
            split (str): 'train', 'val', or 'test'. Used to determine if targets are present.
            debug (bool): If True, limits the dataset size for debugging.
            debug_size (int): Number of samples to use in debug mode.
        """
        self.split = split
        self.metadata_path = metadata_path
        self.cache_path = cache_path

        # Load Metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        if debug:
            print(f"DEBUG Mode: Limiting {split} dataset to {debug_size} samples.")
            self.df = self.df.iloc[:debug_size].reset_index(drop=True)

        # Load/Process Graphs
        # process_dataset handles caching logic internally (loading from npz or processing from scratch)
        self.graphs_dict = process_dataset(
            metadata_path=metadata_path, cache_path=cache_path, load_cached_data=True
        )

        # Coupling Type Encoding
        self.type_to_idx = {t: i for i, t in enumerate(Config.COUPLING_TYPES)}

        # Filter DF to only include molecules we successfully processed
        # process_dataset might skip molecules if structure files are missing
        available_mols = set(self.graphs_dict.keys())
        initial_len = len(self.df)
        self.df = self.df[self.df["molecule_name"].isin(available_mols)].reset_index(
            drop=True
        )
        final_len = len(self.df)

        if final_len < initial_len:
            print(
                f"Warning: Dropped {initial_len - final_len} rows from {split} set due to missing structure data."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mol_name = row["molecule_name"]

        # Retrieve Graph Data (numpy arrays)
        graph_data = self.graphs_dict[mol_name]

        # Convert to PyTorch Tensors
        # x: Atomic numbers (Int) -> [NumAtoms]
        x = torch.tensor(graph_data["x"], dtype=torch.long)
        # pos: Coordinates (Float) -> [NumAtoms, 3]
        pos = torch.tensor(graph_data["pos"], dtype=torch.float)
        # edge_index: Graph connectivity (Long) -> [2, NumEdges]
        edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)
        # edge_attr: Edge vectors (Float) -> [NumEdges, 3]
        edge_attr = torch.tensor(graph_data["edge_attr"], dtype=torch.float)

        # Coupling Info
        atom_idx_0 = int(row["atom_index_0"])
        atom_idx_1 = int(row["atom_index_1"])
        coupling_type_str = row["type"]
        coupling_type_idx = self.type_to_idx.get(coupling_type_str, -1)

        # Target
        if "scalar_coupling_constant" in row:
            y = torch.tensor([row["scalar_coupling_constant"]], dtype=torch.float)
        else:
            y = torch.tensor([0.0], dtype=torch.float)  # Dummy for test

        # Create PyG Data Object
        # We attach extra attributes for the task
        data = Data(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            # Task specific fields (wrapped in lists/tensors to ensure correct batching)
            coupling_atom_0=torch.tensor([atom_idx_0], dtype=torch.long),
            coupling_atom_1=torch.tensor([atom_idx_1], dtype=torch.long),
            coupling_type=torch.tensor([coupling_type_idx], dtype=torch.long),
            y=y,
            id=torch.tensor([row.get("id", -1)], dtype=torch.long),
        )

        return data


def collate_graphs(batch):
    """
    Custom collate function to batch PyG Data objects.
    Handles the shifting of node indices for the coupling pairs.
    """
    # Use PyG's Batch.from_data_list to handle standard graph attributes
    # (x, edge_index, edge_attr, batch, ptr)
    batched_data = Batch.from_data_list(batch)

    # Batch.from_data_list automatically increments edge_index.
    # However, our custom fields 'coupling_atom_0' and 'coupling_atom_1'
    # refer to node indices and must be shifted manually based on the graph positions in the batch.

    # batched_data.ptr contains the cumulative number of nodes: [0, N1, N1+N2, ...]
    # The shift for the i-th graph is ptr[i].

    # ptr has shape [B+1]. We use ptr[:-1] to get the starting index of each graph.
    node_offset = batched_data.ptr[:-1]

    # Apply shift to the coupling indices
    # coupling_atom_0 is a [B] tensor (concatenated from [1] tensors)
    batched_data.coupling_atom_0 = batched_data.coupling_atom_0 + node_offset
    batched_data.coupling_atom_1 = batched_data.coupling_atom_1 + node_offset

    return batched_data
