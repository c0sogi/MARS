import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from library.config import Config
from library.graph_builder import DualGraphBuilder


class DualGraphData(Data):
    """
    Custom PyG Data object for Dual Graphs (Atom Graph + Line Graph).
    Handles correct batching increment logic for the Line Graph and Targets.
    """

    def __inc__(self, key, value, *args, **kwargs):
        # The 'line_edge_index' contains indices of edges in the Atom Graph.
        # When batching multiple graphs, these indices must be incremented
        # by the number of *edges* in the previous graphs, not the number of nodes.
        if key == "line_edge_index":
            return self.edge_attr.size(0)

        # 'target_index' contains indices of nodes (atoms).
        # These should be incremented by the number of nodes.
        if key == "target_index":
            return self.num_nodes

        # Standard behavior for edge_index is to shift by num_nodes
        if key == "edge_index":
            return self.num_nodes

        return super().__inc__(key, value, *args, **kwargs)


class MoleculeDataset(Dataset):
    """
    PyTorch Dataset for Scalar Coupling Prediction.
    Loads pre-processed dual graphs from cache or generates them via DualGraphBuilder.
    Efficiently slices large concatenated arrays to return individual molecule graphs.
    """

    def __init__(self, split="train", load_cached=True):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached (bool): Whether to try loading from disk cache.
        """
        super().__init__()
        self.split = split

        # Select Metadata Path based on split
        if split == "train":
            meta_path = Config.TRAIN_META_PATH
        elif split == "val":
            meta_path = Config.VAL_META_PATH
        elif split == "test":
            meta_path = Config.TEST_META_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        # Initialize Builder and Load Data
        # The builder handles caching internally based on the split name
        builder = DualGraphBuilder()
        data_dict = builder.process_split(split, meta_path, load_cached=load_cached)

        print(f"Initializing {split} dataset from memory...")

        # ----------------------------------------------------------
        # 1. Convert Numpy Arrays to PyTorch Tensors
        # ----------------------------------------------------------
        # We keep data on CPU to avoid VRAM overflow; it moves to GPU in the training loop.
        self.node_x = torch.from_numpy(data_dict["node_x"]).long()
        self.node_batch = torch.from_numpy(data_dict["node_batch"]).long()

        self.edge_index = torch.from_numpy(data_dict["edge_index"]).long()
        self.edge_attr = torch.from_numpy(data_dict["edge_attr"]).float()
        self.edge_batch = torch.from_numpy(data_dict["edge_batch"]).long()

        self.line_edge_index = torch.from_numpy(data_dict["line_edge_index"]).long()
        self.line_edge_attr = torch.from_numpy(data_dict["line_edge_attr"]).float()
        # Note: Line edges don't have a direct batch array in the dict; we derive it later.

        self.target_index = torch.from_numpy(data_dict["target_index"]).long()
        self.target_type = torch.from_numpy(data_dict["target_type"]).long()
        self.target_val = torch.from_numpy(data_dict["target_val"]).float()
        self.target_batch = torch.from_numpy(data_dict["target_batch"]).long()
        self.meta_ids = torch.from_numpy(data_dict["meta_ids"]).long()

        self.aux_shielding = torch.from_numpy(data_dict["aux_shielding"]).float()
        self.aux_charges = torch.from_numpy(data_dict["aux_charges"]).float()

        self.mol_dipole = torch.from_numpy(data_dict["mol_dipole"]).float()
        self.mol_potential = torch.from_numpy(data_dict["mol_potential"]).float()

        self.num_molecules = self.mol_dipole.size(0)

        # ----------------------------------------------------------
        # 2. Pre-compute Slices for O(1) Access
        # ----------------------------------------------------------
        # We need to know the start index and count of items for each molecule
        # for every array type (nodes, edges, line_edges, targets).

        def compute_slices(batch_tensor, num_mols):
            """
            Computes start indices and counts for each molecule ID in a sorted batch tensor.
            """
            # batch_tensor is sorted [0,0,..,1,1,..]
            # bincount gives the number of elements per molecule
            counts = torch.bincount(batch_tensor, minlength=num_mols)
            # cumsum gives the end indices
            ends = torch.cumsum(counts, dim=0)
            # shift to get start indices
            starts = torch.cat([torch.zeros(1, dtype=torch.long), ends[:-1]])
            return starts, counts

        # Nodes
        self.node_starts, self.node_counts = compute_slices(
            self.node_batch, self.num_molecules
        )

        # Edges
        self.edge_starts, self.edge_counts = compute_slices(
            self.edge_batch, self.num_molecules
        )

        # Targets
        self.target_starts, self.target_counts = compute_slices(
            self.target_batch, self.num_molecules
        )

        # Line Edges
        # We don't have line_edge_batch directly.
        # However, line_edge_index[0] contains indices of the source edges.
        # We can look up which molecule those edges belong to using edge_batch.
        if self.line_edge_index.size(1) > 0:
            # Map line_edge -> source_edge -> molecule_index
            src_edges = self.line_edge_index[0]
            line_mol_indices = self.edge_batch[src_edges]
            self.line_edge_starts, self.line_edge_counts = compute_slices(
                line_mol_indices, self.num_molecules
            )
        else:
            self.line_edge_starts = torch.zeros(self.num_molecules, dtype=torch.long)
            self.line_edge_counts = torch.zeros(self.num_molecules, dtype=torch.long)

    def __len__(self):
        return self.num_molecules

    def __getitem__(self, idx):
        """
        Constructs and returns a DualGraphData object for the molecule at `idx`.
        Slices the global arrays and adjusts indices to be 0-based for the subgraph.
        """
        # --- Nodes ---
        n_start = self.node_starts[idx]
        n_count = self.node_counts[idx]

        x = self.node_x[n_start : n_start + n_count]
        aux_shielding = self.aux_shielding[n_start : n_start + n_count]
        aux_charges = self.aux_charges[n_start : n_start + n_count]

        # --- Edges (Atom Graph) ---
        e_start = self.edge_starts[idx]
        e_count = self.edge_counts[idx]

        # Global indices -> Local indices (subtract node offset)
        # The global edge_index is constructed such that subtracting n_start makes it 0-based
        edge_index = self.edge_index[:, e_start : e_start + e_count] - n_start
        edge_attr = self.edge_attr[e_start : e_start + e_count]

        # --- Line Edges (Line Graph) ---
        l_start = self.line_edge_starts[idx]
        l_count = self.line_edge_counts[idx]

        # Global indices -> Local indices (subtract edge offset)
        # Line graph nodes correspond to Atom Graph edges, so we subtract e_start
        line_edge_index = self.line_edge_index[:, l_start : l_start + l_count] - e_start
        line_edge_attr = self.line_edge_attr[l_start : l_start + l_count]

        # --- Targets ---
        t_start = self.target_starts[idx]
        t_count = self.target_counts[idx]

        # Target indices refer to atoms, so subtract node offset
        target_index = self.target_index[:, t_start : t_start + t_count] - n_start
        target_type = self.target_type[t_start : t_start + t_count]
        target_val = self.target_val[t_start : t_start + t_count]
        meta_ids = self.meta_ids[t_start : t_start + t_count]

        # --- Global Properties ---
        # Unsqueeze to keep batch dimension (1, D)
        mol_dipole = self.mol_dipole[idx].unsqueeze(0)
        mol_potential = self.mol_potential[idx].unsqueeze(0)

        # Create Data Object
        data = DualGraphData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            line_edge_index=line_edge_index,
            line_edge_attr=line_edge_attr,
            target_index=target_index,
            target_type=target_type,
            y=target_val,
            id=meta_ids,
            aux_shielding=aux_shielding,
            aux_charges=aux_charges,
            mol_dipole=mol_dipole,
            mol_potential=mol_potential,
            num_nodes=n_count,  # Explicitly set num_nodes for safety
        )

        return data
