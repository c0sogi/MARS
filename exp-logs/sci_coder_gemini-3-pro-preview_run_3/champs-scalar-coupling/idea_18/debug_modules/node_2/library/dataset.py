import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict, Any, Optional

from library.config import Config
from library.data_preprocessing import SoAPreprocessor


class MoleculeGraphDataset(Dataset):
    """
    PyTorch Dataset backed by flattened Structure-of-Arrays (SoA) data.
    Efficiently slices monolithic arrays to retrieve individual molecule graphs.
    """

    def __init__(self, split: str, load_cached_data: bool = True):
        """
        Args:
            split: One of 'train', 'val', 'test'.
            load_cached_data: Whether to try loading from cache first.
        """
        self.split = split

        # Initialize preprocessor and load data
        preprocessor = SoAPreprocessor()
        self.data = preprocessor.process_split(split, load_cached_data=load_cached_data)

        # Unpack commonly accessed arrays for slightly faster attribute access
        self.mol_atom_ptr = self.data["mol_atom_ptr"]
        self.mol_edge_ptr = self.data["mol_edge_ptr"]
        self.mol_coupling_ptr = self.data["mol_coupling_ptr"]

        self.atom_types = self.data["atom_types"]
        self.atom_coords = self.data["atom_coords"]

        self.edge_indices = self.data["edge_indices"]
        self.edge_distances = self.data["edge_distances"]

        self.coupling_pairs = self.data["coupling_pairs"]
        self.coupling_types = self.data["coupling_types"]
        self.coupling_ids = self.data["coupling_ids"]

        # Target values exist only for train/val
        self.coupling_values = self.data.get("coupling_values", None)
        self.has_targets = self.coupling_values is not None

        self.num_molecules = len(self.data["mol_names"])

    def __len__(self) -> int:
        return self.num_molecules

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves the graph structure and couplings for molecule at index `idx`.
        """
        # 1. Slice Atoms
        a_start = self.mol_atom_ptr[idx]
        a_end = self.mol_atom_ptr[idx + 1]

        atoms = self.atom_types[a_start:a_end]
        coords = self.atom_coords[a_start:a_end]

        # 2. Slice Edges
        e_start = self.mol_edge_ptr[idx]
        e_end = self.mol_edge_ptr[idx + 1]

        # edge_indices is (2, E), so we slice the second dimension
        edge_index = self.edge_indices[:, e_start:e_end]
        edge_dist = self.edge_distances[e_start:e_end]

        # 3. Slice Couplings
        c_start = self.mol_coupling_ptr[idx]
        c_end = self.mol_coupling_ptr[idx + 1]

        # coupling_pairs is (2, C)
        c_pairs = self.coupling_pairs[:, c_start:c_end]
        c_types = self.coupling_types[c_start:c_end]
        c_ids = self.coupling_ids[c_start:c_end]

        sample = {
            "num_nodes": a_end - a_start,
            "atom_types": atoms,
            "atom_coords": coords,
            "edge_index": edge_index,
            "edge_attr": edge_dist,
            "coupling_atom_index": c_pairs,
            "coupling_type": c_types,
            "coupling_id": c_ids,
            "mol_name": self.data["mol_names"][idx],
        }

        if self.has_targets:
            sample["coupling_value"] = self.coupling_values[c_start:c_end]

        return sample


class GraphCollate:
    """
    Collates a list of molecule samples into a single disjoint union graph batch.
    Handles index offsetting for edges and coupling pairs.
    """

    def __init__(self, device=None):
        self.device = device  # Optional: can keep on CPU and move later

    def __call__(self, batch_list: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Initialize lists for accumulation
        atom_types_list = []
        atom_coords_list = []
        batch_idx_list = []

        edge_index_list = []
        edge_attr_list = []

        coupling_atom_index_list = []
        coupling_type_list = []
        coupling_id_list = []
        coupling_value_list = []

        # Track cumulative node count for index offsetting
        node_offset = 0

        has_targets = "coupling_value" in batch_list[0]

        for i, sample in enumerate(batch_list):
            num_nodes = sample["num_nodes"]

            # --- Nodes ---
            atom_types_list.append(sample["atom_types"])
            atom_coords_list.append(sample["atom_coords"])

            # Create batch vector [0, 0, ..., 1, 1, ...]
            # Using np.full is slightly faster than creating tensor and repeating
            batch_idx_list.append(np.full(num_nodes, i, dtype=np.int64))

            # --- Edges ---
            # Offset edge indices by current total node count
            if sample["edge_index"].shape[1] > 0:
                edge_index_list.append(sample["edge_index"] + node_offset)
                edge_attr_list.append(sample["edge_attr"])

            # --- Couplings ---
            # Offset coupling atom indices
            if sample["coupling_atom_index"].shape[1] > 0:
                coupling_atom_index_list.append(
                    sample["coupling_atom_index"] + node_offset
                )
                coupling_type_list.append(sample["coupling_type"])
                coupling_id_list.append(sample["coupling_id"])

                if has_targets:
                    coupling_value_list.append(sample["coupling_value"])

            # Update offset
            node_offset += num_nodes

        # --- Concatenation & Tensor Conversion ---

        # Nodes
        x = torch.from_numpy(np.concatenate(atom_types_list)).long()
        pos = torch.from_numpy(np.concatenate(atom_coords_list)).float()
        batch = torch.from_numpy(np.concatenate(batch_idx_list)).long()

        # Edges
        if edge_index_list:
            edge_index = torch.from_numpy(
                np.concatenate(edge_index_list, axis=1)
            ).long()
            edge_attr = torch.from_numpy(np.concatenate(edge_attr_list)).float()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.float)

        # Couplings
        if coupling_atom_index_list:
            coupling_atom_index = torch.from_numpy(
                np.concatenate(coupling_atom_index_list, axis=1)
            ).long()
            coupling_type = torch.from_numpy(np.concatenate(coupling_type_list)).long()
            coupling_id = torch.from_numpy(np.concatenate(coupling_id_list)).long()
        else:
            coupling_atom_index = torch.empty((2, 0), dtype=torch.long)
            coupling_type = torch.empty((0,), dtype=torch.long)
            coupling_id = torch.empty((0,), dtype=torch.long)

        batch_dict = {
            "x": x,
            "pos": pos,
            "batch": batch,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "coupling_atom_index": coupling_atom_index,
            "coupling_type": coupling_type,
            "coupling_id": coupling_id,
        }

        if has_targets:
            if coupling_value_list:
                coupling_value = torch.from_numpy(
                    np.concatenate(coupling_value_list)
                ).float()
            else:
                coupling_value = torch.empty((0,), dtype=torch.float)
            batch_dict["coupling_value"] = coupling_value

        return batch_dict
