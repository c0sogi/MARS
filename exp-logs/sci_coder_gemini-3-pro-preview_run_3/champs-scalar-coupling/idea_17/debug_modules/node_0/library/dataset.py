import torch
import numpy as np
import os
from torch.utils.data import Dataset
from library.config import Config


class MoleculeDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed molecule data from SoA arrays.
    Supports 'Molecule-Parallel' training by retrieving full molecular graphs.
    """

    def __init__(self, split: str = "train"):
        """
        Args:
            split: One of 'train', 'val', 'test'.
        """
        self.split = split
        self.split_map = {"train": 0, "val": 1, "test": 2}

        if split not in self.split_map:
            raise ValueError(
                f"Invalid split: {split}. Must be one of {list(self.split_map.keys())}"
            )

        target_split_idx = self.split_map[split]
        cache_dir = Config.CACHE_DIR

        # Ensure cache exists
        if not os.path.exists(os.path.join(cache_dir, "completed.flag")):
            raise RuntimeError(
                f"Cache not found in {cache_dir}. Please run preprocessing first."
            )

        # Load all necessary arrays into memory
        # Given the dataset size and available RAM (220GB), loading full arrays is efficient.
        try:
            self.atom_types = np.load(os.path.join(cache_dir, "atom_types.npy"))
            self.atom_coords = np.load(os.path.join(cache_dir, "atom_coords.npy"))
            self.mol_atom_map = np.load(os.path.join(cache_dir, "mol_atom_map.npy"))

            self.edge_indices = np.load(os.path.join(cache_dir, "edge_indices.npy"))
            self.edge_attrs = np.load(os.path.join(cache_dir, "edge_attrs.npy"))
            self.mol_edge_map = np.load(os.path.join(cache_dir, "mol_edge_map.npy"))

            self.coupling_atom_indices = np.load(
                os.path.join(cache_dir, "coupling_atom_indices.npy")
            )
            self.coupling_types = np.load(os.path.join(cache_dir, "coupling_types.npy"))
            self.coupling_values = np.load(
                os.path.join(cache_dir, "coupling_values.npy")
            )
            self.coupling_ids = np.load(os.path.join(cache_dir, "coupling_ids.npy"))
            self.mol_coupling_map = np.load(
                os.path.join(cache_dir, "mol_coupling_map.npy")
            )

            # Load coupling splits to determine which molecules belong to this dataset split
            coupling_splits = np.load(os.path.join(cache_dir, "coupling_splits.npy"))

        except FileNotFoundError as e:
            raise RuntimeError(f"Missing cached file: {e}")

        # Filter molecules based on split
        # We determine the split of a molecule by looking at the split of its first coupling.
        # Every molecule in this dataset has at least one coupling.
        mol_starts = self.mol_coupling_map[:, 0]

        # Vectorized lookup of split for each molecule
        mol_splits = coupling_splits[mol_starts]

        # Get indices of molecules matching the target split
        self.indices = np.where(mol_splits == target_split_idx)[0]

        print(f"Dataset ({split}) loaded. Molecules: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the global molecule index
        mol_idx = self.indices[idx]

        # --- Atoms ---
        a_start, a_count = self.mol_atom_map[mol_idx]
        # Slice and convert to tensor
        x = torch.from_numpy(self.atom_types[a_start : a_start + a_count]).long()
        pos = torch.from_numpy(self.atom_coords[a_start : a_start + a_count]).float()

        # --- Edges ---
        e_start, e_count = self.mol_edge_map[mol_idx]
        if e_count > 0:
            # Transpose to (2, E) for PyG convention
            edge_index = (
                torch.from_numpy(self.edge_indices[e_start : e_start + e_count])
                .long()
                .t()
            )
            # Ensure edge_attr is (E, 1)
            edge_attr = (
                torch.from_numpy(self.edge_attrs[e_start : e_start + e_count])
                .float()
                .unsqueeze(1)
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)

        # --- Couplings ---
        c_start, c_count = self.mol_coupling_map[mol_idx]
        # Transpose to (2, C)
        coupling_index = (
            torch.from_numpy(self.coupling_atom_indices[c_start : c_start + c_count])
            .long()
            .t()
        )
        coupling_type = torch.from_numpy(
            self.coupling_types[c_start : c_start + c_count]
        ).long()
        coupling_value = torch.from_numpy(
            self.coupling_values[c_start : c_start + c_count]
        ).float()
        coupling_id = torch.from_numpy(
            self.coupling_ids[c_start : c_start + c_count]
        ).long()

        return {
            "x": x,
            "pos": pos,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "coupling_index": coupling_index,
            "coupling_type": coupling_type,
            "coupling_value": coupling_value,
            "coupling_id": coupling_id,
            "num_atoms": a_count,
        }


def collate_molecular_graphs(batch):
    """
    Custom collate function to batch multiple molecular graphs into a single disjoint graph.
    Handles index offsetting for edges and couplings.
    """
    x_list = []
    pos_list = []
    edge_index_list = []
    edge_attr_list = []
    coupling_index_list = []
    coupling_type_list = []
    coupling_value_list = []
    coupling_id_list = []

    batch_list = []
    coupling_batch_list = []

    atom_offset = 0

    for i, data in enumerate(batch):
        num_atoms = data["num_atoms"]

        # Nodes
        x_list.append(data["x"])
        pos_list.append(data["pos"])
        batch_list.append(torch.full((num_atoms,), i, dtype=torch.long))

        # Edges (with offset)
        edge_index_list.append(data["edge_index"] + atom_offset)
        edge_attr_list.append(data["edge_attr"])

        # Couplings (with offset)
        coupling_index_list.append(data["coupling_index"] + atom_offset)
        coupling_type_list.append(data["coupling_type"])
        coupling_value_list.append(data["coupling_value"])
        coupling_id_list.append(data["coupling_id"])
        coupling_batch_list.append(
            torch.full((data["coupling_type"].size(0),), i, dtype=torch.long)
        )

        atom_offset += num_atoms

    # Concatenate
    return {
        "x": torch.cat(x_list, dim=0),
        "pos": torch.cat(pos_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "edge_attr": torch.cat(edge_attr_list, dim=0),
        "coupling_index": torch.cat(coupling_index_list, dim=1),
        "coupling_type": torch.cat(coupling_type_list, dim=0),
        "coupling_value": torch.cat(coupling_value_list, dim=0),
        "coupling_id": torch.cat(coupling_id_list, dim=0),
        "batch": torch.cat(batch_list, dim=0),
        "coupling_batch": torch.cat(coupling_batch_list, dim=0),
    }
