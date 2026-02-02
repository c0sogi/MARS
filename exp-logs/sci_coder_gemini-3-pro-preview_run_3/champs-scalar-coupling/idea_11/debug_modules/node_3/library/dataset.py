import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from library.config import Config
from library.data_prep import DataProcessor


class MolecularData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "triplets":
            return self.edge_index.size(1)
        if key == "coupling_atom_index":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "triplets" or key == "coupling_atom_index":
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


class MolecularGraphDataset(Dataset):
    """
    PyTorch Dataset for Molecular Graphs using Flattened Structure-of-Arrays (SoA).

    This dataset loads pre-processed monolithic numpy arrays containing data for all molecules
    and slices them on-the-fly to construct PyG Data objects. This avoids the overhead of
    handling millions of small files or objects.
    """

    def __init__(self, split="train", debug=False):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            debug (bool): If True, limits the dataset size for debugging.
        """
        super().__init__()
        self.split = split
        self.debug = debug

        # Initialize DataProcessor to ensure data is processed and loaded
        # This handles caching automatically.
        processor = DataProcessor()
        data_dict = processor.process_all(load_cached_data=True)

        # Store references to the large arrays
        # These are numpy arrays loaded in memory (or mmap if modified in DataProcessor, but currently memory)
        self.nodes = data_dict["nodes"]
        self.coords = data_dict["coords"]  # Shape: (N_total_atoms, 3)

        self.edge_indices = data_dict["edge_indices"]  # Shape: (2, E_total)
        self.edge_attrs = data_dict["edge_attrs"]  # Shape: (E_total, 4)

        self.triplets = data_dict["triplets"]  # Shape: (2, T_total)

        self.mol_indices = data_dict["mol_indices"]  # Shape: (M_molecules, 8)

        self.aux_charges = data_dict["aux_charges"]
        self.aux_shielding = data_dict["aux_shielding"]

        self.coupling_meta = data_dict[
            "coupling_meta"
        ]  # Shape: (C_total, 5) -> [a0, a1, type, id, split]
        self.coupling_values = data_dict["coupling_values"]  # Shape: (C_total,)

        # ==========================================
        # Filter Molecules by Split
        # ==========================================
        # mol_indices columns: [n_start, n_cnt, e_start, e_cnt, t_start, t_cnt, c_start, c_cnt]
        c_starts = self.mol_indices[:, 6]
        c_counts = self.mol_indices[:, 7]

        # 1. Identify molecules that actually have couplings
        # (Some structures might exist without target couplings in the CSVs, though unlikely in this clean dataset)
        valid_mask = c_counts > 0
        valid_mol_indices = np.where(valid_mask)[0]

        # 2. Determine split for these valid molecules
        # We look at the 'split' column (index 4) of the first coupling of each molecule
        valid_c_starts = c_starts[valid_mask]
        mol_splits = self.coupling_meta[valid_c_starts, 4]

        # Map string split to integer code used in DataProcessor
        # 0=Train, 1=Val, 2=Test
        split_map = {"train": 0, "val": 1, "test": 2}
        if split not in split_map:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of {list(split_map.keys())}"
            )
        target_split_code = split_map[split]

        # 3. Select molecules matching the requested split
        match_mask = mol_splits == target_split_code
        self.indices = valid_mol_indices[match_mask]

        # ==========================================
        # Debugging
        # ==========================================
        if self.debug:
            limit = min(len(self.indices), Config.DEBUG_SAMPLE_SIZE)
            self.indices = self.indices[:limit]
            print(
                f"DEBUG MODE: Limited {split} dataset to {len(self.indices)} molecules."
            )
        else:
            print(f"Loaded {split} dataset: {len(self.indices)} molecules.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Constructs a Data object for a single molecule by slicing the global arrays.
        """
        # Get the global index of the molecule
        mol_idx = self.indices[idx]

        # Retrieve offsets and counts from the index array
        # [n_start, n_cnt, e_start, e_cnt, t_start, t_cnt, c_start, c_cnt]
        indices = self.mol_indices[mol_idx]

        n_start, n_cnt = indices[0], indices[1]
        e_start, e_cnt = indices[2], indices[3]
        t_start, t_cnt = indices[4], indices[5]
        c_start, c_cnt = indices[6], indices[7]

        # ==========================================
        # Slice Node Data
        # ==========================================
        # x: Atomic numbers/types (Long)
        x = torch.from_numpy(self.nodes[n_start : n_start + n_cnt]).long()
        # pos: 3D Coordinates (Float)
        pos = torch.from_numpy(self.coords[n_start : n_start + n_cnt])

        # Aux Targets (Standardized)
        aux_q = torch.from_numpy(self.aux_charges[n_start : n_start + n_cnt])
        aux_s = torch.from_numpy(self.aux_shielding[n_start : n_start + n_cnt])

        # ==========================================
        # Slice Edge Data
        # ==========================================
        # edge_index: (2, E_local). The stored indices are already 0-based local to the molecule.
        edge_index = torch.from_numpy(
            self.edge_indices[:, e_start : e_start + e_cnt]
        ).long()
        # edge_attr: (E_local, 4) -> [dist, vec_x, vec_y, vec_z]
        edge_attr = torch.from_numpy(self.edge_attrs[e_start : e_start + e_cnt])

        # ==========================================
        # Slice Triplet Data
        # ==========================================
        # triplets: (2, T_local) -> [incoming_edge_index, outgoing_edge_index]
        triplets = torch.from_numpy(self.triplets[:, t_start : t_start + t_cnt]).long()

        # ==========================================
        # Slice Coupling Data (Targets)
        # ==========================================
        # Meta: [atom0, atom1, type, id, split]
        c_meta = self.coupling_meta[c_start : c_start + c_cnt]
        c_values = self.coupling_values[c_start : c_start + c_cnt]

        # coupling_atom_index: (2, n_couplings)
        coupling_atom_index = torch.from_numpy(c_meta[:, 0:2].T).long()

        # coupling_type: (n_couplings,)
        coupling_type = torch.from_numpy(c_meta[:, 2]).long()

        # coupling_id: (n_couplings,) - Used for submission
        coupling_id = torch.from_numpy(c_meta[:, 3]).long()

        # y: (n_couplings,) - Standardized target values
        y = torch.from_numpy(c_values).float()

        # ==========================================
        # Construct PyG Data Object
        # ==========================================
        data = Data(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            # Custom attributes for this task
            triplets=triplets,
            coupling_atom_index=coupling_atom_index,
            coupling_type=coupling_type,
            coupling_id=coupling_id,
            aux_charge=aux_q,
            aux_shielding=aux_s,
        )

        return data
