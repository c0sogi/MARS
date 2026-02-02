import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.geometry import process_and_cache_dataset, Geometry


class CouplingDataset(Dataset):
    """
    PyTorch Dataset for molecular coupling prediction.
    Manages loading, slicing, and serving molecular graphs and coupling targets.
    """

    def __init__(self, metadata_path, cache_path, load_cached_data=True, split="train"):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            cache_path (str): Path to the .npz cache file for the processed dataset.
            load_cached_data (bool): Whether to attempt loading from cache.
            split (str): 'train', 'val', or 'test'. Used for naming slice cache files.
        """
        self.split = split
        self.metadata_path = metadata_path

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        self.metadata_df = pd.read_csv(metadata_path)

        # Process or load the full dataset (concatenated arrays)
        self.data = process_and_cache_dataset(
            self.metadata_df, cache_path, load_cached_data=load_cached_data
        )

        # Compute or load slice indices to separate molecules
        self._setup_slices(cache_path, load_cached_data)

    def _setup_slices(self, data_cache_path, load_cached_data):
        """
        Computes start/end indices for atoms, edges, triplets, and couplings
        for each molecule to allow efficient retrieval.
        """
        # Define path for slices cache
        slices_path = os.path.join(
            os.path.dirname(data_cache_path), f"{self.split}_slices.npz"
        )

        # Try to load slices
        if load_cached_data and os.path.exists(slices_path):
            try:
                slices = np.load(slices_path)
                self.atom_slices = slices["atom_slices"]
                self.edge_slices = slices["edge_slices"]
                self.triplet_slices = slices["triplet_slices"]
                self.coupling_slices = slices["coupling_slices"]
                self.num_molecules = len(self.atom_slices) - 1
                return
            except Exception as e:
                print(f"Failed to load slices: {e}. Recomputing...")

        print(f"Computing dataset slices for {self.split}...")

        # Replicate the molecule order logic from process_and_cache_dataset
        grouped = self.metadata_df.groupby("molecule_name")
        unique_molecules = list(grouped.groups.keys())

        # Handle DEBUG mode subsetting to match data generation
        if Config.DEBUG:
            unique_molecules = unique_molecules[: Config.DEBUG_SUBSET_SIZE]

        self.num_molecules = len(unique_molecules)

        # 1. Atom Slices
        # We need to know how many atoms each molecule has.
        # Reading the first line of XYZ files is fast.
        atom_counts = []
        for mol_name in unique_molecules:
            xyz_path = os.path.join(Config.STRUCTURES_DIR, f"{mol_name}.xyz")
            with open(xyz_path, "r") as f:
                line = f.readline()
                atom_counts.append(int(line.strip()))

        self.atom_slices = np.concatenate([[0], np.cumsum(atom_counts)])

        # 2. Edge Slices
        # Edges are sorted by source atom index globally.
        # We find where the source atom index jumps to the next molecule's range.
        edge_src = self.data["edge_index"][0]
        # searchsorted finds indices where elements should be inserted to maintain order
        self.edge_slices = np.searchsorted(edge_src, self.atom_slices)
        # Ensure the last slice matches total edges
        self.edge_slices[-1] = self.data["num_edges_total"]

        # 3. Triplet Slices
        # Triplets are sorted by the edge index they reference (edge_kj).
        # We find where the edge index jumps to the next molecule's range.
        if self.data["triplet_indices"].shape[1] > 0:
            triplet_edge_idx = self.data["triplet_indices"][0]
            self.triplet_slices = np.searchsorted(triplet_edge_idx, self.edge_slices)
        else:
            self.triplet_slices = np.zeros(len(self.edge_slices), dtype=np.int64)

        # 4. Coupling Slices
        # coupling_mol_indices contains the molecule index for each coupling.
        # It is sorted by construction (0,0,0, 1,1, ...).
        mol_indices = np.arange(self.num_molecules + 1)
        self.coupling_slices = np.searchsorted(
            self.data["coupling_mol_indices"], mol_indices
        )

        # Save slices
        np.savez(
            slices_path,
            atom_slices=self.atom_slices,
            edge_slices=self.edge_slices,
            triplet_slices=self.triplet_slices,
            coupling_slices=self.coupling_slices,
        )

    def __len__(self):
        return self.num_molecules

    def __getitem__(self, idx):
        """
        Returns the graph and coupling data for the molecule at index `idx`.
        Indices are shifted to be local (0-based) for the molecule.
        """
        # Get ranges
        a_start, a_end = self.atom_slices[idx], self.atom_slices[idx + 1]
        e_start, e_end = self.edge_slices[idx], self.edge_slices[idx + 1]
        t_start, t_end = self.triplet_slices[idx], self.triplet_slices[idx + 1]
        c_start, c_end = self.coupling_slices[idx], self.coupling_slices[idx + 1]

        # Extract data
        sample = {
            "atom_types": self.data["atom_types"][a_start:a_end],
            "coords": self.data["coords"][a_start:a_end],
            # Shift edge indices to local (0 to num_atoms-1)
            "edge_index": self.data["edge_index"][:, e_start:e_end] - a_start,
            "edge_rbf": self.data["edge_rbf"][e_start:e_end],
            # Shift triplet indices to local (0 to num_edges-1)
            "triplet_indices": self.data["triplet_indices"][:, t_start:t_end] - e_start,
            "triplet_sbf": self.data["triplet_sbf"][t_start:t_end],
            # Shift coupling atom indices to local
            "coupling_atom_0": self.data["coupling_atom_0"][c_start:c_end] - a_start,
            "coupling_atom_1": self.data["coupling_atom_1"][c_start:c_end] - a_start,
            "coupling_types": self.data["coupling_types"][c_start:c_end],
        }

        # Include targets if available
        if "coupling_targets" in self.data:
            sample["coupling_targets"] = self.data["coupling_targets"][c_start:c_end]

        return sample


def collate_graphs(batch):
    """
    Collates a list of molecule samples into a single batch.
    Re-indexes atoms and edges to form a disjoint union graph.
    """
    # Lists to collect features
    atom_types = []
    coords = []
    edge_index = []
    edge_rbf = []
    triplet_indices = []
    triplet_sbf = []
    coupling_atom_0 = []
    coupling_atom_1 = []
    coupling_types = []
    coupling_targets = []

    # Batch indices for Global Attention (maps atoms to graph index)
    batch_indices = []

    # Offsets for re-indexing
    atom_offset = 0
    edge_offset = 0

    for i, sample in enumerate(batch):
        num_atoms = len(sample["atom_types"])
        num_edges = sample["edge_index"].shape[1]

        # Atoms
        atom_types.append(torch.tensor(sample["atom_types"], dtype=torch.long))
        coords.append(torch.tensor(sample["coords"], dtype=torch.float))
        batch_indices.append(torch.full((num_atoms,), i, dtype=torch.long))

        # Edges (shifted)
        edge_index.append(
            torch.tensor(sample["edge_index"], dtype=torch.long) + atom_offset
        )
        edge_rbf.append(torch.tensor(sample["edge_rbf"], dtype=torch.float))

        # Triplets (shifted)
        if sample["triplet_indices"].shape[1] > 0:
            triplet_indices.append(
                torch.tensor(sample["triplet_indices"], dtype=torch.long) + edge_offset
            )
            triplet_sbf.append(torch.tensor(sample["triplet_sbf"], dtype=torch.float))

        # Couplings (shifted)
        coupling_atom_0.append(
            torch.tensor(sample["coupling_atom_0"], dtype=torch.long) + atom_offset
        )
        coupling_atom_1.append(
            torch.tensor(sample["coupling_atom_1"], dtype=torch.long) + atom_offset
        )
        coupling_types.append(torch.tensor(sample["coupling_types"], dtype=torch.long))

        if "coupling_targets" in sample:
            coupling_targets.append(
                torch.tensor(sample["coupling_targets"], dtype=torch.float)
            )

        atom_offset += num_atoms
        edge_offset += num_edges

    # Concatenate
    batch_data = {
        "atom_types": torch.cat(atom_types),
        "coords": torch.cat(coords),
        "batch": torch.cat(batch_indices),
        "edge_index": torch.cat(edge_index, dim=1),
        "edge_rbf": torch.cat(edge_rbf),
        "coupling_atom_0": torch.cat(coupling_atom_0),
        "coupling_atom_1": torch.cat(coupling_atom_1),
        "coupling_types": torch.cat(coupling_types),
    }

    if triplet_indices:
        batch_data["triplet_indices"] = torch.cat(triplet_indices, dim=1)
        batch_data["triplet_sbf"] = torch.cat(triplet_sbf)
    else:
        batch_data["triplet_indices"] = torch.zeros((2, 0), dtype=torch.long)
        batch_data["triplet_sbf"] = torch.zeros((0, Config.NUM_SBF), dtype=torch.float)

    if coupling_targets:
        batch_data["coupling_targets"] = torch.cat(coupling_targets)

    return batch_data
