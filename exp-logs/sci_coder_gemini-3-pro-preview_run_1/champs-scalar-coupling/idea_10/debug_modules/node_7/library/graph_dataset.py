import os
import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from torch_geometric.data import Data, InMemoryDataset
from library import config, utils


class GaussianSmearing(torch.nn.Module):
    """
    Expands distances in a Gaussian basis (RBF).
    """

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1)) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class MolecularGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric Dataset for Molecular Coupling Prediction.
    Constructs graphs based on Adaptive Covalent Radii.
    """

    def __init__(self, root, split="train", transform=None, pre_transform=None):
        self.split = split
        # Map atomic numbers to dense indices (0-4) for embeddings
        # H(1), C(6), N(7), O(8), F(9)
        self.atom_mapper = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
        # Inverse mapping for coupling types
        self.type_mapper = {t: i for i, t in enumerate(config.COUPLING_TYPES)}

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return [f"graph_data_{self.split}.pt"]

    def process(self):
        print(f"Processing {self.split} dataset... This may take a few minutes.")

        # 1. Load Data
        df_meta = utils.load_metadata(self.split)
        df_structures = utils.load_structures(load_cached_data=True)

        # 2. Setup RBF Expander
        rbf = GaussianSmearing(
            start=0.0,
            stop=config.GNN_PARAMS["cutoff"],
            num_gaussians=config.GNN_PARAMS["num_rbf"],
        )

        # 3. Prepare Structure Lookup
        # Create a dictionary for fast access: molecule_name -> (atomic_numbers, coordinates)
        print("Grouping structures by molecule...")
        # Sort to ensure consistent order if needed, though groupby preserves order usually
        df_structures = df_structures.sort_values(["molecule_name", "atom_index"])

        # Extract arrays
        mol_names = df_structures["molecule_name"].values
        atom_nums = df_structures["atomic_number"].values
        coords = df_structures[["x", "y", "z"]].values

        # Identify split indices
        # We use numpy unique to get start indices for each molecule
        unique_mols, start_indices = np.unique(mol_names, return_index=True)
        # Append end index
        end_indices = np.append(start_indices[1:], len(mol_names))

        mol_struct_map = {}
        for i, name in enumerate(unique_mols):
            s, e = start_indices[i], end_indices[i]
            mol_struct_map[name] = {"z": atom_nums[s:e], "pos": coords[s:e]}

        # 4. Group Metadata
        print("Grouping metadata by molecule...")
        meta_grouped = df_meta.groupby("molecule_name")

        data_list = []

        # 5. Iterate and Build Graphs
        # We iterate over the molecules present in the metadata split
        unique_meta_mols = df_meta["molecule_name"].unique()

        print(f"Building graphs for {len(unique_meta_mols)} molecules...")

        for mol_name in unique_meta_mols:
            if mol_name not in mol_struct_map:
                continue

            # --- Node Features ---
            struct_info = mol_struct_map[mol_name]
            atoms = struct_info["z"]
            pos = struct_info["pos"]

            # Map atomic numbers to indices
            x = torch.tensor(
                [self.atom_mapper.get(a, 0) for a in atoms], dtype=torch.long
            ).view(-1, 1)
            pos_t = torch.tensor(pos, dtype=torch.float)

            # --- Edge Construction (Adaptive Radii) ---
            # Calculate pairwise distances
            # pdist returns condensed distance matrix
            dists_condensed = pdist(pos)
            dists_matrix = squareform(dists_condensed)

            # Get radii for all atoms
            radii = np.array([config.COVALENT_RADII.get(a, 0.7) for a in atoms])

            # Create threshold matrix: r_i + r_j + tolerance
            # broadcasting: (N, 1) + (1, N)
            radii_matrix = (
                radii[:, None] + radii[None, :] + config.CONNECTIVITY_TOLERANCE
            )

            # Mask: connected if dist < threshold AND dist > 0 (no self loops)
            adj_mask = (dists_matrix <= radii_matrix) & (dists_matrix > 0)

            # Convert to edge_index
            src, dst = np.where(adj_mask)
            edge_index = torch.tensor(np.vstack((src, dst)), dtype=torch.long)

            # Edge Attributes (RBF of distances)
            if len(src) > 0:
                edge_dists = torch.tensor(dists_matrix[src, dst], dtype=torch.float)
                edge_attr = rbf(edge_dists)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty(
                    (0, config.GNN_PARAMS["num_rbf"]), dtype=torch.float
                )

            # --- Coupling Targets ---
            # Get all couplings for this molecule
            group = meta_grouped.get_group(mol_name)

            # Indices of the atom pairs
            idx0 = group["atom_index_0"].values
            idx1 = group["atom_index_1"].values
            coupling_edge_index = torch.tensor(
                np.vstack((idx0, idx1)), dtype=torch.long
            )

            # Coupling Types
            types = group["type"].values
            type_idx = torch.tensor(
                [self.type_mapper[t] for t in types], dtype=torch.long
            )

            # IDs
            ids = torch.tensor(group["id"].values, dtype=torch.long)

            # Targets (y)
            if "scalar_coupling_constant" in group.columns:
                y = torch.tensor(
                    group["scalar_coupling_constant"].values, dtype=torch.float
                )
            else:
                # Test set placeholder
                y = torch.zeros(len(group), dtype=torch.float)

            # Create Data Object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=pos_t,
                coupling_edge_index=coupling_edge_index,
                coupling_type_idx=type_idx,
                y=y,
                id=ids,
                num_nodes=len(atoms),  # Explicitly set num_nodes
            )

            data_list.append(data)

        print(f"Graph construction complete. Saving {len(data_list)} graphs...")

        if len(data_list) == 0:
            # Cite debug_lesson_7: Handle empty results explicitly to prevent downstream crashes
            raise RuntimeError(
                f"No graphs were constructed for split '{self.split}'. "
                "This indicates that the molecules in the metadata were not found in the structure files."
            )

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __inc__(self, key, value, *args, **kwargs):
        # This tells the DataLoader how to increment indices when batching
        if key == "coupling_edge_index":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


def get_dataset(split="train", load_cached_data=True):
    """
    Factory function to get the MolecularGraphDataset.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If False, forces reprocessing (handled by PyG internal check + manual removal if needed,
                                 but here we rely on PyG's processed_file_names check).
                                 To strictly enforce 'recompute if load_cached_data=False', we check existence.
    """
    root = os.path.join(config.WORKING_DIR, "graph_data")
    os.makedirs(root, exist_ok=True)

    # If load_cached_data is False, we remove the processed file to force regeneration
    dataset = MolecularGraphDataset(root, split=split)
    processed_file = dataset.processed_paths[0]

    if not load_cached_data and os.path.exists(processed_file):
        print(f"Force reprocessing: Removing {processed_file}")
        os.remove(processed_file)
        # Re-instantiate to trigger process()
        dataset = MolecularGraphDataset(root, split=split)

    return dataset
