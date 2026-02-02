import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from library.config import Config


class SoAPreprocessor:
    """
    Preprocessing module that converts raw CSV/XYZ data into a flattened
    Structure-of-Arrays (SoA) format optimized for molecule-parallel GNN training.
    """

    def __init__(self):
        self.atom_map = Config.ATOM_MAP
        self.coupling_map = Config.COUPLING_TYPE_MAP
        self.rbf_end = Config.RBF_END

    def process_split(self, split_name: str, load_cached_data: bool = True):
        """
        Processes a specific data split (train, val, test).

        Args:
            split_name: One of 'train', 'val', 'test'.
            load_cached_data: If True, attempts to load from disk first.

        Returns:
            Dictionary containing the processed numpy arrays.
        """
        # 1. Determine Paths and Configuration
        if split_name == "train":
            meta_path = Config.TRAIN_METADATA
            cache_dir = Config.CACHE_DIR_TRAIN
            has_targets = True
        elif split_name == "val":
            meta_path = Config.VAL_METADATA
            cache_dir = Config.CACHE_DIR_VAL
            has_targets = True
        elif split_name == "test":
            meta_path = Config.TEST_METADATA
            cache_dir = Config.CACHE_DIR_TEST
            has_targets = False
        else:
            raise ValueError(f"Unknown split: {split_name}")

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        # Define expected files
        expected_files = [
            "atom_types.npy",
            "atom_coords.npy",
            "mol_atom_ptr.npy",
            "edge_indices.npy",
            "edge_distances.npy",
            "mol_edge_ptr.npy",
            "coupling_pairs.npy",
            "coupling_types.npy",
            "coupling_ids.npy",
            "mol_coupling_ptr.npy",
            "mol_names.npy",
        ]
        if has_targets:
            expected_files.append("coupling_values.npy")

        # 2. Try Loading from Cache
        if load_cached_data:
            all_exist = all(
                os.path.exists(os.path.join(cache_dir, f)) for f in expected_files
            )
            if all_exist:
                print(f"Loading cached data for '{split_name}' from {cache_dir}...")
                data = {}
                for f in expected_files:
                    key = f.replace(".npy", "")
                    data[key] = np.load(os.path.join(cache_dir, f))
                return data

        # 3. Process from Scratch
        print(f"Processing data for '{split_name}' (Cache miss or force reload)...")

        # Load Metadata and Structures
        df_meta = pd.read_csv(meta_path)
        df_struct = pd.read_csv(Config.STRUCTURES_CSV)

        # Apply Debugging Limits
        if Config.DEBUG:
            print(f"DEBUG Mode: Sampling {Config.DEBUG_SAMPLE_SIZE} molecules.")
            unique_mols = df_meta["molecule_name"].unique()
            if len(unique_mols) > Config.DEBUG_SAMPLE_SIZE:
                sample_mols = unique_mols[: Config.DEBUG_SAMPLE_SIZE]
                df_meta = df_meta[df_meta["molecule_name"].isin(sample_mols)]

        # Identify relevant molecules for this split
        relevant_mols = df_meta["molecule_name"].unique()
        # Sort for deterministic order
        relevant_mols = sorted(relevant_mols)

        # Filter structures to only relevant molecules
        df_struct = df_struct[df_struct["molecule_name"].isin(relevant_mols)]

        # Sort structures by molecule and atom_index to ensure 0..N-1 indexing matches row order
        df_struct = df_struct.sort_values(["molecule_name", "atom_index"])

        # Group dataframes for fast iteration
        struct_grp = df_struct.groupby("molecule_name")
        meta_grp = df_meta.groupby("molecule_name")

        # Initialize Accumulators (Lists are faster than repetitive np.append)
        # Atoms
        all_atom_types = []
        all_atom_coords = []
        mol_atom_ptr = [0]

        # Edges
        all_edge_indices = []
        all_edge_dists = []
        mol_edge_ptr = [0]

        # Couplings
        all_coup_pairs = []
        all_coup_types = []
        all_coup_ids = []
        all_coup_vals = []
        mol_coup_ptr = [0]

        saved_mol_names = []

        # 4. Iterate over Molecules
        for mol_name in relevant_mols:
            # --- Process Structure (Nodes & Edges) ---
            try:
                mol_struct = struct_grp.get_group(mol_name)
            except KeyError:
                # Should not happen if logic is correct
                continue

            # Extract Atom Features
            # Map atom symbol to integer
            atoms = mol_struct["atom"].map(self.atom_map).values.astype(np.int8)
            coords = mol_struct[["x", "y", "z"]].values.astype(np.float32)
            n_atoms = len(atoms)

            all_atom_types.append(atoms)
            all_atom_coords.append(coords)
            mol_atom_ptr.append(mol_atom_ptr[-1] + n_atoms)

            # Compute Edges (Radius Graph)
            if n_atoms > 1:
                # pdist computes pairwise distances (condensed)
                dists_condensed = pdist(coords)
                dists = squareform(dists_condensed)

                # Select interactions within cutoff, excluding self-loops
                # np.where returns (row_indices, col_indices)
                src, dst = np.where((dists < self.rbf_end) & (dists > 0))

                edge_d = dists[src, dst].astype(np.float32)
                edges = np.stack([src, dst], axis=0).astype(np.int64)  # Shape (2, E)
            else:
                # Handle single atom case (rare/impossible in this dataset but good for robustness)
                edge_d = np.empty((0,), dtype=np.float32)
                edges = np.empty((2, 0), dtype=np.int64)

            all_edge_indices.append(edges)
            all_edge_dists.append(edge_d)
            mol_edge_ptr.append(mol_edge_ptr[-1] + len(edge_d))

            # --- Process Couplings (Targets) ---
            if mol_name in meta_grp.groups:
                mol_meta = meta_grp.get_group(mol_name)

                # Extract Coupling Features
                # Transpose to get (2, C) shape for pairs
                c_pairs = mol_meta[["atom_index_0", "atom_index_1"]].values.T.astype(
                    np.int64
                )
                c_types = mol_meta["type"].map(self.coupling_map).values.astype(np.int8)
                c_ids = mol_meta["id"].values.astype(np.int64)

                all_coup_pairs.append(c_pairs)
                all_coup_types.append(c_types)
                all_coup_ids.append(c_ids)

                if has_targets:
                    c_vals = mol_meta["scalar_coupling_constant"].values.astype(
                        np.float32
                    )
                    all_coup_vals.append(c_vals)

                mol_coup_ptr.append(mol_coup_ptr[-1] + len(c_types))
            else:
                # No couplings for this molecule
                mol_coup_ptr.append(mol_coup_ptr[-1])

            saved_mol_names.append(mol_name)

        # 5. Concatenate and Finalize Arrays
        data = {}

        # Nodes
        data["atom_types"] = np.concatenate(all_atom_types)
        data["atom_coords"] = np.concatenate(all_atom_coords)
        data["mol_atom_ptr"] = np.array(mol_atom_ptr, dtype=np.int64)

        # Edges
        # axis=1 because shape is (2, E)
        data["edge_indices"] = (
            np.concatenate(all_edge_indices, axis=1)
            if all_edge_indices
            else np.empty((2, 0), dtype=np.int64)
        )
        data["edge_distances"] = (
            np.concatenate(all_edge_dists)
            if all_edge_dists
            else np.empty((0,), dtype=np.float32)
        )
        data["mol_edge_ptr"] = np.array(mol_edge_ptr, dtype=np.int64)

        # Couplings
        if all_coup_pairs:
            data["coupling_pairs"] = np.concatenate(all_coup_pairs, axis=1)  # (2, C)
            data["coupling_types"] = np.concatenate(all_coup_types)
            data["coupling_ids"] = np.concatenate(all_coup_ids)
            if has_targets:
                data["coupling_values"] = np.concatenate(all_coup_vals)
        else:
            data["coupling_pairs"] = np.empty((2, 0), dtype=np.int64)
            data["coupling_types"] = np.empty((0,), dtype=np.int8)
            data["coupling_ids"] = np.empty((0,), dtype=np.int64)
            if has_targets:
                data["coupling_values"] = np.empty((0,), dtype=np.float32)

        data["mol_coupling_ptr"] = np.array(mol_coup_ptr, dtype=np.int64)
        data["mol_names"] = np.array(saved_mol_names)

        # 6. Save to Cache
        print(f"Saving processed data to {cache_dir}...")
        for k, v in data.items():
            np.save(os.path.join(cache_dir, f"{k}.npy"), v)

        print(f"Successfully processed {len(saved_mol_names)} molecules.")
        return data
