import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import TargetStandardizer


class DataPreprocessor:
    """
    Handles the conversion of raw CSV/XYZ data into a flattened Structure-of-Arrays (SoA)
    format optimized for molecule-parallel GNN training.
    """

    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.flag_path = os.path.join(self.cache_dir, "completed.flag")

    def process(
        self,
        load_cached_data: bool = True,
        debug_sample_size: int = Config.DEBUG_SAMPLE_SIZE,
    ):
        """
        Main processing pipeline.

        Args:
            load_cached_data: If True, checks for cached files and skips processing if found.
            debug_sample_size: If set, limits the number of molecules processed for debugging.
        """
        # 1. Check Cache
        if load_cached_data and os.path.exists(self.flag_path):
            print(f"Loading cached data from {self.cache_dir}...")
            # Ensure standardizer stats are loaded into Config even if we don't re-compute
            self._load_standardizer()
            return

        print("Starting data preprocessing...")

        # 2. Load Metadata
        print("Loading metadata...")
        df_train = pd.read_csv(Config.TRAIN_META_PATH)
        df_val = pd.read_csv(Config.VAL_META_PATH)
        df_test = pd.read_csv(Config.TEST_META_PATH)

        # 3. Debug Sampling
        if debug_sample_size is not None:
            print(f"DEBUG MODE: Sampling {debug_sample_size} molecules per split.")
            train_mols = df_train["molecule_name"].unique()[:debug_sample_size]
            val_mols = df_val["molecule_name"].unique()[:debug_sample_size]
            test_mols = df_test["molecule_name"].unique()[:debug_sample_size]

            df_train = df_train[df_train["molecule_name"].isin(train_mols)].copy()
            df_val = df_val[df_val["molecule_name"].isin(val_mols)].copy()
            df_test = df_test[df_test["molecule_name"].isin(test_mols)].copy()

        # 4. Merge and Tag Splits
        df_train["split"] = 0
        df_val["split"] = 1
        df_test["split"] = 2

        # Initialize test targets as NaN
        if "scalar_coupling_constant" not in df_test.columns:
            df_test["scalar_coupling_constant"] = np.nan

        # Select relevant columns and concatenate
        cols = [
            "id",
            "molecule_name",
            "atom_index_0",
            "atom_index_1",
            "type",
            "scalar_coupling_constant",
            "split",
        ]
        df_all = pd.concat(
            [df_train[cols], df_val[cols], df_test[cols]], ignore_index=True
        )

        # 5. Molecule Indexing (Canonical Order)
        # We define a global order of molecules: Train -> Val -> Test
        unique_mols = df_all["molecule_name"].unique()
        mol_to_idx = {m: i for i, m in enumerate(unique_mols)}
        num_mols = len(unique_mols)

        print(f"Total unique molecules: {num_mols}")

        # Map molecules to IDs and sort couplings by molecule ID
        # This ensures all couplings for a molecule are contiguous
        df_all["mol_id"] = df_all["molecule_name"].map(mol_to_idx)
        df_all.sort_values("mol_id", inplace=True)

        # 6. Process Structures (Atoms)
        print("Processing molecular structures...")
        df_struct = pd.read_csv(Config.STRUCTURES_CSV)

        # Filter to only relevant molecules
        df_struct = df_struct[df_struct["molecule_name"].isin(unique_mols)].copy()

        # Map to IDs and sort
        df_struct["mol_id"] = df_struct["molecule_name"].map(mol_to_idx)
        # Sort by mol_id, then atom_index to ensure atoms 0..N are contiguous
        df_struct.sort_values(["mol_id", "atom_index"], inplace=True)

        # Encode Atom Types
        df_struct["atom_type_idx"] = (
            df_struct["atom"].map(Config.ATOM_TO_IDX).astype(np.int8)
        )

        # Extract flattened arrays
        atom_types = df_struct["atom_type_idx"].values
        atom_coords = df_struct[["x", "y", "z"]].values.astype(np.float32)

        # Create Mol-Atom Map (Start Index, Count)
        mol_sizes = df_struct.groupby("mol_id", sort=True).size().values

        # Validation: Ensure we have sizes for all molecules (0 to num_mols-1)
        if len(mol_sizes) != num_mols:
            # This handles cases where a molecule might be in metadata but missing in structures (unlikely)
            # or if groupby missed empty groups. Reindex fixes this.
            mol_sizes = (
                df_struct.groupby("mol_id", sort=True)
                .size()
                .reindex(range(num_mols), fill_value=0)
                .values
            )

        mol_starts = np.concatenate(([0], np.cumsum(mol_sizes)[:-1])).astype(np.int32)
        mol_counts = mol_sizes.astype(np.int32)
        mol_atom_map = np.stack([mol_starts, mol_counts], axis=1)

        # 7. Process Edges (Radius Graph)
        print(f"Computing radius graphs (Cutoff={Config.CUTOFF}A)...")
        edge_indices_list = []
        edge_attrs_list = []
        mol_edge_map_list = []

        current_edge_offset = 0

        # Iterate over molecules to compute edges
        # Note: Vectorizing over variable-sized molecules is hard, but a loop over 85k mols
        # with numpy inner operations is efficient enough (< 2 mins).
        for i in range(num_mols):
            start = mol_starts[i]
            count = mol_counts[i]

            if count == 0:
                mol_edge_map_list.append([current_edge_offset, 0])
                continue

            # Get atom coordinates for this molecule
            xyz = atom_coords[start : start + count]  # (N, 3)

            # Compute pairwise distances via broadcasting
            # (N, 1, 3) - (1, N, 3) -> (N, N, 3)
            diff = xyz[:, None, :] - xyz[None, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))  # (N, N)

            # Create adjacency mask
            # 1. Distance <= Cutoff
            # 2. Distance > epsilon (remove self-loops)
            mask = (dists <= Config.CUTOFF) & (dists > 1e-6)

            # Get indices (local to molecule)
            src, dst = np.where(mask)
            d = dists[src, dst]

            # Store
            edge_indices_list.append(np.stack([src, dst], axis=1).astype(np.int32))
            edge_attrs_list.append(d.astype(np.float32))

            n_edges = len(d)
            mol_edge_map_list.append([current_edge_offset, n_edges])
            current_edge_offset += n_edges

        # Concatenate all edges
        if edge_indices_list:
            edge_indices = np.concatenate(edge_indices_list, axis=0)
            edge_attrs = np.concatenate(edge_attrs_list, axis=0)
        else:
            edge_indices = np.empty((0, 2), dtype=np.int32)
            edge_attrs = np.empty((0,), dtype=np.float32)

        mol_edge_map = np.array(mol_edge_map_list, dtype=np.int32)

        # 8. Process Couplings
        print("Processing coupling targets...")
        # Encode Coupling Types
        df_all["type_idx"] = df_all["type"].map(Config.TYPE_TO_IDX).astype(np.int8)

        # Extract arrays
        coupling_mol_ids = df_all["mol_id"].values.astype(np.int32)
        # Atom indices in CSV are already local (0-based relative to molecule)
        coupling_atom_indices = df_all[["atom_index_0", "atom_index_1"]].values.astype(
            np.int32
        )
        coupling_types = df_all["type_idx"].values
        coupling_values = df_all["scalar_coupling_constant"].values.astype(np.float32)
        coupling_splits = df_all["split"].values.astype(np.int8)
        coupling_ids = df_all["id"].values.astype(np.int32)

        # Create Mol-Coupling Map
        c_counts = (
            df_all.groupby("mol_id", sort=True)
            .size()
            .reindex(range(num_mols), fill_value=0)
            .values
        )
        c_starts = np.concatenate(([0], np.cumsum(c_counts)[:-1])).astype(np.int32)
        mol_coupling_map = np.stack([c_starts, c_counts.astype(np.int32)], axis=1)

        # 9. Save to Disk
        print(f"Saving processed arrays to {self.cache_dir}...")

        # Atoms
        np.save(os.path.join(self.cache_dir, "atom_types.npy"), atom_types)
        np.save(os.path.join(self.cache_dir, "atom_coords.npy"), atom_coords)
        np.save(os.path.join(self.cache_dir, "mol_atom_map.npy"), mol_atom_map)

        # Edges
        np.save(os.path.join(self.cache_dir, "edge_indices.npy"), edge_indices)
        np.save(os.path.join(self.cache_dir, "edge_attrs.npy"), edge_attrs)
        np.save(os.path.join(self.cache_dir, "mol_edge_map.npy"), mol_edge_map)

        # Couplings
        np.save(os.path.join(self.cache_dir, "coupling_mol_ids.npy"), coupling_mol_ids)
        np.save(
            os.path.join(self.cache_dir, "coupling_atom_indices.npy"),
            coupling_atom_indices,
        )
        np.save(os.path.join(self.cache_dir, "coupling_types.npy"), coupling_types)
        np.save(os.path.join(self.cache_dir, "coupling_values.npy"), coupling_values)
        np.save(os.path.join(self.cache_dir, "coupling_splits.npy"), coupling_splits)
        np.save(os.path.join(self.cache_dir, "coupling_ids.npy"), coupling_ids)
        np.save(os.path.join(self.cache_dir, "mol_coupling_map.npy"), mol_coupling_map)

        # 10. Fit Target Standardizer
        print("Computing target statistics (Standardization)...")
        # Filter for training data only
        train_mask = df_all["split"] == 0
        df_train_subset = df_all[train_mask]

        standardizer = TargetStandardizer()
        standardizer.fit(df_train_subset, load_cached_data=False)

        # Write completion flag
        with open(self.flag_path, "w") as f:
            f.write("done")

        print("Preprocessing complete.")

    def _load_standardizer(self):
        """Helper to load standardizer stats into Config without re-processing data."""
        standardizer = TargetStandardizer()
        standardizer.fit(df=None, load_cached_data=True)
