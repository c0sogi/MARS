import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from library.config import Config
from library.utils import Standardizer


class DataFactory:
    """
    Handles the processing of raw molecular data into a flattened, contiguous memory layout
    optimized for Directional Message Passing Neural Networks.
    """

    def __init__(self):
        self.config = Config()
        self.atom_map = self.config.ATOM_MAP
        self.type_map = self.config.TYPE_MAP

    def process_dataset(self, split="train", load_cached_data=True):
        """
        Main entry point to get the processed dataset.
        Checks for cached files; if not found, processes from scratch.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            dict: A dictionary containing numpy arrays of the processed data.
        """
        cache_dir = os.path.join(self.config.PROCESSED_DATA_DIR, split)
        os.makedirs(cache_dir, exist_ok=True)
        flag_file = os.path.join(cache_dir, "completed.flag")

        if load_cached_data and os.path.exists(flag_file):
            print(f"Loading cached {split} data from {cache_dir}...")
            try:
                return self._load_cache(cache_dir)
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        print(f"Processing {split} data from scratch...")
        data = self._process_raw_data(split)

        print(f"Saving {split} data to cache...")
        self._save_cache(data, cache_dir)

        # If training data, compute and save statistics
        if split == "train":
            self._compute_and_save_stats(data)

        return data

    def _process_raw_data(self, split):
        """
        Internal method to process raw CSVs and XYZs into flattened arrays.
        """
        # 1. Load Metadata
        if split == "train":
            meta_path = self.config.TRAIN_META_PATH
        elif split == "val":
            meta_path = self.config.VAL_META_PATH
        else:
            meta_path = self.config.TEST_META_PATH

        df_meta = pd.read_csv(meta_path)

        # Debugging subset
        if self.config.DEBUG:
            print(
                f"DEBUG MODE: Processing only {self.config.DEBUG_SAMPLE_SIZE} molecules."
            )
            unique_mols = df_meta["molecule_name"].unique()[
                : self.config.DEBUG_SAMPLE_SIZE
            ]
            df_meta = df_meta[df_meta["molecule_name"].isin(unique_mols)]

        relevant_mols = df_meta["molecule_name"].unique()

        # 2. Load Structures
        print("Loading structures...")
        df_struct = pd.read_csv(self.config.STRUCTURES_CSV)
        df_struct = df_struct[df_struct["molecule_name"].isin(relevant_mols)]

        # Sort structures to ensure contiguous atoms per molecule
        df_struct = df_struct.sort_values(["molecule_name", "atom_index"]).reset_index(
            drop=True
        )

        # Create a dictionary for fast structure lookup: mol_name -> DataFrame slice
        # Using groupby is efficient here
        struct_groups = df_struct.groupby("molecule_name")

        # 3. Load Auxiliary Data (only for train/val)
        aux_shielding = {}
        aux_charges = {}

        if split in ["train", "val"]:
            print("Loading auxiliary data...")
            # Magnetic Shielding
            df_ms = pd.read_csv(self.config.MAGNETIC_SHIELDING_CSV)
            df_ms = df_ms[df_ms["molecule_name"].isin(relevant_mols)]
            # Calculate isotropic shielding
            df_ms["iso"] = (df_ms["XX"] + df_ms["YY"] + df_ms["ZZ"]) / 3.0
            # Map: (mol_name, atom_index) -> iso
            # To make lookup fast, we can create a dict or merge.
            # Since we iterate by molecule, a dict of dicts or grouped df is best.
            ms_groups = df_ms.groupby("molecule_name")

            # Mulliken Charges
            df_mc = pd.read_csv(self.config.MULLIKEN_CHARGES_CSV)
            df_mc = df_mc[df_mc["molecule_name"].isin(relevant_mols)]
            mc_groups = df_mc.groupby("molecule_name")

        # 4. Group Metadata for fast iteration
        meta_groups = df_meta.groupby("molecule_name")

        # 5. Initialize Lists for Flattened Arrays
        # Nodes
        all_node_x = []
        all_node_pos = []
        all_node_batch = []
        all_aux_shield = []
        all_aux_charge = []

        # Edges
        all_edge_index = []  # (2, M)
        all_edge_attr = []  # (M, 2) -> dist, dummy
        all_edge_batch = []

        # Targets (Couplings)
        all_coupling_val = []
        all_coupling_type = []
        all_coupling_edge_idx = []  # Index into the global edge array
        all_coupling_id = []

        # Counters for global indexing
        node_offset = 0
        edge_offset = 0

        # 6. Iterate over molecules
        print("Constructing graphs and flattening data...")
        # Use sorted unique molecules to ensure deterministic order
        sorted_mols = sorted(relevant_mols)

        for mol_idx, mol_name in enumerate(tqdm(sorted_mols)):
            # --- Node Processing ---
            mol_struct = struct_groups.get_group(mol_name)
            coords = mol_struct[["x", "y", "z"]].values.astype(np.float32)
            atoms = mol_struct["atom"].values
            n_atoms = len(atoms)

            # Map atoms to integers
            atom_ids = np.array([self.atom_map[a] for a in atoms], dtype=np.int64)

            all_node_x.append(atom_ids)
            all_node_pos.append(coords)
            all_node_batch.append(np.full(n_atoms, mol_idx, dtype=np.int64))

            # Aux targets
            if split in ["train", "val"]:
                # Shielding
                if mol_name in ms_groups.groups:
                    ms_data = (
                        ms_groups.get_group(mol_name)
                        .sort_values("atom_index")["iso"]
                        .values
                    )
                    all_aux_shield.append(ms_data.astype(np.float32))
                else:
                    all_aux_shield.append(np.zeros(n_atoms, dtype=np.float32))

                # Charges
                if mol_name in mc_groups.groups:
                    mc_data = (
                        mc_groups.get_group(mol_name)
                        .sort_values("atom_index")["mulliken_charge"]
                        .values
                    )
                    all_aux_charge.append(mc_data.astype(np.float32))
                else:
                    all_aux_charge.append(np.zeros(n_atoms, dtype=np.float32))

            # --- Edge Processing (Radius Graph) ---
            # Compute distance matrix
            # shape (N, N)
            dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

            # Find pairs < cutoff (excluding self-loops)
            mask = (dist_mat <= self.config.CUTOFF_RADIUS) & (dist_mat > 1e-6)
            src_local, dst_local = np.where(mask)

            n_edges = len(src_local)
            if n_edges == 0:
                # Handle isolated atoms case (rare but possible in filtered debug)
                node_offset += n_atoms
                continue

            # Calculate edge features
            # Vector: dst - src
            vecs = coords[dst_local] - coords[src_local]  # (M, 3)
            dists = dist_mat[src_local, dst_local]  # (M,)

            # Global indices for edges
            src_global = src_local + node_offset
            dst_global = dst_local + node_offset

            all_edge_index.append(np.stack([src_global, dst_global], axis=0))
            all_edge_attr.append(dists.astype(np.float32))
            all_edge_batch.append(np.full(n_edges, mol_idx, dtype=np.int64))

            # --- Target Mapping ---
            # Create edge lookup for coupling mapping
            edge_lookup = np.full((n_atoms, n_atoms), -1, dtype=np.int64)
            edge_lookup[src_local, dst_local] = np.arange(n_edges)
            if mol_name in meta_groups.groups:
                mol_meta = meta_groups.get_group(mol_name)

                # For each coupling, find the corresponding edge index
                # Coupling: (atom0, atom1). We need edge index for (atom0, atom1).
                # Note: The graph has both (0,1) and (1,0).
                # We can pick either, but must be consistent. Usually we pick both or just one.
                # Let's pick (atom0, atom1) as defined in CSV.

                a0 = mol_meta["atom_index_0"].values
                a1 = mol_meta["atom_index_1"].values

                # Lookup edge indices
                # We need to handle cases where edge might not exist (cutoff)
                # Using the dense lookup table
                target_edge_indices = []
                valid_mask = []

                for idx_pair in range(len(a0)):
                    u, v = a0[idx_pair], a1[idx_pair]
                    if u < n_atoms and v < n_atoms:
                        e_idx = edge_lookup[u, v]
                        if e_idx != -1:
                            target_edge_indices.append(e_idx + edge_offset)
                            valid_mask.append(True)
                        else:
                            # This coupling is outside cutoff radius
                            target_edge_indices.append(-1)
                            valid_mask.append(False)
                    else:
                        target_edge_indices.append(-1)
                        valid_mask.append(False)

                valid_mask = np.array(valid_mask, dtype=bool)

                if np.any(valid_mask):
                    # Filter valid targets
                    valid_meta = mol_meta[valid_mask]
                    valid_edge_indices = np.array(target_edge_indices)[valid_mask]

                    all_coupling_edge_idx.append(valid_edge_indices)
                    all_coupling_id.append(valid_meta["id"].values)

                    # Map Types
                    types = [self.type_map[t] for t in valid_meta["type"].values]
                    all_coupling_type.append(np.array(types, dtype=np.int64))

                    # Map Values (if train/val)
                    if "scalar_coupling_constant" in valid_meta.columns:
                        all_coupling_val.append(
                            valid_meta["scalar_coupling_constant"].values.astype(
                                np.float32
                            )
                        )

            # Update offsets
            node_offset += n_atoms
            edge_offset += n_edges

        # 7. Concatenate and Return
        print("Concatenating arrays...")
        data = {
            "node_x": np.concatenate(all_node_x),
            "node_pos": np.concatenate(all_node_pos),
            "node_batch": np.concatenate(all_node_batch),
            "edge_index": np.concatenate(all_edge_index, axis=1),
            "edge_attr": np.concatenate(all_edge_attr),
            "edge_batch": np.concatenate(all_edge_batch),
            "coupling_edge_index": (
                np.concatenate(all_coupling_edge_idx)
                if all_coupling_edge_idx
                else np.empty((0,), dtype=np.int64)
            ),
            "coupling_type": (
                np.concatenate(all_coupling_type)
                if all_coupling_type
                else np.empty((0,), dtype=np.int64)
            ),
            "coupling_id": (
                np.concatenate(all_coupling_id)
                if all_coupling_id
                else np.empty((0,), dtype=np.int64)
            ),
        }

        if all_coupling_val:
            data["coupling_value"] = np.concatenate(all_coupling_val)
        else:
            # For test set, create dummy values or omit
            data["coupling_value"] = np.zeros(
                len(data["coupling_id"]), dtype=np.float32
            )

        if split in ["train", "val"]:
            data["aux_shielding"] = np.concatenate(all_aux_shield)
            data["aux_charge"] = np.concatenate(all_aux_charge)
        else:
            # Dummy aux for test
            data["aux_shielding"] = np.zeros(len(data["node_x"]), dtype=np.float32)
            data["aux_charge"] = np.zeros(len(data["node_x"]), dtype=np.float32)

        return data

    def _save_cache(self, data, cache_dir):
        """Saves the data dictionary to numpy files."""
        for key, value in data.items():
            np.save(os.path.join(cache_dir, f"{key}.npy"), value)

        with open(os.path.join(cache_dir, "completed.flag"), "w") as f:
            f.write("done")

    def _load_cache(self, cache_dir):
        """Loads data from numpy files."""
        data = {}
        files = [f for f in os.listdir(cache_dir) if f.endswith(".npy")]
        for f in files:
            key = f.replace(".npy", "")
            data[key] = np.load(os.path.join(cache_dir, f))
        return data

    def _compute_and_save_stats(self, data):
        """
        Computes mean and std for targets and saves them.
        """
        print("Computing training statistics...")
        stats = {}

        # Primary Target: Per-Type Statistics
        vals = data["coupling_value"]
        types = data["coupling_type"]

        type_stats = {}
        for t_name, t_idx in self.type_map.items():
            mask = types == t_idx
            if np.any(mask):
                subset = vals[mask]
                type_stats[t_idx] = {
                    "mean": float(np.mean(subset)),
                    "std": float(np.std(subset)),
                }
            else:
                type_stats[t_idx] = {"mean": 0.0, "std": 1.0}

        stats["coupling_stats"] = type_stats

        # Aux Targets: Global Statistics
        if "aux_shielding" in data:
            stats["shielding_mean"] = float(np.mean(data["aux_shielding"]))
            stats["shielding_std"] = float(np.std(data["aux_shielding"]))

        if "aux_charge" in data:
            stats["charge_mean"] = float(np.mean(data["aux_charge"]))
            stats["charge_std"] = float(np.std(data["aux_charge"]))

        # Save to file
        stats_path = os.path.join(self.config.PROCESSED_DATA_DIR, "stats.npy")
        np.save(stats_path, stats)
        print(f"Statistics saved to {stats_path}")
