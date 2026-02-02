import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library import geometry


class MoleculeDataset(Dataset):
    def __init__(self, metadata_path, mode="train", load_cached=True, debug_size=None):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            load_cached (bool): Whether to try loading from cache.
            debug_size (int, optional): Limit dataset size for debugging.
        """
        self.mode = mode
        self.metadata_path = metadata_path
        self.debug_size = debug_size

        # 1. Load Metadata
        print(f"[{mode.upper()}] Loading metadata from {metadata_path}...")
        self.df = pd.read_csv(metadata_path)

        if self.debug_size is not None:
            print(
                f"[{mode.upper()}] Debug mode: limiting to {self.debug_size} samples."
            )
            self.df = self.df.iloc[: self.debug_size].reset_index(drop=True)

        # Identify unique molecules in this dataset
        self.unique_mols = self.df["molecule_name"].unique()
        print(f"[{mode.upper()}] Found {len(self.unique_mols)} unique molecules.")

        # 2. Compute Normalization Stats (always from training data)
        self.stats = self._compute_stats()

        # 3. Load or Compute Graph Data
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        cache_filename = f"cached_{mode}_v3.npz"
        if self.debug_size:
            cache_filename = f"cached_{mode}_debug_{self.debug_size}_v3.npz"

        self.cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

        if load_cached and os.path.exists(self.cache_path):
            print(
                f"[{mode.upper()}] Loading cached graph data from {self.cache_path}..."
            )
            self._load_cache()
        else:
            print(f"[{mode.upper()}] Computing graph data from scratch...")
            self._process_and_cache()

        # Pre-map coupling types to integers for faster getitem
        self.df["type_idx"] = self.df["type"].map(Config.COUPLING_TYPE_MAP)

    def _compute_stats(self):
        """Computes mean and std for targets from the training set."""
        # We always load the full train metadata to get correct stats
        # even if we are in debug mode or processing val/test
        if not os.path.exists(Config.TRAIN_METADATA_PATH):
            # Fallback if train metadata missing
            return {t: {"mean": 0.0, "std": 1.0} for t in Config.COUPLING_TYPES}

        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        stats = {}
        for t in Config.COUPLING_TYPES:
            subset = train_df[train_df["type"] == t]
            if len(subset) > 0:
                vals = subset["scalar_coupling_constant"].values
                stats[t] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            else:
                stats[t] = {"mean": 0.0, "std": 1.0}
        return stats

    def _process_and_cache(self):
        # Load all structures
        print(
            f"[{self.mode.upper()}] Reading structures from {Config.STRUCTURES_PATH}..."
        )
        df_struct = pd.read_csv(Config.STRUCTURES_PATH)

        # Filter to relevant molecules
        relevant_mols = set(self.unique_mols)
        df_struct = df_struct[df_struct["molecule_name"].isin(relevant_mols)]

        # Sort by molecule and atom index to ensure contiguous blocks
        df_struct = df_struct.sort_values(["molecule_name", "atom_index"])

        # Convert to numpy for fast processing
        all_mol_names = df_struct["molecule_name"].values
        all_atom_symbols = df_struct["atom"].values
        all_coords = df_struct[["x", "y", "z"]].values.astype(np.float32)

        # Identify slices for each molecule
        unique_mols_struct, start_indices, counts = np.unique(
            all_mol_names, return_index=True, return_counts=True
        )

        # Create a map from molecule_name to (start, count)
        mol_slice_map = {
            m: (s, c) for m, s, c in zip(unique_mols_struct, start_indices, counts)
        }

        # Arrays to store aggregated data
        data_atom_types = []
        data_coords = []
        data_edge_indices = []
        data_edge_dists = []
        data_triplets = []
        data_triplet_angles = []
        data_triplet_edge_indices = (
            []
        )  # Maps triplet to (incoming_edge_idx, outgoing_edge_idx)

        # Meta-index to retrieve data for a specific molecule
        mol_index_map = {}

        current_atom_offset = 0
        current_edge_offset = 0
        current_triplet_offset = 0

        total_mols = len(self.unique_mols)
        print_interval = max(1, total_mols // 10)

        for i, mol_name in enumerate(self.unique_mols):
            if i % print_interval == 0:
                print(f"[{self.mode.upper()}] Processing molecule {i}/{total_mols}...")

            if mol_name not in mol_slice_map:
                continue

            start, count = mol_slice_map[mol_name]

            # 1. Node Features
            mol_atoms = all_atom_symbols[start : start + count]
            mol_coords = all_coords[start : start + count]

            # Map atoms to integers
            mol_atom_ids = np.array(
                [Config.ATOM_MAP[a] for a in mol_atoms], dtype=np.int64
            )

            # 2. Edges
            edge_index = geometry.get_neighbors(mol_coords)
            edge_dists = geometry.compute_distances(mol_coords, edge_index)

            # 3. Triplets
            triplets, e1_idx, e2_idx = geometry.get_triplets(edge_index, count)
            triplet_angles = geometry.compute_angles(mol_coords, triplets)

            num_atoms = len(mol_atom_ids)
            num_edges = edge_index.shape[1]
            num_triplets = triplets.shape[1]

            # Store Data
            data_atom_types.append(mol_atom_ids)
            data_coords.append(mol_coords)
            data_edge_indices.append(edge_index)
            data_edge_dists.append(edge_dists)
            data_triplets.append(triplets)
            data_triplet_angles.append(triplet_angles)

            # Store triplet-to-edge mapping (stack e1 and e2)
            if num_triplets > 0:
                t_edge_map = np.vstack([e1_idx, e2_idx])
            else:
                t_edge_map = np.empty((2, 0), dtype=np.int64)
            data_triplet_edge_indices.append(t_edge_map)

            # Record Slices
            mol_index_map[mol_name] = {
                "atom_start": current_atom_offset,
                "atom_count": num_atoms,
                "edge_start": current_edge_offset,
                "edge_count": num_edges,
                "triplet_start": current_triplet_offset,
                "triplet_count": num_triplets,
            }

            current_atom_offset += num_atoms
            current_edge_offset += num_edges
            current_triplet_offset += num_triplets

        # Concatenate all
        if len(data_atom_types) > 0:
            print(f"[{self.mode.upper()}] Concatenating arrays...")
            self.arr_atom_types = np.concatenate(data_atom_types)
            self.arr_coords = np.concatenate(data_coords)
            self.arr_edge_indices = np.concatenate(
                data_edge_indices, axis=1
            )  # (2, Total_Edges)
            self.arr_edge_dists = np.concatenate(data_edge_dists)
            self.arr_triplets = np.concatenate(
                data_triplets, axis=1
            )  # (3, Total_Triplets)
            self.arr_triplet_angles = np.concatenate(data_triplet_angles)
            self.arr_triplet_edge_indices = np.concatenate(
                data_triplet_edge_indices, axis=1
            )  # (2, Total_Triplets)
        else:
            print(
                f"[{self.mode.upper()}] WARNING: No valid molecules found. Creating empty dataset."
            )
            self.arr_atom_types = np.empty((0,), dtype=np.int64)
            self.arr_coords = np.empty((0, 3), dtype=np.float32)
            self.arr_edge_indices = np.empty((2, 0), dtype=np.int64)
            self.arr_edge_dists = np.empty((0,), dtype=np.float32)
            self.arr_triplets = np.empty((3, 0), dtype=np.int64)
            self.arr_triplet_angles = np.empty((0,), dtype=np.float32)
            self.arr_triplet_edge_indices = np.empty((2, 0), dtype=np.int64)

        # Save map (decompose dict to arrays for npz)
        map_keys = np.array(list(mol_index_map.keys()))
        if len(mol_index_map) > 0:
            map_vals = np.array(
                [
                    [
                        v["atom_start"],
                        v["atom_count"],
                        v["edge_start"],
                        v["edge_count"],
                        v["triplet_start"],
                        v["triplet_count"],
                    ]
                    for v in mol_index_map.values()
                ],
                dtype=np.int64,
            )
        else:
            map_vals = np.empty((0, 6), dtype=np.int64)

        self.mol_map_keys = map_keys
        self.mol_map_vals = map_vals

        # Save to disk
        print(f"[{self.mode.upper()}] Saving to {self.cache_path}...")
        np.savez_compressed(
            self.cache_path,
            atom_types=self.arr_atom_types,
            coords=self.arr_coords,
            edge_indices=self.arr_edge_indices,
            edge_dists=self.arr_edge_dists,
            triplets=self.arr_triplets,
            triplet_angles=self.arr_triplet_angles,
            triplet_edge_indices=self.arr_triplet_edge_indices,
            map_keys=map_keys,
            map_vals=map_vals,
        )

        self.mol_index_map = mol_index_map

        # Filter metadata to only include successfully processed molecules
        if len(mol_index_map) < len(self.unique_mols):
            print(
                f"[{self.mode.upper()}] Filtering metadata: keeping {len(mol_index_map)}/{len(self.unique_mols)} molecules."
            )
            valid_mols = set(mol_index_map.keys())
            self.df = self.df[self.df["molecule_name"].isin(valid_mols)].reset_index(
                drop=True
            )

    def _load_cache(self):
        data = np.load(self.cache_path)
        self.arr_atom_types = data["atom_types"]
        self.arr_coords = data["coords"]
        self.arr_edge_indices = data["edge_indices"]
        self.arr_edge_dists = data["edge_dists"]
        self.arr_triplets = data["triplets"]
        self.arr_triplet_angles = data["triplet_angles"]
        self.arr_triplet_edge_indices = data["triplet_edge_indices"]

        map_keys = data["map_keys"]
        map_vals = data["map_vals"]

        self.mol_index_map = {}
        for k, v in zip(map_keys, map_vals):
            self.mol_index_map[k] = {
                "atom_start": v[0],
                "atom_count": v[1],
                "edge_start": v[2],
                "edge_count": v[3],
                "triplet_start": v[4],
                "triplet_count": v[5],
            }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mol_name = row["molecule_name"]

        # Retrieve Graph Data
        meta = self.mol_index_map[mol_name]

        # Atoms
        a_s, a_c = meta["atom_start"], meta["atom_count"]
        atom_types = self.arr_atom_types[a_s : a_s + a_c]
        coords = self.arr_coords[a_s : a_s + a_c]

        # Edges
        e_s, e_c = meta["edge_start"], meta["edge_count"]
        edge_index = self.arr_edge_indices[:, e_s : e_s + e_c]
        edge_dists = self.arr_edge_dists[e_s : e_s + e_c]

        # Triplets
        t_s, t_c = meta["triplet_start"], meta["triplet_count"]
        triplet_angles = self.arr_triplet_angles[t_s : t_s + t_c]
        triplet_edge_index = self.arr_triplet_edge_indices[:, t_s : t_s + t_c]

        # Target Info
        atom_0 = int(row["atom_index_0"])
        atom_1 = int(row["atom_index_1"])
        type_idx = int(row["type_idx"])

        # Normalize Target
        if "scalar_coupling_constant" in row:
            raw_target = float(row["scalar_coupling_constant"])
            coupling_type = Config.INVERSE_COUPLING_TYPE_MAP[type_idx]
            mean = self.stats[coupling_type]["mean"]
            std = self.stats[coupling_type]["std"]
            target = (raw_target - mean) / std
        else:
            target = 0.0  # Test set

        # Identify the edge index corresponding to the target pair
        # Find where source == atom_0 AND dest == atom_1
        mask_fwd = (edge_index[0] == atom_0) & (edge_index[1] == atom_1)
        mask_bwd = (edge_index[0] == atom_1) & (edge_index[1] == atom_0)

        idx_fwd = np.where(mask_fwd)[0]
        idx_bwd = np.where(mask_bwd)[0]

        target_edge_idx_0 = idx_fwd[0] if len(idx_fwd) > 0 else -1
        target_edge_idx_1 = idx_bwd[0] if len(idx_bwd) > 0 else -1

        return {
            "atom_types": torch.tensor(atom_types, dtype=torch.long),
            "coords": torch.tensor(coords, dtype=torch.float32),
            "edge_index": torch.tensor(edge_index, dtype=torch.long),
            "edge_dists": torch.tensor(edge_dists, dtype=torch.float32),
            "triplet_angles": torch.tensor(triplet_angles, dtype=torch.float32),
            "triplet_edge_index": torch.tensor(triplet_edge_index, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float32),
            "type_idx": torch.tensor(type_idx, dtype=torch.long),
            "target_edge_indices": torch.tensor(
                [target_edge_idx_0, target_edge_idx_1], dtype=torch.long
            ),
            "id": row.get("id", -1),
        }


def collate_dmpnn(batch):
    """
    Custom collate function to batch graphs.
    """
    # Initialize lists
    atom_types_list = []
    coords_list = []
    edge_index_list = []
    edge_dists_list = []
    triplet_angles_list = []
    triplet_edge_index_list = []
    targets_list = []
    type_idxs_list = []
    target_edge_indices_list = []
    ids_list = []

    # Counters for offsets
    num_atoms_cum = 0
    num_edges_cum = 0

    for item in batch:
        # Nodes
        num_atoms = item["atom_types"].shape[0]
        atom_types_list.append(item["atom_types"])
        coords_list.append(item["coords"])

        # Edges (adjust indices by atom offset)
        num_edges = item["edge_index"].shape[1]
        edge_index_list.append(item["edge_index"] + num_atoms_cum)
        edge_dists_list.append(item["edge_dists"])

        # Triplets (adjust indices by edge offset)
        # triplet_edge_index refers to edges, so we add edge offset
        triplet_edge_index_list.append(item["triplet_edge_index"] + num_edges_cum)
        triplet_angles_list.append(item["triplet_angles"])

        # Target Edges (adjust by edge offset)
        # Handle -1 (missing edge) carefully
        tei = item["target_edge_indices"].clone()
        mask = tei >= 0
        tei[mask] += num_edges_cum
        target_edge_indices_list.append(tei)

        # Scalars
        targets_list.append(item["target"])
        type_idxs_list.append(item["type_idx"])
        ids_list.append(item["id"])

        # Update offsets
        num_atoms_cum += num_atoms
        num_edges_cum += num_edges

    # Concatenate
    batch_data = {
        "atom_types": torch.cat(atom_types_list, dim=0),
        "coords": torch.cat(coords_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "edge_dists": torch.cat(edge_dists_list, dim=0),
        "triplet_angles": torch.cat(triplet_angles_list, dim=0),
        "triplet_edge_index": torch.cat(triplet_edge_index_list, dim=1),
        "targets": torch.stack(targets_list),
        "type_idxs": torch.stack(type_idxs_list),
        "target_edge_indices": torch.stack(target_edge_indices_list),
        "ids": ids_list,
        "batch_num_nodes": num_atoms_cum,
        "batch_num_edges": num_edges_cum,
    }

    return batch_data
