import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.spatial import cKDTree
from library.config import Config
from library.utils import (
    load_structures_table,
    map_atom_types,
    compute_bond_angle,
    TargetStandardizer,
)


class DualGraphBuilder:
    """
    Constructs dual graphs (Atom Graph + Line Graph) for molecular property prediction.
    Handles data loading, graph generation, target standardization, and caching.
    """

    def __init__(self):
        self.structures_df = None
        self.structures_grp = None
        self.dipole_df = None
        self.potential_df = None
        self.shielding_grp = None
        self.charges_grp = None
        self.standardizer = TargetStandardizer()

        # Mapping for coupling types
        self.type_map = {t: i for i, t in enumerate(Config.COUPLING_TYPES)}

    def _load_raw_data(self):
        """Loads raw CSV files into memory and groups them for fast access."""
        if self.structures_df is not None:
            return

        print("Loading raw data files...")

        # Structures
        self.structures_df = load_structures_table()
        self.structures_grp = self.structures_df.groupby("molecule_name")

        # Molecule-level properties
        if os.path.exists(Config.DIPOLE_MOMENTS_CSV):
            self.dipole_df = pd.read_csv(Config.DIPOLE_MOMENTS_CSV).set_index(
                "molecule_name"
            )

        if os.path.exists(Config.POTENTIAL_ENERGY_CSV):
            self.potential_df = pd.read_csv(Config.POTENTIAL_ENERGY_CSV).set_index(
                "molecule_name"
            )

        # Atom-level properties
        if os.path.exists(Config.MAGNETIC_SHIELDING_CSV):
            shielding_df = pd.read_csv(Config.MAGNETIC_SHIELDING_CSV)
            self.shielding_grp = shielding_df.groupby("molecule_name")

        if os.path.exists(Config.MULLIKEN_CHARGES_CSV):
            charges_df = pd.read_csv(Config.MULLIKEN_CHARGES_CSV)
            self.charges_grp = charges_df.groupby("molecule_name")

    def _get_molecule_data(self, molecule_name):
        """Retrieves all raw data for a single molecule."""
        # Structure
        struct = self.structures_grp.get_group(molecule_name)
        atoms = struct["atom"].values
        coords = struct[["x", "y", "z"]].values.astype(np.float32)

        # Aux: Dipole
        dipole = np.zeros(3, dtype=np.float32)
        if self.dipole_df is not None and molecule_name in self.dipole_df.index:
            dipole = self.dipole_df.loc[molecule_name, ["X", "Y", "Z"]].values.astype(
                np.float32
            )

        # Aux: Potential
        potential = np.zeros(1, dtype=np.float32)
        if self.potential_df is not None and molecule_name in self.potential_df.index:
            potential = self.potential_df.loc[
                molecule_name, ["potential_energy"]
            ].values.astype(np.float32)

        # Aux: Shielding (N, 9)
        shielding = np.zeros((len(atoms), 9), dtype=np.float32)
        if (
            self.shielding_grp is not None
            and molecule_name in self.shielding_grp.groups
        ):
            s_data = self.shielding_grp.get_group(molecule_name)
            # Ensure order matches atom_index
            s_data = s_data.sort_values("atom_index")
            shielding = s_data[
                ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
            ].values.astype(np.float32)

        # Aux: Charges (N, 1)
        charges = np.zeros((len(atoms), 1), dtype=np.float32)
        if self.charges_grp is not None and molecule_name in self.charges_grp.groups:
            c_data = self.charges_grp.get_group(molecule_name)
            c_data = c_data.sort_values("atom_index")
            charges = c_data[["mulliken_charge"]].values.astype(np.float32)

        return atoms, coords, dipole, potential, shielding, charges

    def _build_graph(self, atoms, coords):
        """
        Constructs Atom Graph and Line Graph.
        Returns:
            edge_index: (2, E)
            edge_attr: (E, 1) - Distances
            line_edge_index: (2, L) - Triplet connections
            line_edge_attr: (L, 1) - Cosine angles
        """
        num_atoms = len(atoms)

        # 1. Atom Graph Construction (Radius Graph)
        tree = cKDTree(coords)
        # Query pairs within cutoff
        # k=MAX_NEIGHBORS + 1 because the point itself is included
        dists, indices = tree.query(
            coords,
            k=min(num_atoms, Config.MAX_NEIGHBORS + 1),
            distance_upper_bound=Config.RADIUS_CUTOFF,
        )

        src_list = []
        dst_list = []
        dist_list = []

        for i in range(num_atoms):
            for j_idx, d in zip(indices[i], dists[i]):
                if j_idx == num_atoms:
                    continue  # Infinite distance placeholder
                if i == j_idx:
                    continue  # Self-loop

                src_list.append(i)
                dst_list.append(j_idx)
                dist_list.append(d)

        if len(src_list) == 0:
            # Fallback for isolated atoms (rare/impossible in valid molecules)
            return (
                np.zeros((2, 0), int),
                np.zeros((0, 1), np.float32),
                np.zeros((2, 0), int),
                np.zeros((0, 1), np.float32),
            )

        edge_index = np.array([src_list, dst_list], dtype=np.int64)
        edge_attr = np.array(dist_list, dtype=np.float32).reshape(-1, 1)

        # 2. Line Graph Construction (Triplets)
        # We need pairs of edges (j->i) and (j->k) sharing source j
        # This corresponds to angle <ijk centered at j

        # Sort edges by source to group neighbors
        sort_idx = np.argsort(edge_index[0])
        edge_index_sorted = edge_index[:, sort_idx]
        # We need to map sorted indices back to original edge indices for the Line Graph nodes
        # The Line Graph nodes are indices of edges in the Atom Graph.
        # Let's map: original_edge_idx -> sorted_edge_idx is not needed.
        # We need: for each atom j, which original edge indices start at j?

        # Group edges by source atom
        adj = [[] for _ in range(num_atoms)]
        for e_idx, (src, dst) in enumerate(zip(edge_index[0], edge_index[1])):
            adj[src].append((dst, e_idx))

        line_srcs = []
        line_dsts = []
        angles = []

        # Convert coords to tensor for angle computation
        coords_t = torch.tensor(coords)

        for j in range(num_atoms):
            neighbors = adj[j]  # List of (neighbor_atom_index, edge_index)
            if len(neighbors) < 2:
                continue

            # Create all pairs of neighbors (i, k) for center j
            # We treat (j->i, j->k) as an interaction
            # neighbors is a list of tuples

            # Extract neighbor indices and edge indices
            n_indices = [n[0] for n in neighbors]
            e_indices = [n[1] for n in neighbors]

            k = len(neighbors)
            # We want all permutations (i, k) where i != k
            # Because angle(i, j, k) is symmetric, but in directed graphs we usually keep both directions
            # or just one. For GNN message passing, usually we want flow between all connected bonds.

            # Vectorized angle calculation for this atom
            # Create meshgrid of indices
            idx_grid_1, idx_grid_2 = np.meshgrid(np.arange(k), np.arange(k))
            mask = idx_grid_1 != idx_grid_2

            idx_1 = idx_grid_1[mask]
            idx_2 = idx_grid_2[mask]

            # Get atom indices for i and k
            atom_i = [n_indices[x] for x in idx_1]
            atom_k = [n_indices[x] for x in idx_2]

            # Get edge indices (nodes in Line Graph)
            edge_1 = [e_indices[x] for x in idx_1]
            edge_2 = [e_indices[x] for x in idx_2]

            line_srcs.extend(edge_1)
            line_dsts.extend(edge_2)

            # Compute angles
            pos_i = coords_t[atom_i]
            pos_j = coords_t[j].unsqueeze(0).expand(len(atom_i), 3)
            pos_k = coords_t[atom_k]

            cos_theta = compute_bond_angle(pos_i, pos_j, pos_k)
            angles.append(cos_theta.numpy())

        if len(line_srcs) > 0:
            line_edge_index = np.array([line_srcs, line_dsts], dtype=np.int64)
            line_edge_attr = np.concatenate(angles).reshape(-1, 1).astype(np.float32)
        else:
            line_edge_index = np.zeros((2, 0), dtype=np.int64)
            line_edge_attr = np.zeros((0, 1), dtype=np.float32)

        return edge_index, edge_attr, line_edge_index, line_edge_attr

    def process_split(self, split_name, metadata_path, load_cached=True):
        """
        Main processing function for a dataset split.
        """
        cache_dir = os.path.join(Config.CACHE_DIR, split_name)
        os.makedirs(cache_dir, exist_ok=True)

        # Define cache file paths
        files = {
            "node_x": "node_x.npy",
            "node_batch": "node_batch.npy",
            "edge_index": "edge_index.npy",
            "edge_attr": "edge_attr.npy",
            "edge_batch": "edge_batch.npy",
            "line_edge_index": "line_edge_index.npy",
            "line_edge_attr": "line_edge_attr.npy",
            "target_index": "target_index.npy",
            "target_type": "target_type.npy",
            "target_val": "target_val.npy",
            "target_batch": "target_batch.npy",
            "aux_shielding": "aux_shielding.npy",
            "aux_charges": "aux_charges.npy",
            "mol_dipole": "mol_dipole.npy",
            "mol_potential": "mol_potential.npy",
            "meta_ids": "meta_ids.npy",  # Original IDs from csv
        }

        # Check cache
        if load_cached:
            all_exist = all(
                os.path.exists(os.path.join(cache_dir, f)) for f in files.values()
            )
            if all_exist:
                print(f"Loading {split_name} from cache...")
                data = {}
                for k, v in files.items():
                    data[k] = np.load(os.path.join(cache_dir, v))

                # Load standardizer stats if training
                if split_name == "train":
                    self.standardizer.load(Config.WORKING_DIR)

                return data

        # Load raw data
        self._load_raw_data()

        print(f"Processing {split_name} dataset...")
        df = pd.read_csv(metadata_path)

        # Debug Mode
        if Config.DEBUG:
            print(f"DEBUG MODE: Sampling {Config.DEBUG_SAMPLE_SIZE} rows")
            mols = df["molecule_name"].unique()
            if len(mols) > Config.DEBUG_SAMPLE_SIZE:
                # Sample molecules, not rows, to keep integrity
                sampled_mols = np.random.choice(
                    mols, Config.DEBUG_SAMPLE_SIZE // 10, replace=False
                )  # approx
                df = df[df["molecule_name"].isin(sampled_mols)].copy()

        # Fit standardizer if train
        if split_name == "train" and Config.NORMALIZE_TARGETS:
            print("Fitting target standardizer...")
            self.standardizer.fit(df)
            self.standardizer.save(Config.WORKING_DIR)

        # Group by molecule for efficient iteration
        # We need to preserve the list of targets for each molecule
        mol_groups = df.groupby("molecule_name")
        unique_mols = df["molecule_name"].unique()

        # Lists to accumulate data
        list_node_x = []
        list_node_batch = []

        list_edge_index = []
        list_edge_attr = []
        list_edge_batch = []

        list_line_edge_index = []
        list_line_edge_attr = []

        list_target_index = []
        list_target_type = []
        list_target_val = []
        list_target_batch = []
        list_meta_ids = []

        list_aux_shielding = []
        list_aux_charges = []
        list_mol_dipole = []
        list_mol_potential = []

        # Counters for offset handling
        node_offset = 0
        edge_offset = 0

        for mol_idx, mol_name in enumerate(tqdm(unique_mols)):
            # 1. Get Molecule Data
            atoms, coords, dipole, potential, shielding, charges = (
                self._get_molecule_data(mol_name)
            )

            # 2. Build Graph
            edge_index, edge_attr, line_edge_index, line_edge_attr = self._build_graph(
                atoms, coords
            )

            num_nodes = len(atoms)
            num_edges = edge_index.shape[1]

            # 3. Get Targets for this molecule
            mol_targets = mol_groups.get_group(mol_name)

            # Map atom indices
            target_atom_0 = mol_targets["atom_index_0"].values
            target_atom_1 = mol_targets["atom_index_1"].values
            target_types_str = mol_targets["type"].values
            target_ids = mol_targets["id"].values

            # Encode types
            target_types = np.array(
                [self.type_map[t] for t in target_types_str], dtype=np.int64
            )

            # Target values (if exist)
            if "scalar_coupling_constant" in mol_targets.columns:
                raw_vals = mol_targets["scalar_coupling_constant"].values
                # Standardize if needed (and if we are not in test mode, though test won't have the col)
                # Note: We manually standardize using the fitted standardizer
                # But here we just store raw/transformed.
                # Let's use the standardizer's transform method which expects a DF
                if Config.NORMALIZE_TARGETS and split_name != "test":
                    # Create a mini df to use the transform method
                    mini_df = pd.DataFrame(
                        {"type": target_types_str, "scalar_coupling_constant": raw_vals}
                    )
                    target_vals = self.standardizer.transform(mini_df).astype(
                        np.float32
                    )
                else:
                    target_vals = raw_vals.astype(np.float32)
            else:
                target_vals = np.zeros(len(target_types), dtype=np.float32)

            # 4. Append to lists

            # Nodes
            list_node_x.append(map_atom_types(atoms))
            list_node_batch.append(np.full(num_nodes, mol_idx, dtype=np.int64))
            list_aux_shielding.append(shielding)
            list_aux_charges.append(charges)

            # Edges (Atom Graph)
            # Add offset to edge indices
            list_edge_index.append(edge_index + node_offset)
            list_edge_attr.append(edge_attr)
            list_edge_batch.append(np.full(num_edges, mol_idx, dtype=np.int64))

            # Edges (Line Graph)
            # Add offset to line edge indices (which point to edges)
            list_line_edge_index.append(line_edge_index + edge_offset)
            list_line_edge_attr.append(line_edge_attr)

            # Targets
            # Target indices point to nodes, so add node_offset
            t_idx = np.stack([target_atom_0, target_atom_1], axis=0) + node_offset
            list_target_index.append(t_idx)
            list_target_type.append(target_types)
            list_target_val.append(target_vals)
            list_target_batch.append(
                np.full(len(target_types), mol_idx, dtype=np.int64)
            )
            list_meta_ids.append(target_ids)

            # Global
            list_mol_dipole.append(dipole)
            list_mol_potential.append(potential)

            # Update offsets
            node_offset += num_nodes
            edge_offset += num_edges

        print("Concatenating data...")
        data = {
            "node_x": np.concatenate(list_node_x),
            "node_batch": np.concatenate(list_node_batch),
            "edge_index": np.concatenate(list_edge_index, axis=1),
            "edge_attr": np.concatenate(list_edge_attr),
            "edge_batch": np.concatenate(list_edge_batch),
            "line_edge_index": np.concatenate(list_line_edge_index, axis=1),
            "line_edge_attr": np.concatenate(list_line_edge_attr),
            "target_index": np.concatenate(list_target_index, axis=1),
            "target_type": np.concatenate(list_target_type),
            "target_val": np.concatenate(list_target_val),
            "target_batch": np.concatenate(list_target_batch),
            "aux_shielding": np.concatenate(list_aux_shielding),
            "aux_charges": np.concatenate(list_aux_charges),
            "mol_dipole": np.stack(list_mol_dipole),
            "mol_potential": np.stack(list_mol_potential),
            "meta_ids": np.concatenate(list_meta_ids),
        }

        print(f"Saving {split_name} to cache...")
        for k, v in data.items():
            np.save(os.path.join(cache_dir, files[k]), v)

        return data
