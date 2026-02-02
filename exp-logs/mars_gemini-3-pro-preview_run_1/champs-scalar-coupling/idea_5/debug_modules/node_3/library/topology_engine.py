import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from library import config, utils

# Disable chained assignment warning
pd.options.mode.chained_assignment = None


class TopologyEngine:
    """
    Engine for Adaptive Graph Construction and Hierarchical Feature Engineering.
    Implements the Multi-Hop Topological Stratified Ensemble feature generation.
    """

    def __init__(self, structures_df, verbose=True):
        """
        Args:
            structures_df (pd.DataFrame): The dataframe containing molecular structures.
            verbose (bool): Whether to print progress.
        """
        self.structures_df = structures_df
        self.verbose = verbose

        # Pre-process structures for fast access
        # Group by molecule_name for efficient retrieval
        if self.verbose:
            print("Grouping structures by molecule...")
        self.mol_groups = self.structures_df.groupby("molecule_name")

        # Mappings for atom types
        # Ensure deterministic order
        self.atom_types = sorted(list(config.ATOMIC_NUMBERS.keys()))
        self.atom_map = {a: i for i, a in enumerate(self.atom_types)}
        self.rev_atom_map = {i: a for a, i in self.atom_map.items()}
        self.num_atom_types = len(self.atom_types)

    def _get_molecule_data(self, molecule_name):
        """Retrieves coordinates and atom types for a molecule."""
        try:
            group = self.mol_groups.get_group(molecule_name)
        except KeyError:
            raise ValueError(f"Molecule {molecule_name} not found in structures data.")

        atoms = group["atom"].values
        coords = group[["x", "y", "z"]].values
        atom_indices = group["atom_index"].values
        return atoms, coords, atom_indices

    def _compute_molecule_features(self, molecule_name):
        """
        Computes all node and pair features for a single molecule.

        Returns:
            node_feats (dict): Map of atom_index -> dict of features
            pair_feats (dict): Map of (atom_i, atom_j) -> dict of features
        """
        atoms, coords, indices = self._get_molecule_data(molecule_name)
        n_atoms = len(atoms)

        # ---------------------------------------------------------
        # 1. Adaptive Adjacency Construction
        # ---------------------------------------------------------
        # Calculate pairwise distances: Shape (N, N)
        # Using broadcasting: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
        diffs = coords[:, None, :] - coords[None, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))

        # Get covalent radii for all atoms
        radii = np.array([config.COVALENT_RADII.get(a, 0.7) for a in atoms])

        # Connectivity threshold: r_i + r_j + tolerance
        # Shape: (N, N)
        thresholds = radii[:, None] + radii[None, :] + config.BOND_TOLERANCE

        # Adjacency Matrix (exclude self-loops)
        # Connected if distance is within sum of radii + tolerance
        adj_matrix = (dists <= thresholds) & (dists > 0)

        # ---------------------------------------------------------
        # 2. Node Features (Level 0, 1, 2)
        # ---------------------------------------------------------

        # Level 0: Atomic Properties
        l0_feats = []
        for i in range(n_atoms):
            atom = atoms[i]
            feat = {
                "en": config.ATOM_ELECTRONEGATIVITY.get(atom, 2.5),
                "rad": config.COVALENT_RADII.get(atom, 0.7),
                "n_bonds": np.sum(adj_matrix[i]),
            }
            l0_feats.append(feat)

        # Prepare for Matrix Multiplication
        # One-hot encode atoms: (N, num_types)
        atom_one_hot = np.zeros((n_atoms, self.num_atom_types))
        for i, atom in enumerate(atoms):
            if atom in self.atom_map:
                atom_one_hot[i, self.atom_map[atom]] = 1

        # Level 1: 1-Hop Aggregation (Bag of Neighbors)
        # L1 Counts: Adj * OneHot -> (N, num_types)
        l1_counts = adj_matrix.astype(float) @ atom_one_hot

        # Level 2: 2-Hop Aggregation (Bag of Neighbors' Neighbors)
        # L2 Counts: Adj * L1_Counts -> (N, num_types)
        l2_counts = adj_matrix.astype(float) @ l1_counts

        # Compile Node Features
        node_features = {}
        for i in range(n_atoms):
            idx = indices[i]
            f = l0_feats[i].copy()

            # Add L1 counts
            for t_idx in range(self.num_atom_types):
                atom_type = self.rev_atom_map[t_idx]
                f[f"L1_{atom_type}"] = l1_counts[i, t_idx]

            # Add L2 counts
            for t_idx in range(self.num_atom_types):
                atom_type = self.rev_atom_map[t_idx]
                f[f"L2_{atom_type}"] = l2_counts[i, t_idx]

            node_features[idx] = f

        # ---------------------------------------------------------
        # 3. Pair Features (Geometry & Field Projections)
        # ---------------------------------------------------------

        pair_features = {}

        # Pre-compute normalized bond vectors for all atoms
        # bond_vectors[i] is a matrix of shape (num_neighbors, 3)
        bond_vectors = []
        for i in range(n_atoms):
            neighbors = np.where(adj_matrix[i])[0]
            if len(neighbors) > 0:
                vecs = coords[neighbors] - coords[i]  # Vectors from i to neighbors
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0  # Safety
                normalized_vecs = vecs / norms
                bond_vectors.append(normalized_vecs)
            else:
                bond_vectors.append(np.empty((0, 3)))

        # Iterate over all pairs (i, j)
        # Note: We compute for all pairs to be safe and simple.
        # Given N is small (<30), N*N is <900, which is very fast.
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    continue

                idx_i = indices[i]
                idx_j = indices[j]

                # Distance features
                d = dists[i, j]
                # Avoid division by zero if atoms overlap (unlikely)
                d_safe = d if d > 1e-6 else 1e-6

                p_feats = {
                    "dist": d,
                    "dist_inv": 1.0 / d_safe,
                    "dist_inv2": 1.0 / (d_safe**2),
                    "dist_inv3": 1.0 / (d_safe**3),
                }

                # Vector u_ij = P_j - P_i (Direction from i to j)
                u_ij = coords[j] - coords[i]
                u_norm = np.linalg.norm(u_ij)
                u_hat = u_ij / u_norm if u_norm > 1e-9 else u_ij

                # Field Projection for Atom i
                # Project neighbors of i onto the axis i->j
                # cos_theta = v_ik . u_ij
                b_vecs_i = bond_vectors[i]
                if b_vecs_i.shape[0] > 0:
                    projections_i = b_vecs_i @ u_hat  # (num_neighbors, )
                    p_feats["fp_0_mean"] = np.mean(projections_i)
                    p_feats["fp_0_min"] = np.min(projections_i)
                    p_feats["fp_0_max"] = np.max(projections_i)
                else:
                    p_feats["fp_0_mean"] = 0.0
                    p_feats["fp_0_min"] = 0.0
                    p_feats["fp_0_max"] = 0.0

                # Field Projection for Atom j
                # Project neighbors of j onto the axis j->i (which is -u_hat)
                b_vecs_j = bond_vectors[j]
                if b_vecs_j.shape[0] > 0:
                    projections_j = b_vecs_j @ (
                        -u_hat
                    )  # Dot product with vector pointing back to i
                    p_feats["fp_1_mean"] = np.mean(projections_j)
                    p_feats["fp_1_min"] = np.min(projections_j)
                    p_feats["fp_1_max"] = np.max(projections_j)
                else:
                    p_feats["fp_1_mean"] = 0.0
                    p_feats["fp_1_min"] = 0.0
                    p_feats["fp_1_max"] = 0.0

                pair_features[(idx_i, idx_j)] = p_feats

        return node_features, pair_features

    def generate_features(self, metadata_df, load_cached_data=True, split_name="train"):
        """
        Generates features for the provided metadata dataframe.

        Args:
            metadata_df (pd.DataFrame): The dataframe containing pairs to process.
            load_cached_data (bool): Whether to load from cache.
            split_name (str): Name for the cache file (e.g., 'train', 'test').

        Returns:
            pd.DataFrame: The metadata dataframe enriched with features.
        """
        # Ensure working directory exists
        os.makedirs(config.WORKING_DIR, exist_ok=True)

        cache_path = os.path.join(config.WORKING_DIR, f"features_{split_name}.parquet")

        # 1. Attempt Cache Load
        if load_cached_data and os.path.exists(cache_path):
            if self.verbose:
                print(f"Loading features from cache: {cache_path}")
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        if self.verbose:
            print(f"Generating features for {split_name}...")

        # 2. Processing
        # Sort metadata by molecule to optimize group retrieval
        df_sorted = metadata_df.sort_values("molecule_name").copy()
        meta_groups = df_sorted.groupby("molecule_name")
        unique_molecules = df_sorted["molecule_name"].unique()

        # Cite debug_lesson_4: Intersect Data Indices Instead of Padding with Defaults
        # Filter to ensure we only process molecules that exist in the structures data
        available_molecules = set(self.mol_groups.groups.keys())
        valid_molecules = [m for m in unique_molecules if m in available_molecules]

        if self.verbose and len(valid_molecules) < len(unique_molecules):
            print(
                f"Skipping {len(unique_molecules) - len(valid_molecules)} molecules missing from structures."
            )

        results = []

        # Iterator
        iterator = (
            tqdm(valid_molecules, desc=f"Processing {split_name}")
            if self.verbose
            else valid_molecules
        )

        for mol_name in iterator:
            # Compute features for the molecule
            node_feats, pair_feats = self._compute_molecule_features(mol_name)

            # Get relevant rows
            if mol_name not in meta_groups.groups:
                continue

            group = meta_groups.get_group(mol_name)

            # Map features to rows
            # Using to_dict('records') is faster than iterating rows manually
            group_records = group.to_dict("records")

            for row in group_records:
                idx0 = row["atom_index_0"]
                idx1 = row["atom_index_1"]

                # Enrich with Node 0 Features
                if idx0 in node_feats:
                    for k, v in node_feats[idx0].items():
                        row[f"a0_{k}"] = v

                # Enrich with Node 1 Features
                if idx1 in node_feats:
                    for k, v in node_feats[idx1].items():
                        row[f"a1_{k}"] = v

                # Enrich with Pair Features
                if (idx0, idx1) in pair_feats:
                    for k, v in pair_feats[(idx0, idx1)].items():
                        row[k] = v

                results.append(row)

        # 3. DataFrame Construction
        if self.verbose:
            print("Constructing final DataFrame...")

        if not results:
            # Fallback: Preserve schema if all data was filtered
            final_df = pd.DataFrame(columns=metadata_df.columns)
        else:
            final_df = pd.DataFrame(results)

        # Optimize memory
        final_df = utils.reduce_mem_usage(final_df, verbose=self.verbose)

        # 4. Save to Cache
        if self.verbose:
            print(f"Saving features to cache: {cache_path}")
        final_df.to_parquet(cache_path, index=False)

        return final_df
