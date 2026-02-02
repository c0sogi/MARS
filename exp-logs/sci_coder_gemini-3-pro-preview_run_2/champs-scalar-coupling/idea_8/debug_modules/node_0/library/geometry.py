import os
import numpy as np
import pandas as pd
import torch
from library.config import Config


class Geometry:
    """
    Handles low-level geometric computations for molecular graphs.
    """

    # Mapping for atom types (Atomic Numbers could also be used, but 0-based index is better for embeddings)
    ATOM_MAP = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}

    # Mapping for coupling types
    COUPLING_TYPE_MAP = {
        "1JHC": 0,
        "2JHC": 1,
        "3JHC": 2,
        "1JHN": 3,
        "2JHN": 4,
        "3JHN": 5,
        "2JHH": 6,
        "3JHH": 7,
    }

    @staticmethod
    def load_structure(xyz_path):
        """
        Parses an XYZ file to extract atoms and coordinates.

        Args:
            xyz_path (str): Relative path to the .xyz file.

        Returns:
            atoms (np.ndarray): Integer array of atom types.
            coords (np.ndarray): Float array of shape (N, 3) containing coordinates.
        """
        full_path = os.path.join(Config.INPUT_DIR, xyz_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Structure file not found: {full_path}")

        with open(full_path, "r") as f:
            lines = f.readlines()

        # First line is number of atoms
        try:
            num_atoms = int(lines[0].strip())
        except ValueError:
            raise ValueError(f"Invalid XYZ format in {xyz_path}")

        atoms = []
        coords = []

        # Skip number of atoms and comment line
        for line in lines[2:]:
            parts = line.split()
            if not parts:
                continue

            atom_symbol = parts[0]
            # Handle potential scientific notation or extra precision in coords
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except (ValueError, IndexError):
                continue

            if atom_symbol in Geometry.ATOM_MAP:
                atoms.append(Geometry.ATOM_MAP[atom_symbol])
            else:
                # Fallback for unknown atoms, though not expected in this dataset
                atoms.append(len(Geometry.ATOM_MAP))

            coords.append([x, y, z])

        return np.array(atoms, dtype=np.int64), np.array(coords, dtype=np.float32)

    @staticmethod
    def build_neighbor_graph(
        coords, cutoff=Config.SPATIAL_CUTOFF, max_neighbors=Config.MAX_NEIGHBORS
    ):
        """
        Constructs a directed graph connecting atoms within a spatial cutoff.

        Args:
            coords (np.ndarray): (N, 3) array of coordinates.
            cutoff (float): Distance threshold in Angstroms.
            max_neighbors (int): Maximum number of neighbors per node.

        Returns:
            edge_index (np.ndarray): (2, E) array of source and target indices.
            edge_vec (np.ndarray): (E, 3) array of edge vectors (target - source).
            edge_dist (np.ndarray): (E,) array of Euclidean distances.
        """
        num_atoms = len(coords)

        # Compute pairwise distance matrix
        # r_i: (N, 1, 3), r_j: (1, N, 3)
        r_i = coords[:, np.newaxis, :]
        r_j = coords[np.newaxis, :, :]

        # Vector from i to j
        vec_ij = r_j - r_i
        dist_ij = np.linalg.norm(vec_ij, axis=-1)

        # Create mask: distance < cutoff AND distance > 0 (exclude self-loops)
        mask = (dist_ij < cutoff) & (dist_ij > 1e-6)

        # Get indices
        # We iterate to enforce max_neighbors
        src_list = []
        dst_list = []

        for i in range(num_atoms):
            # Find neighbors for atom i
            dists = dist_ij[i]
            # Get indices of valid neighbors
            neighbors = np.where(mask[i])[0]

            if len(neighbors) > max_neighbors:
                # Sort by distance and take closest
                neighbor_dists = dists[neighbors]
                sorted_idx = np.argsort(neighbor_dists)
                neighbors = neighbors[sorted_idx[:max_neighbors]]

            for n in neighbors:
                src_list.append(i)
                dst_list.append(n)

        if not src_list:
            return (
                np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        edge_index = np.array([src_list, dst_list], dtype=np.int64)

        # Re-extract vectors and distances for the selected edges
        edge_vec = coords[edge_index[1]] - coords[edge_index[0]]
        edge_dist = np.linalg.norm(edge_vec, axis=1)

        return edge_index, edge_vec.astype(np.float32), edge_dist.astype(np.float32)

    @staticmethod
    def get_triplets(edge_index, num_atoms):
        """
        Finds triplets k -> j -> i for angular computation.

        Args:
            edge_index (np.ndarray): (2, E) array.
            num_atoms (int): Number of atoms.

        Returns:
            triplet_edges (np.ndarray): (2, T) array containing indices of edges [edge_kj, edge_ji].
        """
        src, dst = edge_index

        # Build adjacency list for incoming edges: dst -> list of (src, edge_idx)
        incoming = [[] for _ in range(num_atoms)]
        for idx, (s, d) in enumerate(zip(src, dst)):
            incoming[d].append((s, idx))

        indices_kj = []
        indices_ji = []

        # Iterate over all edges j->i (this is the second leg of the triplet)
        for idx_ji, (j, i) in enumerate(zip(src, dst)):
            # Find all k such that k->j exists
            for k, idx_kj in incoming[j]:
                if k != i:  # Avoid backtracking k == i
                    indices_kj.append(idx_kj)
                    indices_ji.append(idx_ji)

        if not indices_kj:
            return np.zeros((2, 0), dtype=np.int64)

        return np.array([indices_kj, indices_ji], dtype=np.int64)

    @staticmethod
    def compute_basis_expansions(edge_dist, edge_vec, edge_index, num_atoms):
        """
        Computes RBF features for edges and SBF (Legendre) features for triplets.

        Args:
            edge_dist (np.ndarray): (E,) distances.
            edge_vec (np.ndarray): (E, 3) vectors.
            edge_index (np.ndarray): (2, E) indices.
            num_atoms (int): Number of atoms.

        Returns:
            rbf (np.ndarray): (E, NUM_RBF) Radial Basis features.
            sbf (np.ndarray): (T, NUM_SBF) Spherical Basis features.
            triplet_indices (np.ndarray): (2, T) indices of edges forming triplets.
        """
        # --- 1. RBF Expansion (Gaussian) ---
        centers = np.linspace(Config.RBF_START, Config.RBF_END, Config.NUM_RBF)
        width = (Config.RBF_END - Config.RBF_START) / Config.NUM_RBF
        gamma = 1.0 / (width**2)

        # (E, 1) - (1, K) -> (E, K)
        diff = edge_dist[:, np.newaxis] - centers[np.newaxis, :]
        rbf = np.exp(-gamma * (diff**2))

        # --- 2. SBF Expansion (Legendre Polynomials) ---
        triplet_indices = Geometry.get_triplets(edge_index, num_atoms)

        if triplet_indices.shape[1] == 0:
            return (
                rbf.astype(np.float32),
                np.zeros((0, Config.NUM_SBF), dtype=np.float32),
                triplet_indices,
            )

        idx_kj = triplet_indices[0]
        idx_ji = triplet_indices[1]

        # Vector k->j
        vec_kj = edge_vec[idx_kj]
        # Vector j->i
        vec_ji = edge_vec[idx_ji]

        # Calculate angle at j. Vectors should point away from j.
        # vec_kj is k->j, so j->k is -vec_kj
        vec_jk = -vec_kj

        # Normalize
        norm_jk = np.linalg.norm(vec_jk, axis=1, keepdims=True) + 1e-8
        norm_ji = np.linalg.norm(vec_ji, axis=1, keepdims=True) + 1e-8

        u_jk = vec_jk / norm_jk
        u_ji = vec_ji / norm_ji

        # Cosine theta
        cos_theta = np.sum(u_jk * u_ji, axis=1)
        # Clip for numerical stability
        cos_theta = np.clip(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

        # Legendre Polynomials
        # P0(x) = 1, P1(x) = x, Pn = ((2n-1)xPn-1 - (n-1)Pn-2)/n
        sbf_list = []

        p_prev_2 = np.ones_like(cos_theta)  # P0
        p_prev_1 = cos_theta  # P1

        sbf_list.append(p_prev_2)
        if Config.NUM_SBF > 1:
            sbf_list.append(p_prev_1)

        for n in range(2, Config.NUM_SBF):
            p_curr = ((2 * n - 1) * cos_theta * p_prev_1 - (n - 1) * p_prev_2) / n
            sbf_list.append(p_curr)
            p_prev_2 = p_prev_1
            p_prev_1 = p_curr

        sbf = np.stack(sbf_list, axis=1)

        return rbf.astype(np.float32), sbf.astype(np.float32), triplet_indices


def process_and_cache_dataset(metadata_df, cache_path, load_cached_data=True):
    """
    Orchestrates the processing of the entire dataset.
    Loads structures, builds graphs, computes features, and caches the result.

    Args:
        metadata_df (pd.DataFrame): Metadata containing molecule names and targets.
        cache_path (str): Path to save/load the .npz file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing concatenated arrays of graph features and targets.
    """
    # 1. Load from cache if requested and available
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return dict(data)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing dataset (saving to {cache_path})...")

    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Group metadata by molecule to process efficiently
    # We want to process each molecule once, but extract all coupling targets associated with it
    grouped = metadata_df.groupby("molecule_name")
    unique_molecules = list(grouped.groups.keys())

    # Lists to store concatenated data
    all_atom_types = []
    all_coords = []

    all_edge_index = []
    all_edge_rbf = []

    all_triplet_indices = []
    all_triplet_sbf = []

    # Target data
    all_coupling_atom_0 = []
    all_coupling_atom_1 = []
    all_coupling_types = []
    all_coupling_targets = []
    all_coupling_mol_indices = []

    # Offsets for batching
    atom_offset = 0
    edge_offset = 0

    # Bookkeeping
    mol_to_idx = {}

    # Debug mode support
    if Config.DEBUG:
        unique_molecules = unique_molecules[: Config.DEBUG_SUBSET_SIZE]
        print(f"DEBUG MODE: Processing only {len(unique_molecules)} molecules.")

    count = 0
    total = len(unique_molecules)

    for mol_name in unique_molecules:
        count += 1
        if count % 1000 == 0:
            print(f"Processed {count}/{total} molecules...")

        # Get metadata for this molecule
        mol_meta = grouped.get_group(mol_name)

        # Load Structure
        # We assume the structure path is in the first row of the group (it's constant per molecule)
        xyz_path = mol_meta["structure_path"].iloc[0]
        try:
            atoms, coords = Geometry.load_structure(xyz_path)
        except Exception as e:
            print(f"Skipping {mol_name} due to error: {e}")
            continue

        num_atoms = len(atoms)

        # Build Graph
        edge_index, edge_vec, edge_dist = Geometry.build_neighbor_graph(coords)
        num_edges = edge_index.shape[1]

        # Compute Features
        rbf, sbf, triplet_indices = Geometry.compute_basis_expansions(
            edge_dist, edge_vec, edge_index, num_atoms
        )

        # Store Graph Data
        all_atom_types.append(atoms)
        all_coords.append(coords)

        # Offset edge indices by cumulative atom count
        all_edge_index.append(edge_index + atom_offset)
        all_edge_rbf.append(rbf)

        # Offset triplet indices by cumulative edge count
        # triplet_indices contains indices into the edge array
        all_triplet_indices.append(triplet_indices + edge_offset)
        all_triplet_sbf.append(sbf)

        # Store Target Data
        # Map coupling types to integers
        types = mol_meta["type"].map(Geometry.COUPLING_TYPE_MAP).values.astype(np.int32)

        # Coupling atom indices need to be offset to global graph indices
        c_atom_0 = mol_meta["atom_index_0"].values + atom_offset
        c_atom_1 = mol_meta["atom_index_1"].values + atom_offset

        all_coupling_types.append(types)
        all_coupling_atom_0.append(c_atom_0)
        all_coupling_atom_1.append(c_atom_1)

        # Store molecule index for each coupling (useful for splitting/debugging)
        mol_idx = len(mol_to_idx)
        mol_to_idx[mol_name] = mol_idx
        all_coupling_mol_indices.append(np.full(len(types), mol_idx, dtype=np.int32))

        # Store Targets if available (Train/Val)
        if "scalar_coupling_constant" in mol_meta.columns:
            targets = mol_meta["scalar_coupling_constant"].values.astype(np.float32)
            all_coupling_targets.append(targets)

        # Update offsets
        atom_offset += num_atoms
        edge_offset += num_edges

    # Concatenate all lists
    print("Concatenating arrays...")
    packed_data = {
        "atom_types": np.concatenate(all_atom_types),
        "coords": np.concatenate(all_coords),
        "edge_index": np.concatenate(all_edge_index, axis=1),
        "edge_rbf": np.concatenate(all_edge_rbf),
        "triplet_indices": (
            np.concatenate(all_triplet_indices, axis=1)
            if all_triplet_indices
            else np.zeros((2, 0), dtype=np.int64)
        ),
        "triplet_sbf": (
            np.concatenate(all_triplet_sbf)
            if all_triplet_sbf
            else np.zeros((0, Config.NUM_SBF), dtype=np.float32)
        ),
        "coupling_atom_0": np.concatenate(all_coupling_atom_0),
        "coupling_atom_1": np.concatenate(all_coupling_atom_1),
        "coupling_types": np.concatenate(all_coupling_types),
        "coupling_mol_indices": np.concatenate(all_coupling_mol_indices),
        # Save offsets to reconstruct individual graphs if needed
        "num_atoms_total": atom_offset,
        "num_edges_total": edge_offset,
    }

    if all_coupling_targets:
        packed_data["coupling_targets"] = np.concatenate(all_coupling_targets)

    # Save to disk
    print(f"Saving to {cache_path}...")
    np.savez(cache_path, **packed_data)

    return packed_data
