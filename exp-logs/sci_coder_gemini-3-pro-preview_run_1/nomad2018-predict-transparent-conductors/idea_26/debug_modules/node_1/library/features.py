import os
import numpy as np
import pandas as pd
import ase.io
from library import config
from library import utils


def get_atomic_one_hot(atomic_numbers):
    """
    Converts atomic numbers to one-hot encoding based on config.ATOM_TYPES.
    Mapping: Al(13), Ga(31), In(49), O(8).
    """
    # Map atomic number to index
    # ATOM_TYPES = ["Al", "Ga", "In", "O"]
    type_map = {13: 0, 31: 1, 49: 2, 8: 3}

    n_atoms = len(atomic_numbers)
    n_types = len(config.ATOM_TYPES)
    one_hot = np.zeros((n_atoms, n_types), dtype=np.float32)

    for i, z in enumerate(atomic_numbers):
        if z in type_map:
            one_hot[i, type_map[z]] = 1.0

    return one_hot


def compute_chemically_resolved_inverse_proximity(distances, atomic_numbers):
    """
    Calculates the inverse distance to the nearest neighbor of each element type.
    Returns a (N, 4) array.
    """
    n_atoms = len(atomic_numbers)
    n_types = len(config.ATOM_TYPES)
    # Mapping: Al(13)->0, Ga(31)->1, In(49)->2, O(8)->3
    type_map = {13: 0, 31: 1, 49: 2, 8: 3}

    # Initialize with 0.0 (representing infinite distance / absence)
    inv_proximity = np.zeros((n_atoms, n_types), dtype=np.float32)

    # Pre-compute indices for each type
    type_indices = {t: [] for t in range(n_types)}
    for idx, z in enumerate(atomic_numbers):
        if z in type_map:
            type_indices[type_map[z]].append(idx)

    for i in range(n_atoms):
        for t_idx in range(n_types):
            # Get indices of atoms of type t_idx
            target_indices = type_indices[t_idx]

            # Filter out self-interaction
            valid_targets = [idx for idx in target_indices if idx != i]

            if not valid_targets:
                inv_proximity[i, t_idx] = 0.0
            else:
                # Get distances to these targets
                dists = distances[i, valid_targets]
                # Avoid division by zero (though unlikely for distinct atoms)
                min_dist = np.min(dists)
                if min_dist > 1e-6:
                    inv_proximity[i, t_idx] = 1.0 / min_dist
                else:
                    inv_proximity[i, t_idx] = 0.0  # Should not happen in valid geometry

    return inv_proximity


def compute_local_packing_density(distances, k=12):
    """
    Calculates the mean distance to the K nearest neighbors.
    Returns a (N, 1) array.
    """
    n_atoms = distances.shape[0]
    packing_density = np.zeros((n_atoms, 1), dtype=np.float32)

    # If system is too small, adjust k
    # We need k neighbors, so we need at least k+1 atoms (including self)
    effective_k = min(k, n_atoms - 1)

    if effective_k <= 0:
        return packing_density

    for i in range(n_atoms):
        # Sort distances for atom i
        dists = np.sort(distances[i])
        # Exclude self (index 0, dist 0.0) and take next effective_k
        nearest_k = dists[1 : 1 + effective_k]
        packing_density[i, 0] = np.mean(nearest_k)

    return packing_density


def process_geometry(file_path_rel):
    """
    Parses .xyz file, centers coordinates, and extracts atomic info.
    """
    full_path = os.path.join(config.INPUT_DIR, file_path_rel)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Geometry file not found: {full_path}")

    # Cite debug_lesson_8: Explicitly Override File Format Inference for Ambiguous Extensions
    atoms = ase.io.read(full_path, format="aims")

    positions = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    lattice = atoms.get_cell()[:]  # 3x3 array

    # Center coordinates relative to centroid
    centroid = np.mean(positions, axis=0)
    centered_positions = positions - centroid

    return centered_positions, atomic_numbers, lattice


def prepare_features(metadata_path, cache_path, load_cached_data=True):
    """
    Main function to orchestrate feature extraction.

    Args:
        metadata_path (str): Path to the metadata CSV (train/val/test).
        cache_path (str): Path to save/load the processed .npz file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing processed arrays.
    """
    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        try:
            data = np.load(cache_path)
            # Reconstruct dictionary to ensure it's not just a NpzFile object
            return {key: data[key] for key in data.files}
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Computing features from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Limit samples for debugging if configured
    if config.MAX_SAMPLES is not None:
        df = df.iloc[: config.MAX_SAMPLES]

    # Lists to store data
    all_atomic_features = []  # Flattened list of atomic features
    batch_indices = []  # To map atoms to their graph/crystal index
    global_features_list = []
    targets_list = []
    ids_list = []

    total_samples = len(df)

    for idx, row in df.iterrows():
        # --- Atomic Stream Features ---
        rel_path = row["file_path"]
        centered_pos, atomic_nums, lattice = process_geometry(rel_path)

        # Calculate PBC distances
        # Note: We use original positions relative to each other (invariant to centering)
        # But calculate_pbc_distances handles the logic.
        # We can pass centered_pos because relative distances are preserved.
        pbc_dists = utils.calculate_pbc_distances(centered_pos, lattice)

        # 1. Atomic Identity (One-Hot)
        feat_identity = get_atomic_one_hot(atomic_nums)

        # 2. Spatial Context (Centered Coords)
        feat_spatial = centered_pos

        # 3. Inverse Proximity
        feat_prox = compute_chemically_resolved_inverse_proximity(
            pbc_dists, atomic_nums
        )

        # 4. Packing Density
        feat_packing = compute_local_packing_density(pbc_dists, k=config.K_NEIGHBORS)

        # Concatenate atomic features: [Identity(4), Spatial(3), Prox(4), Packing(1)] -> (N, 12)
        atom_feats = np.concatenate(
            [feat_identity, feat_spatial, feat_prox, feat_packing], axis=1
        )

        # Store
        all_atomic_features.append(atom_feats)
        # Create batch index for these atoms (all belong to sample idx)
        # Note: We use the loop index 'idx' (0 to N-1) relative to this dataset
        # But since we might slice the dataframe, we should use a running counter.
        # However, for simplicity in reconstruction, we just append N times the current sample index.
        # Better: use a simple counter 0, 1, 2... corresponding to the row in the output arrays.
        n_atoms = atom_feats.shape[0]
        batch_indices.append(np.full((n_atoms,), idx, dtype=np.int64))

        # --- Global Stream Features ---
        # 1. Lattice Lengths
        lengths = np.array(
            [
                row["lattice_vector_1_ang"],
                row["lattice_vector_2_ang"],
                row["lattice_vector_3_ang"],
            ]
        )

        # 2. Lattice Angles
        angles = np.array(
            [
                row["lattice_angle_alpha_degree"],
                row["lattice_angle_beta_degree"],
                row["lattice_angle_gamma_degree"],
            ]
        )

        # 3. Volume
        volume = utils.get_unit_cell_volume(lengths, angles)

        # 4. Density
        n_total_atoms = row["number_of_total_atoms"]
        density = utils.get_atomic_density(n_total_atoms, volume)

        # 5. Stoichiometry
        stoich = np.array(
            [row["percent_atom_al"], row["percent_atom_ga"], row["percent_atom_in"]]
        )

        # 6. Total Atoms
        total_atoms_feat = np.array([n_total_atoms])

        # Concatenate global: [Lengths(3), Angles(3), Vol(1), Dens(1), Stoich(3), N_atoms(1)] -> (12,)
        glob_feats = np.concatenate(
            [lengths, angles, [volume], [density], stoich, total_atoms_feat]
        )
        global_features_list.append(glob_feats)

        # --- Targets & IDs ---
        ids_list.append(row["id"])

        # Check if targets exist (train/val sets)
        if "formation_energy_ev_natom" in row and "bandgap_energy_ev" in row:
            targets_list.append(
                [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
            )
        else:
            # For test set, placeholder
            targets_list.append([0.0, 0.0])

    # Convert to numpy arrays
    # Flatten atomic features: (Total_Atoms, 12)
    atomic_features_np = np.vstack(all_atomic_features).astype(np.float32)
    batch_indices_np = np.concatenate(batch_indices).astype(np.int64)

    global_features_np = np.vstack(global_features_list).astype(np.float32)
    ids_np = np.array(ids_list, dtype=np.int64)
    targets_np = np.vstack(targets_list).astype(np.float32)

    # 3. Save to cache
    print(f"Saving processed features to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        atomic_features=atomic_features_np,
        batch_indices=batch_indices_np,
        global_features=global_features_np,
        ids=ids_np,
        targets=targets_np,
    )

    return {
        "atomic_features": atomic_features_np,
        "batch_indices": batch_indices_np,
        "global_features": global_features_np,
        "ids": ids_np,
        "targets": targets_np,
    }
