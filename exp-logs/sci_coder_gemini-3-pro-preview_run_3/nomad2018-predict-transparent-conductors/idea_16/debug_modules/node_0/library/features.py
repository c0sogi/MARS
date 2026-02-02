import os
import numpy as np
import pandas as pd
import ase.io
from scipy.spatial import Voronoi, ConvexHull, QhullError
from collections import defaultdict
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    ATOM_TYPES,
    RDF_CUTOFF,
    RDF_NUM_BINS,
    VORONOI_SUPERCELL_REPEAT,
)


def compute_global_descriptors(atoms):
    """
    Computes global physical descriptors for the unit cell.

    Args:
        atoms (ase.Atoms): The atomic structure.

    Returns:
        dict: Dictionary containing 'vol_per_atom' and 'mass_density'.
    """
    volume = atoms.get_volume()
    mass = sum(atoms.get_masses())
    num_atoms = len(atoms)

    return {
        "vol_per_atom": volume / num_atoms if num_atoms > 0 else 0.0,
        "mass_density": mass / volume if volume > 0 else 0.0,
    }


def compute_elemental_rdf(atoms):
    """
    Computes element-resolved Radial Distribution Functions (RDF).

    Args:
        atoms (ase.Atoms): The atomic structure.

    Returns:
        dict: Flattened RDF histograms for each element pair.
    """
    # Create a sufficiently large supercell for RDF calculation to respect cutoff
    # A simple 2x2x2 repeat is usually enough for 6.0 Angstrom cutoff in these crystals
    # but we calculate dynamically to be safe or just use neighbor list logic.
    # For simplicity and speed with ASE, we use get_distances on the original cell
    # with mic=True, but strictly speaking RDF requires neighbors from periodic images.
    # We will use a 3x3x3 supercell to ensure we capture the cutoff distance.

    supercell = atoms.repeat((3, 3, 3))
    center_indices = range(
        len(atoms) * 13, len(atoms) * 14
    )  # Indices of the central image

    # We only care about distances from the central unit cell to all other atoms in supercell
    # within the cutoff.

    distances_dict = defaultdict(list)
    chemical_symbols = np.array(supercell.get_chemical_symbols())
    positions = supercell.get_positions()

    # Optimization: Use KDTree or Cell list if available, but brute force with numpy is fine for small systems
    # Calculate distances from central atoms to all atoms in supercell
    center_pos = positions[center_indices]

    # Vectorized distance calculation
    # Shape: (N_center, N_super)
    dists = np.linalg.norm(center_pos[:, None, :] - positions[None, :, :], axis=2)

    # Mask for cutoff and self-interaction (distance > 0)
    mask = (dists > 0) & (dists <= RDF_CUTOFF)

    # Iterate over central atoms to bin interactions
    original_symbols = atoms.get_chemical_symbols()

    for i, atom_idx in enumerate(center_indices):
        elem_i = original_symbols[i]
        valid_indices = np.where(mask[i])[0]

        for neighbor_idx in valid_indices:
            d = dists[i, neighbor_idx]
            elem_j = chemical_symbols[neighbor_idx]

            # Sort pair to ensure symmetry (Al-O is same as O-Al)
            pair = tuple(sorted((elem_i, elem_j)))
            distances_dict[pair].append(d)

    # Generate Histograms
    features = {}
    bins = np.linspace(0, RDF_CUTOFF, RDF_NUM_BINS + 1)

    # Define all possible pairs
    all_pairs = []
    for i in range(len(ATOM_TYPES)):
        for j in range(i, len(ATOM_TYPES)):
            all_pairs.append(tuple(sorted((ATOM_TYPES[i], ATOM_TYPES[j]))))

    for pair in all_pairs:
        d_list = distances_dict.get(pair, [])
        hist, _ = np.histogram(d_list, bins=bins)

        # Normalize by total number of atoms in the unit cell to make it intensive
        norm_hist = hist / len(atoms)

        for k, val in enumerate(norm_hist):
            features[f"RDF_{pair[0]}_{pair[1]}_bin_{k}"] = val

    return features


def compute_voronoi_fingerprints(atoms):
    """
    Computes topological features based on Voronoi tessellation of the crystal structure.
    Aggregates Voronoi volume and coordination number by chemical element.

    Args:
        atoms (ase.Atoms): The atomic structure.

    Returns:
        dict: Aggregated Voronoi statistics (mean, std) for each element.
    """
    # Jiggle atoms slightly to avoid degenerate vertices in perfect crystals
    atoms_jit = atoms.copy()
    atoms_jit.rattle(stdev=1e-4, seed=42)

    # Create supercell
    supercell = atoms_jit.repeat(VORONOI_SUPERCELL_REPEAT)
    positions = supercell.get_positions()

    # Determine indices of the central unit cell
    # Repeat order in ASE is z, y, x (slowest to fastest) or x, y, z depending on version,
    # but standard 3x3x3 puts the center image at index 13 (0-26 flattened).
    # Center image corresponds to translation (1, 1, 1).
    # Grid size is 3x3x3 = 27 cells.
    # Index offset = (1*9 + 1*3 + 1) = 13.
    n_atoms = len(atoms)
    center_start = 13 * n_atoms
    center_end = 14 * n_atoms
    target_indices = list(range(center_start, center_end))

    features = {}

    try:
        # Compute Voronoi Tessellation
        vor = Voronoi(positions)

        # Storage for metrics
        element_volumes = defaultdict(list)
        element_coordination = defaultdict(list)

        symbols = supercell.get_chemical_symbols()

        for i, atom_idx in enumerate(target_indices):
            element = symbols[atom_idx]
            region_idx = vor.point_region[atom_idx]
            region_vertices_indices = vor.regions[region_idx]

            # Check if region is finite (no -1 in indices and not empty)
            if -1 in region_vertices_indices or len(region_vertices_indices) == 0:
                continue

            vertices = vor.vertices[region_vertices_indices]

            # Calculate Volume
            try:
                hull = ConvexHull(vertices)
                vol = hull.volume
            except QhullError:
                vol = 0.0

            # Calculate Coordination Number (Number of faces of the polyhedra)
            # In Voronoi, number of faces corresponds to number of neighbors sharing a boundary.
            # We can count how many ridge_points involve this atom_idx.
            # ridge_points is a list of [p1, p2] indices.
            coordination = 0
            for p1, p2 in vor.ridge_points:
                if p1 == atom_idx or p2 == atom_idx:
                    coordination += 1

            element_volumes[element].append(vol)
            element_coordination[element].append(coordination)

        # Aggregate statistics
        for elem in ATOM_TYPES:
            vols = element_volumes.get(elem, [])
            coords = element_coordination.get(elem, [])

            if vols:
                features[f"Voronoi_Vol_Mean_{elem}"] = np.mean(vols)
                features[f"Voronoi_Vol_Std_{elem}"] = np.std(vols)
            else:
                features[f"Voronoi_Vol_Mean_{elem}"] = 0.0
                features[f"Voronoi_Vol_Std_{elem}"] = 0.0

            if coords:
                features[f"Voronoi_Coord_Mean_{elem}"] = np.mean(coords)
                features[f"Voronoi_Coord_Std_{elem}"] = np.std(coords)
            else:
                features[f"Voronoi_Coord_Mean_{elem}"] = 0.0
                features[f"Voronoi_Coord_Std_{elem}"] = 0.0

    except Exception as e:
        # Fallback in case of tessellation failure
        for elem in ATOM_TYPES:
            features[f"Voronoi_Vol_Mean_{elem}"] = 0.0
            features[f"Voronoi_Vol_Std_{elem}"] = 0.0
            features[f"Voronoi_Coord_Mean_{elem}"] = 0.0
            features[f"Voronoi_Coord_Std_{elem}"] = 0.0

    return features


def extract_features(atoms):
    """
    Master function to extract all features for a given structure.

    Args:
        atoms (ase.Atoms): The atomic structure.

    Returns:
        pd.Series: Flattened feature vector.
    """
    # 1. Global Descriptors
    global_feats = compute_global_descriptors(atoms)

    # 2. Radial Distribution Functions
    rdf_feats = compute_elemental_rdf(atoms)

    # 3. Voronoi Topological Features
    voronoi_feats = compute_voronoi_fingerprints(atoms)

    # Combine all
    all_features = {**global_feats, **rdf_feats, **voronoi_feats}

    return pd.Series(all_features)


def process_dataset(metadata_df, load_cached_data=True, dataset_name="train"):
    """
    Processes a dataset (train, val, or test) to extract features.
    Handles caching to parquet files.

    Args:
        metadata_df (pd.DataFrame): Dataframe containing 'file_path' and other metadata.
        load_cached_data (bool): If True, attempts to load from cache.
        dataset_name (str): Name of the dataset for cache file naming ('train', 'val', 'test').

    Returns:
        pd.DataFrame: DataFrame containing extracted features and metadata columns.
    """
    cache_path = os.path.join(WORKING_DIR, f"{dataset_name}_features.parquet")

    # Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Computing features for {dataset_name} set ({len(metadata_df)} samples)...")

    features_list = []

    # Iterate over metadata
    for idx, row in metadata_df.iterrows():
        file_path = os.path.join(INPUT_DIR, row["file_path"])

        try:
            atoms = ase.io.read(file_path)
            feats = extract_features(atoms)

            # Add ID for merging/tracking
            feats["id"] = row["id"]
            features_list.append(feats)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Append empty series with ID to maintain alignment, or skip
            # We'll skip and let the merger handle it (inner join) or fillna later
            pass

    # Create DataFrame from features
    features_df = pd.DataFrame(features_list)

    # Merge with original metadata to keep tabular features (composition, spacegroup)
    # We use 'id' as key
    merged_df = pd.merge(metadata_df, features_df, on="id", how="inner")

    # Save to cache
    print(f"Saving features to {cache_path}...")
    merged_df.to_parquet(cache_path, index=False)

    return merged_df
