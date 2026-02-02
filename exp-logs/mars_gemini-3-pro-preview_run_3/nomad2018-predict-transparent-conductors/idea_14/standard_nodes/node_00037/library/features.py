import os
import numpy as np
import pandas as pd
import ase.io
from ase.neighborlist import neighbor_list
from library.config import Config


def compute_physical_descriptors(atoms):
    """
    Computes global physical descriptors: Volume and Density.
    """
    try:
        vol = atoms.get_volume()
        # ASE masses are in atomic mass units (u).
        # Density is proportional to sum(mass) / volume.
        # We don't need strict kg/m^3 units, just consistency.
        total_mass = sum(atoms.get_masses())
        density = total_mass / vol if vol > 1e-9 else 0.0
        return [vol, density]
    except Exception:
        return [0.0, 0.0]


def compute_rdf(atoms, cutoff=6.0, n_bins=60, elements=None):
    """
    Computes element-resolved Radial Distribution Functions.
    Returns a flattened feature vector.
    """
    if elements is None:
        elements = ["Al", "Ga", "In", "O"]

    n_elements = len(elements)
    # Feature vector size: (n_elements * (n_elements + 1) / 2) * n_bins
    # But simpler to just do n_elements * n_elements pairs including self-interaction (which will be empty for distinct atoms usually, but useful for general code)
    # Actually, let's do all unique pairs including self (e.g. Al-Al, Al-Ga, ...)

    # Get all distances with MIC
    # This returns a matrix of distances
    all_distances = atoms.get_all_distances(mic=True)
    symbols = np.array(atoms.get_chemical_symbols())

    rdf_features = []

    # Pre-calculate masks for each element
    masks = {el: (symbols == el) for el in elements}

    # Define bin edges
    bins = np.linspace(0, cutoff, n_bins + 1)

    # Iterate over all unique pairs of elements
    for i in range(n_elements):
        el1 = elements[i]
        mask1 = masks[el1]
        if not np.any(mask1):
            # If element not present, append zeros for all its pairs
            for j in range(i, n_elements):
                rdf_features.extend([0.0] * n_bins)
            continue

        for j in range(i, n_elements):
            el2 = elements[j]
            mask2 = masks[el2]

            if not np.any(mask2):
                rdf_features.extend([0.0] * n_bins)
                continue

            # Extract relevant submatrix of distances
            # We want distances between atoms of type el1 and type el2
            # submatrix shape: (count_el1, count_el2)
            dists_sub = all_distances[np.ix_(mask1, mask2)]

            # Flatten
            dists_flat = dists_sub.flatten()

            # Filter by cutoff and exclude self-distance (0.0) if el1 == el2
            dists_flat = dists_flat[(dists_flat > 0.01) & (dists_flat < cutoff)]

            # Histogram
            hist, _ = np.histogram(dists_flat, bins=bins)

            # Normalize by total number of atoms in cell to make it intensive-ish
            # (Density independent of supercell size)
            norm_factor = len(atoms)
            hist_norm = hist / norm_factor

            rdf_features.extend(hist_norm)

    return rdf_features


def compute_local_moments(atoms, cutoff=3.0, elements=None):
    """
    Computes Chemically-Resolved Local Environment Moments (CR-LEM).
    For each species, calculates Mean/Std of Coordination Number and Bond Angle Variance.
    """
    if elements is None:
        elements = ["Al", "Ga", "In", "O"]

    # Get neighbor list
    # i: index of central atom
    # j: index of neighbor
    # D: distance vector
    i_indices, j_indices, _, D_vectors = neighbor_list("ijdD", atoms, cutoff)

    # Initialize containers for atom-level metrics
    n_atoms = len(atoms)
    coordination_numbers = np.zeros(n_atoms, dtype=int)
    angle_variances = np.zeros(n_atoms, dtype=float)

    # Count neighbors
    # np.bincount works if indices are 0..N-1
    if len(i_indices) > 0:
        counts = np.bincount(i_indices, minlength=n_atoms)
        coordination_numbers = counts

    # Calculate Angle Variance for each atom
    # This is expensive, so we optimize.
    # Group neighbors by central atom index

    # Sort by i to group neighbors
    if len(i_indices) > 0:
        sort_order = np.argsort(i_indices)
        i_sorted = i_indices[sort_order]
        # j_sorted = j_indices[sort_order] # Not strictly needed for angle, we need vectors
        D_sorted = D_vectors[sort_order]

        # Find start/end indices for each atom
        # unique_indices, split_indices = np.unique(i_sorted, return_index=True)
        # We can iterate manually or use split. Split is easier.

        # Get counts for splitting
        counts_sorted = np.bincount(i_sorted, minlength=n_atoms)

        # Calculate split points (cumulative sum of counts)
        split_points = np.cumsum(counts_sorted)[:-1]

        # Split D vectors by atom
        D_grouped = np.split(D_sorted, split_points)

        for atom_idx, vecs in enumerate(D_grouped):
            k = len(vecs)
            if k < 2:
                angle_variances[atom_idx] = 0.0
                continue

            # Normalize vectors
            norms = np.linalg.norm(vecs, axis=1)
            # Avoid division by zero
            norms[norms < 1e-6] = 1.0
            vecs_normalized = vecs / norms[:, np.newaxis]

            # Compute cosine similarity matrix: (k, 3) @ (3, k) -> (k, k)
            cos_matrix = np.dot(vecs_normalized, vecs_normalized.T)

            # Clip for numerical stability
            cos_matrix = np.clip(cos_matrix, -1.0, 1.0)

            # Convert to angles (radians)
            angles = np.arccos(cos_matrix)

            # We only care about upper triangle off-diagonal (unique pairs)
            # k neighbors -> k*(k-1)/2 angles
            tri_indices = np.triu_indices(k, k=1)
            unique_angles = angles[tri_indices]

            # Variance (in degrees squared, or radians squared - let's use radians)
            if len(unique_angles) > 0:
                angle_variances[atom_idx] = np.var(unique_angles)
            else:
                angle_variances[atom_idx] = 0.0

    # Aggregate by species
    symbols = np.array(atoms.get_chemical_symbols())
    lem_features = []

    for el in elements:
        mask = symbols == el
        if np.any(mask):
            cns = coordination_numbers[mask]
            vars = angle_variances[mask]

            lem_features.extend(
                [np.mean(cns), np.std(cns), np.mean(vars), np.std(vars)]
            )
        else:
            # Element not present
            lem_features.extend([0.0, 0.0, 0.0, 0.0])

    return lem_features


def process_dataset(metadata_path, cache_file, load_cached_data=True, debug=False):
    """
    Main processing function.
    Reads metadata, loads XYZ files, computes features, and returns a DataFrame.
    """
    cache_path = os.path.join(Config.WORKING_DIR, cache_file)

    # 1. Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    # 2. Compute from scratch
    print(f"Computing features for {metadata_path}...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    if debug:
        print(f"Debug mode: processing first {Config.DEBUG_SAMPLE_SIZE} samples.")
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    # Feature lists
    phys_feats = []
    rdf_feats = []
    lem_feats = []

    # Iterate
    for idx, row in df.iterrows():
        # Construct full path
        # The metadata file_path is relative to input dir, e.g. "train/1/geometry.xyz"
        # Config.INPUT_DIR is "./input"
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            atoms = ase.io.read(full_path, format="aims")

            # 1. Physical
            phys_feats.append(compute_physical_descriptors(atoms))

            # 2. RDF
            rdf_feats.append(
                compute_rdf(
                    atoms,
                    cutoff=Config.RDF_CUTOFF,
                    n_bins=Config.RDF_NUM_BINS,
                    elements=Config.ELEMENTS,
                )
            )

            # 3. LEM
            lem_feats.append(
                compute_local_moments(
                    atoms, cutoff=Config.NEIGHBOR_CUTOFF, elements=Config.ELEMENTS
                )
            )

        except Exception as e:
            print(f"Error processing {full_path}: {e}")
            # Append zeros to maintain shape
            # Phys: 2
            phys_feats.append([0.0, 0.0])

            # RDF: n_pairs * n_bins. n_elements=4 -> 10 pairs (4+3+2+1) -> 10 * 60 = 600
            # Wait, implementation loop:
            # i=0 (Al): Al-Al, Al-Ga, Al-In, Al-O (4)
            # i=1 (Ga): Ga-Ga, Ga-In, Ga-O (3)
            # i=2 (In): In-In, In-O (2)
            # i=3 (O):  O-O (1)
            # Total 10 pairs.
            n_pairs = 10
            rdf_feats.append([0.0] * (n_pairs * Config.RDF_NUM_BINS))

            # LEM: n_elements * 4 stats = 16
            lem_feats.append([0.0] * (len(Config.ELEMENTS) * 4))

    # Convert to DataFrames
    phys_cols = ["vol", "density"]

    # Generate RDF column names
    rdf_cols = []
    elements = Config.ELEMENTS
    n_bins = Config.RDF_NUM_BINS
    for i in range(len(elements)):
        for j in range(i, len(elements)):
            pair_name = f"{elements[i]}_{elements[j]}"
            for b in range(n_bins):
                rdf_cols.append(f"rdf_{pair_name}_{b}")

    # Generate LEM column names
    lem_cols = []
    for el in elements:
        lem_cols.extend(
            [
                f"lem_{el}_cn_mean",
                f"lem_{el}_cn_std",
                f"lem_{el}_ang_mean",
                f"lem_{el}_ang_std",
            ]
        )

    df_phys = pd.DataFrame(phys_feats, columns=phys_cols, index=df.index)
    df_rdf = pd.DataFrame(rdf_feats, columns=rdf_cols, index=df.index)
    df_lem = pd.DataFrame(lem_feats, columns=lem_cols, index=df.index)

    # Concatenate with original metadata (excluding file_path to save space if needed, but keeping id)
    # We keep tabular features from metadata: spacegroup, percent_atom_*, lattice_*
    # Identify tabular columns
    exclude_cols = ["file_path", "formation_energy_ev_natom", "bandgap_energy_ev", "id"]
    tabular_cols = [c for c in df.columns if c not in exclude_cols]

    df_tabular = df[tabular_cols]

    # Final Feature Matrix
    # We also keep ID and targets for training convenience
    df_final = pd.concat([df[["id"]], df_tabular, df_phys, df_lem, df_rdf], axis=1)

    # Add targets if they exist
    if "formation_energy_ev_natom" in df.columns:
        df_final["target_formation"] = df["formation_energy_ev_natom"]
        df_final["target_bandgap"] = df["bandgap_energy_ev"]

    # Save to cache
    print(f"Saving features to {cache_path}...")
    df_final.to_parquet(cache_path)

    return df_final
