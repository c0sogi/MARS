import os
import numpy as np
import pandas as pd
import ase.io
from ase.neighborlist import neighbor_list
import library.config as config

# Set random seeds for reproducibility
np.random.seed(config.RANDOM_SEED)


def get_physical_descriptors(atoms):
    """
    Calculates unit cell volume and mass density.
    """
    vol = atoms.get_volume()
    mass = sum(atoms.get_masses())
    # Density in AMU / Angstrom^3
    density = mass / vol if vol > 0 else 0.0
    return {"volume": vol, "density": density}


def get_rdf_features(atoms):
    """
    Computes element-resolved Radial Distribution Functions.
    Normalized by atom count.
    """
    # Get all distances up to cutoff
    # 'i' and 'j' are atom indices, 'd' is distance
    i_indices, j_indices, d_indices = neighbor_list("ijd", atoms, config.RDF_CUTOFF)

    symbols = np.array(atoms.get_chemical_symbols())
    n_atoms = len(atoms)

    features = {}

    # Define bins
    bins = np.linspace(0, config.RDF_CUTOFF, config.RDF_BINS + 1)

    # Get symbols for neighbor pairs
    if len(i_indices) > 0:
        sym_i = symbols[i_indices]
        sym_j = symbols[j_indices]
    else:
        sym_i = np.array([])
        sym_j = np.array([])

    # Iterate over all unique pairs of elements
    # We use sorted tuples to ensure A-B and B-A are treated as the same interaction type
    unique_pairs = []
    for i in range(len(config.ELEMENTS)):
        for j in range(i, len(config.ELEMENTS)):
            unique_pairs.append(tuple(sorted((config.ELEMENTS[i], config.ELEMENTS[j]))))

    for el1, el2 in unique_pairs:
        if len(d_indices) == 0:
            hist = np.zeros(config.RDF_BINS)
        else:
            # Create mask for this pair type
            if el1 == el2:
                mask = (sym_i == el1) & (sym_j == el2)
            else:
                mask = ((sym_i == el1) & (sym_j == el2)) | (
                    (sym_i == el2) & (sym_j == el1)
                )

            dists = d_indices[mask]

            # Compute histogram
            hist, _ = np.histogram(dists, bins=bins)

        # Normalize by total number of atoms
        norm_factor = n_atoms
        hist = hist / norm_factor if norm_factor > 0 else hist

        for k, val in enumerate(hist):
            features[f"rdf_{el1}_{el2}_bin_{k}"] = val

    return features


def get_distributional_local_env(atoms):
    """
    Calculates atom-level CN and Bond Angle Variance, aggregated by element percentiles.
    """
    # Get neighbors for bonding
    # 'i' is central atom index, 'j' is neighbor index, 'D' is vector i->j
    i_indices, j_indices, d_vectors = neighbor_list(
        "ijD", atoms, config.NEIGHBOR_CUTOFF
    )

    n_atoms = len(atoms)
    symbols = np.array(atoms.get_chemical_symbols())

    # Initialize containers
    cns = np.zeros(n_atoms)
    angle_vars = np.zeros(n_atoms)  # Default to 0.0

    # Calculate CN
    if len(i_indices) > 0:
        counts = np.bincount(i_indices, minlength=n_atoms)
        cns = counts.astype(float)

        # Calculate Bond Angle Variance
        # Iterate only over atoms that have neighbors
        unique_centers = np.unique(i_indices)

        for center_idx in unique_centers:
            mask = i_indices == center_idx
            vecs = d_vectors[mask]

            if len(vecs) < 2:
                angle_vars[center_idx] = 0.0
                continue

            # Normalize vectors
            norms = np.linalg.norm(vecs, axis=1)
            # Avoid division by zero
            with np.errstate(invalid="ignore"):
                unit_vecs = vecs / norms[:, np.newaxis]

            # Compute angles between all pairs of neighbors
            # Dot product of unit vectors gives cos(theta)
            cos_angles = np.dot(unit_vecs, unit_vecs.T)

            # Clip for numerical stability
            cos_angles = np.clip(cos_angles, -1.0, 1.0)

            # Get angles in degrees
            angles = np.degrees(np.arccos(cos_angles))

            # We only want unique pairs (i < j), excluding self (diagonal)
            triu_indices = np.triu_indices(len(vecs), k=1)
            unique_angles = angles[triu_indices]

            if len(unique_angles) > 0:
                angle_vars[center_idx] = np.var(unique_angles)
            else:
                angle_vars[center_idx] = 0.0

    # Aggregate by element
    features = {}

    for el in config.ELEMENTS:
        # Mask for this element
        el_mask = symbols == el

        if np.sum(el_mask) == 0:
            # Element not present in this structure
            for p in config.PERCENTILES:
                features[f"dle_{el}_cn_p{p}"] = 0.0
                features[f"dle_{el}_angvar_p{p}"] = 0.0
        else:
            el_cns = cns[el_mask]
            el_angvars = angle_vars[el_mask]

            cn_percentiles = np.percentile(el_cns, config.PERCENTILES)
            angvar_percentiles = np.percentile(el_angvars, config.PERCENTILES)

            for i, p in enumerate(config.PERCENTILES):
                features[f"dle_{el}_cn_p{p}"] = cn_percentiles[i]
                features[f"dle_{el}_angvar_p{p}"] = angvar_percentiles[i]

    return features


def generate_features(metadata_path, output_path, load_cached_data=True):
    """
    Main function to generate or load features.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(output_path):
        print(f"Loading cached features from {output_path}")
        return pd.read_parquet(output_path)

    print(f"Generating features for {metadata_path}...")

    # 2. Load Metadata
    df = pd.read_csv(metadata_path)

    # 3. Iterate and Extract
    feature_rows = []

    # Pre-define exclusion list for tabular features
    exclude_cols = [
        "id",
        "file_path",
        "formation_energy_ev_natom",
        "bandgap_energy_ev",
        "stratify_bin",
    ]
    tabular_cols = [c for c in df.columns if c not in exclude_cols]

    for _, row in df.iterrows():
        # Load Geometry
        rel_path = row["file_path"]
        full_path = os.path.join(config.INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Geometry file not found: {full_path}")

        atoms = ase.io.read(full_path, format="aims")

        # Extract Features
        phys_feats = get_physical_descriptors(atoms)
        rdf_feats = get_rdf_features(atoms)
        dle_feats = get_distributional_local_env(atoms)

        # Combine all features
        combined = {**phys_feats, **rdf_feats, **dle_feats}

        # Add ID for tracking
        combined["id"] = row["id"]

        # Add tabular features from metadata
        for col in tabular_cols:
            combined[col] = row[col]

        feature_rows.append(combined)

    # 4. Create DataFrame
    features_df = pd.DataFrame(feature_rows)

    # 5. Save to Cache
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features_df.to_parquet(output_path, index=False)
    print(f"Features saved to {output_path}")

    return features_df
