import os
import numpy as np
import pandas as pd
import ase.io
from ase.neighborlist import neighbor_list
from ase.data import chemical_symbols

from library.config import CACHE_DIR, ELEMENT_PROPERTIES, FEATURE_CONFIG, RANDOM_SEED
from library.utils import safe_mean, safe_variance, safe_percentile, safe_min, safe_max
from library.data_loader import DataLoader

# -----------------------------------------------------------------------------
# Core Descriptor Logic
# -----------------------------------------------------------------------------


def get_global_descriptors(atoms):
    """
    Calculates global physical properties of the structure.
    """
    volume = atoms.get_volume()
    # Calculate density: sum of atomic masses / volume
    # ASE masses are in atomic mass units (u).
    # To get g/cm^3, conversion factor is approx 1.6605e-24 / 1e-24 = 1.6605
    total_mass = sum(atoms.get_masses())
    density = (total_mass / volume) * 1.660539 if volume > 0 else 0.0

    return {
        "Global_Volume": volume,
        "Global_Density": density,
        "Global_NumAtoms": len(atoms),
    }


def get_rdf_fingerprints(atoms, cutoff=6.0, bins=30):
    """
    Computes Element-Resolved Radial Distribution Functions (RDF).
    """
    # Define pairs of interest based on composition (Al, Ga, In, O)
    elements = sorted(list(ELEMENT_PROPERTIES.keys()))
    pairs = []
    for i in range(len(elements)):
        for j in range(i, len(elements)):
            pairs.append((elements[i], elements[j]))

    # Get all pairwise distances within cutoff
    # 'd' returns distances
    i_indices, j_indices, dists = neighbor_list("ijd", atoms, cutoff)

    symbols = np.array(atoms.get_chemical_symbols())

    rdf_features = {}

    # Pre-calculate bin edges
    bin_edges = np.linspace(0, cutoff, bins + 1)

    for el1, el2 in pairs:
        # Mask for pairs (i is el1 AND j is el2) OR (i is el2 AND j is el1)
        # neighbor_list returns both i->j and j->i, so we can just check one direction if el1==el2,
        # but for el1!=el2 we need to be careful.
        # Actually, neighbor_list returns i, j. We can filter by symbols[i] == el1 and symbols[j] == el2.

        mask = (symbols[i_indices] == el1) & (symbols[j_indices] == el2)
        pair_dists = dists[mask]

        # Compute histogram
        hist, _ = np.histogram(pair_dists, bins=bin_edges)

        # Normalize by total number of atoms to make it intensive-like
        # (Density of pairs per atom)
        hist = hist / len(atoms)

        for k, count in enumerate(hist):
            feature_name = f"RDF_{el1}_{el2}_bin_{k}"
            rdf_features[feature_name] = count

    return rdf_features


def get_local_sublattice_fingerprints(
    atoms, cutoff=3.0, percentiles=[0, 25, 50, 75, 100]
):
    """
    Computes Cation Geometric and Anion Chemo-Structural fingerprints.
    Uses a local cutoff (default 3.0 A) to define the first coordination shell.
    """
    # Get neighbor list with vectors (D) to calculate angles
    # i: center, j: neighbor
    i_indices, j_indices, dists, vectors = neighbor_list("ijdD", atoms, cutoff)

    symbols = np.array(atoms.get_chemical_symbols())

    # Storage for collecting values per element type
    # Structure: { 'Al': {'ECoN': [], 'AngleVar': []}, 'O': {'MeanEN': [], ...} }
    cation_data = {el: {"ECoN": [], "AngleVar": []} for el in ["Al", "Ga", "In"]}
    anion_data = {"O": {"MeanEN": [], "MeanRadius": [], "MeanAngle": []}}

    # Iterate over each atom in the cell
    for atom_idx in range(len(atoms)):
        symbol = symbols[atom_idx]

        # Indices of neighbors for this atom
        # neighbor_list returns flattened arrays, find where i == atom_idx
        nbs_mask = i_indices == atom_idx
        if not np.any(nbs_mask):
            # No neighbors within cutoff (isolated atom?)
            # Assign NaNs or 0s. Let's append 0 for ECoN, NaN for others.
            if symbol in cation_data:
                cation_data[symbol]["ECoN"].append(0.0)
                cation_data[symbol]["AngleVar"].append(np.nan)
            elif symbol == "O":
                anion_data["O"]["MeanEN"].append(np.nan)
                anion_data["O"]["MeanRadius"].append(np.nan)
                anion_data["O"]["MeanAngle"].append(np.nan)
            continue

        nb_indices = j_indices[nbs_mask]
        nb_dists = dists[nbs_mask]
        nb_vectors = vectors[nbs_mask]
        nb_symbols = symbols[nb_indices]

        # --- Cation Logic (Al, Ga, In) ---
        if symbol in cation_data:
            # 1. ECoN (Effective Coordination Number)
            # Simplified continuous definition: sum(exp(1 - (d/d_min)^6)) is common,
            # but here we use a simpler weighted sum or just count for robustness.
            # Let's use the exponential drop-off based on average distance.
            # ECoN = sum_j exp(1 - (d_ij / d_av)^6)
            if len(nb_dists) > 0:
                d_av = np.mean(nb_dists)
                # Avoid division by zero
                if d_av < 1e-3:
                    d_av = 1.0
                econ = np.sum(np.exp(1.0 - (nb_dists / d_av) ** 6))
            else:
                econ = 0.0

            cation_data[symbol]["ECoN"].append(econ)

            # 2. Angle Variance
            # Calculate all bond angles O-M-O
            angles = []
            n_nbs = len(nb_indices)
            if n_nbs > 1:
                # Compute angles between all pairs of neighbors
                # Vector from atom->neighbor is nb_vectors
                # Angle between v1 and v2: arccos(dot(v1, v2) / (|v1||v2|))
                # Normalize vectors
                norms = np.linalg.norm(nb_vectors, axis=1)
                # Avoid zero division
                norms[norms < 1e-6] = 1.0
                unit_vectors = nb_vectors / norms[:, np.newaxis]

                # Dot product matrix
                dot_products = np.dot(unit_vectors, unit_vectors.T)
                # Clip for numerical stability
                dot_products = np.clip(dot_products, -1.0, 1.0)

                # Extract upper triangle off-diagonal
                # These are cos(theta)
                # We want angles in degrees
                angles_rad = np.arccos(dot_products)
                # Get indices for upper triangle, k=1
                tri_idx = np.triu_indices(n_nbs, k=1)
                valid_angles = np.degrees(angles_rad[tri_idx])
                angles.extend(valid_angles)

            if len(angles) > 0:
                angle_var = np.var(angles)
            else:
                angle_var = 0.0  # No angles defined

            cation_data[symbol]["AngleVar"].append(angle_var)

        # --- Anion Logic (Oxygen) ---
        elif symbol == "O":
            # Neighbors are likely Cations
            # 1. Mean Neighbor Electronegativity & Radius
            ens = []
            radii = []
            for nb_sym in nb_symbols:
                props = ELEMENT_PROPERTIES.get(nb_sym, {})
                if props:
                    ens.append(props.get("EN", np.nan))
                    radii.append(props.get("Radius", np.nan))

            mean_en = safe_mean(ens)
            mean_radius = safe_mean(radii)

            anion_data["O"]["MeanEN"].append(mean_en)
            anion_data["O"]["MeanRadius"].append(mean_radius)

            # 2. Mean M-O-M Angle
            # Similar angle calculation as cations, but centered on Oxygen
            angles = []
            n_nbs = len(nb_indices)
            if n_nbs > 1:
                norms = np.linalg.norm(nb_vectors, axis=1)
                norms[norms < 1e-6] = 1.0
                unit_vectors = nb_vectors / norms[:, np.newaxis]
                dot_products = np.dot(unit_vectors, unit_vectors.T)
                dot_products = np.clip(dot_products, -1.0, 1.0)
                angles_rad = np.arccos(dot_products)
                tri_idx = np.triu_indices(n_nbs, k=1)
                valid_angles = np.degrees(angles_rad[tri_idx])
                angles.extend(valid_angles)

            mean_angle = safe_mean(angles)
            anion_data["O"]["MeanAngle"].append(mean_angle)

    # --- Aggregation (Percentiles) ---
    features = {}

    # Process Cations
    for el in ["Al", "Ga", "In"]:
        for metric in ["ECoN", "AngleVar"]:
            values = cation_data[el][metric]
            # Calculate percentiles
            # If values is empty (e.g. no In atoms), safe_percentile returns NaNs
            # We can fill NaNs with 0 later or keep them to indicate absence
            # For tree models, NaN is fine.
            if not values:
                # If element not present, fill with 0 to avoid massive NaN columns if that helps,
                # but physically NaN is more correct. Let's stick to NaN (safe_percentile handles empty).
                pass

            stats = safe_percentile(values, percentiles)
            # stats is an array of shape (len(percentiles),)
            if np.isnan(stats).all():
                # If element is missing, maybe 0 is better for "Count" logic, but for properties NaN is safer.
                # However, to keep feature vector consistent:
                pass

            for p, val in zip(percentiles, stats):
                features[f"Cation_{el}_{metric}_p{p}"] = val

    # Process Anions
    for metric in ["MeanEN", "MeanRadius", "MeanAngle"]:
        values = anion_data["O"][metric]
        # Filter out NaNs from the list before percentile if any
        values = [v for v in values if not np.isnan(v)]

        stats = safe_percentile(values, percentiles)
        for p, val in zip(percentiles, stats):
            features[f"Anion_O_{metric}_p{p}"] = val

    return features


def process_single_structure(row, data_loader):
    """
    Worker function to process a single row from metadata.
    """
    file_path = row["file_path"]
    atoms = data_loader.load_geometry(file_path)

    if atoms is None:
        # Return None or empty dict, handled by caller
        return None

    # 1. Global
    feats = get_global_descriptors(atoms)

    # 2. RDF
    rdf = get_rdf_fingerprints(
        atoms, cutoff=FEATURE_CONFIG["rdf_cutoff"], bins=FEATURE_CONFIG["rdf_bins"]
    )
    feats.update(rdf)

    # 3. Local (Cation + Anion)
    local_feats = get_local_sublattice_fingerprints(
        atoms, cutoff=3.0, percentiles=FEATURE_CONFIG["percentiles"]
    )
    feats.update(local_feats)

    # 4. Pass-through Tabular Metadata (Lattice info)
    # These are already in the row, we just ensure they are in the final feature set if needed.
    # The prompt says "Include explicit Lattice Vectors... from the CSV".
    # We will assume the caller (generate_features) merges the original DF with these new features.

    return feats


def generate_features(metadata_df, load_cached_data=True, split="train"):
    """
    Main function to generate features for a dataset split.
    Handles caching to parquet files.
    """
    cache_file = os.path.join(CACHE_DIR, f"{split}_features.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}")
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute Features
    print(f"Generating features for {split} set ({len(metadata_df)} samples)...")

    data_loader = DataLoader()
    feature_list = []

    # Iterate rows (could be parallelized, but keeping simple for single file requirement)
    # Using a simple loop for clarity and robustness
    for idx, row in metadata_df.iterrows():
        feats = process_single_structure(row, data_loader)
        if feats is None:
            # Handle error (e.g. file missing), fill with NaNs or 0
            # We'll append an empty dict which becomes NaNs in DataFrame
            feature_list.append({})
        else:
            feature_list.append(feats)

    # Create DataFrame
    features_df = pd.DataFrame(feature_list)

    # 3. Merge with original metadata (to keep ID and tabular features like lattice vectors)
    # We want to keep the original index alignment
    # Drop file_path from metadata to avoid duplication if needed, but keeping it is fine.
    # We specifically want lattice info and composition.
    # Ensure indices match
    features_df.index = metadata_df.index

    # Concatenate
    # We exclude targets from metadata if they exist, to keep X separate from y usually,
    # but here we return the full feature set. The caller can separate X and y.
    # Actually, let's return everything combined.
    combined_df = pd.concat([metadata_df, features_df], axis=1)

    # 4. Save Cache
    print(f"Saving features to {cache_file}")
    combined_df.to_parquet(cache_file, index=False)

    return combined_df
