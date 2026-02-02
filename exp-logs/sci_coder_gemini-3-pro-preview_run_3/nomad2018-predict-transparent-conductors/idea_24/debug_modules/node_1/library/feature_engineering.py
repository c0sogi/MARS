import os
import numpy as np
import pandas as pd
import ase.io
from ase.neighborlist import neighbor_list
from library.config import INPUT_DIR, WORKING_DIR, FEATURE_PARAMS, RANDOM_SEED


def compute_rdf(atoms, cutoff=6.0, bins=60, elements=None):
    """
    Computes element-resolved Radial Distribution Functions (RDF).
    """
    if elements is None:
        elements = ["Al", "Ga", "In", "O"]

    # Get all pairwise distances respecting PBC
    # 'd' returns distances
    i_indices, j_indices, dists = neighbor_list("ijd", atoms, cutoff)

    symbols = np.array(atoms.get_chemical_symbols())

    # Initialize RDF dictionary
    rdf_features = {}

    # Define bin edges
    bin_edges = np.linspace(0, cutoff, bins + 1)

    # Iterate over pairs of elements to create specific RDFs (e.g., Al-O, Ga-O)
    # We focus on Metal-Oxygen and Metal-Metal interactions primarily, but let's do all unique pairs
    # to be comprehensive as per the strategy.

    # Map indices to symbols
    sym_i = symbols[i_indices]
    sym_j = symbols[j_indices]

    for el1 in elements:
        for el2 in elements:
            if el1 > el2:
                continue  # Avoid duplicates (Al-O is same as O-Al distance-wise)

            # Filter distances for this pair
            mask = ((sym_i == el1) & (sym_j == el2)) | ((sym_i == el2) & (sym_j == el1))
            pair_dists = dists[mask]

            # Compute histogram
            hist, _ = np.histogram(pair_dists, bins=bin_edges)

            # Normalize by volume and number of atoms to make it intensive
            # A simple normalization is by total number of atoms in cell
            norm_factor = len(atoms)
            if norm_factor > 0:
                hist = hist / norm_factor

            # Store features
            for b in range(bins):
                rdf_features[f"rdf_{el1}_{el2}_bin_{b}"] = hist[b]

    return rdf_features


def compute_site_metrics(atoms, cutoff=3.0):
    """
    Computes atom-centric continuous geometric descriptors:
    1. Effective Coordination Number (ECoN)
    2. Radial Strain (Bond Length Variance)
    3. Angular Strain (Bond Angle Variance)
    """
    # Get neighbor list with distances (d) and vector distances (D)
    i_idx, j_idx, d, D = neighbor_list("ijdD", atoms, cutoff)

    num_atoms = len(atoms)

    # Initialize arrays for metrics
    econ = np.zeros(num_atoms)
    rad_strain = np.zeros(num_atoms)
    ang_strain = np.zeros(num_atoms)

    # Pre-calculate counts to avoid division by zero
    counts = np.bincount(i_idx, minlength=num_atoms)

    for a in range(num_atoms):
        # Indices of neighbors for atom a
        neighbors_mask = i_idx == a

        if not np.any(neighbors_mask):
            continue

        d_neighbors = d[neighbors_mask]
        D_neighbors = D[neighbors_mask]

        # --- 1. Effective Coordination Number (ECoN) ---
        # Formula: sum(exp(1 - (d_i / d_avg)^6))
        d_avg = np.mean(d_neighbors)
        if d_avg > 1e-6:
            contributions = np.exp(1 - (d_neighbors / d_avg) ** 6)
            econ[a] = np.sum(contributions)
        else:
            econ[a] = 0.0

        # --- 2. Radial Strain (Bond Length Variance) ---
        if len(d_neighbors) > 1:
            rad_strain[a] = np.var(d_neighbors)
        else:
            rad_strain[a] = 0.0

        # --- 3. Angular Strain (Bond Angle Variance) ---
        # Calculate angles between all pairs of neighbors
        # Vectors from atom a to neighbors are D_neighbors
        # Normalize vectors
        norms = np.linalg.norm(D_neighbors, axis=1)
        # Avoid division by zero
        valid_norms = norms > 1e-6
        if np.sum(valid_norms) < 2:
            ang_strain[a] = 0.0
            continue

        vecs = D_neighbors[valid_norms] / norms[valid_norms][:, np.newaxis]

        # Compute dot products for all pairs (cosine of angles)
        # We only care about unique pairs (k < l)
        n_neigh = len(vecs)
        angles = []
        for k in range(n_neigh):
            for l in range(k + 1, n_neigh):
                cosine = np.dot(vecs[k], vecs[l])
                # Clip to handle numerical errors outside [-1, 1]
                cosine = np.clip(cosine, -1.0, 1.0)
                angle = np.arccos(cosine) * (180.0 / np.pi)  # Degrees
                angles.append(angle)

        if angles:
            ang_strain[a] = np.var(angles)
        else:
            ang_strain[a] = 0.0

    return econ, rad_strain, ang_strain


def compute_topology_features(atoms, cutoff=3.0, percentiles=[0, 25, 50, 75, 100]):
    """
    Computes global distributions of bond angles for specific motifs:
    - Metal-Oxygen-Metal (M-O-M)
    - Oxygen-Metal-Oxygen (O-M-O)
    """
    i_idx, j_idx, _, D = neighbor_list("ijdD", atoms, cutoff)
    symbols = np.array(atoms.get_chemical_symbols())

    # Identify atom types
    is_oxygen = symbols == "O"
    is_metal = ~is_oxygen

    # Pre-calculate normalized vectors for angle calculation
    # D is vector from i to j.
    norms = np.linalg.norm(D, axis=1)
    # Filter out zero length interactions (self) if any, though neighbor_list usually handles this
    valid = norms > 1e-6
    i_idx = i_idx[valid]
    j_idx = j_idx[valid]
    D = D[valid]
    norms = norms[valid]
    vecs = D / norms[:, np.newaxis]

    # Store angles
    mom_angles = []
    omo_angles = []

    # Iterate over all central atoms
    for center_atom_idx in range(len(atoms)):
        # Find neighbors of this center
        mask = i_idx == center_atom_idx
        if not np.any(mask):
            continue

        neigh_indices = j_idx[mask]
        neigh_vecs = vecs[mask]

        center_symbol = symbols[center_atom_idx]
        neigh_symbols = symbols[neigh_indices]

        # Determine motif type
        if center_symbol == "O":
            # Potential M-O-M angles. Neighbors should be metals.
            # Filter neighbors that are metals
            metal_mask = np.isin(neigh_symbols, ["Al", "Ga", "In"])
            valid_vecs = neigh_vecs[metal_mask]

            # Calculate angles between all pairs of metal neighbors
            n_v = len(valid_vecs)
            for k in range(n_v):
                for l in range(k + 1, n_v):
                    cos_theta = np.dot(valid_vecs[k], valid_vecs[l])
                    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                    mom_angles.append(angle)

        elif center_symbol in ["Al", "Ga", "In"]:
            # Potential O-M-O angles. Neighbors should be oxygens.
            o_mask = neigh_symbols == "O"
            valid_vecs = neigh_vecs[o_mask]

            n_v = len(valid_vecs)
            for k in range(n_v):
                for l in range(k + 1, n_v):
                    cos_theta = np.dot(valid_vecs[k], valid_vecs[l])
                    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                    omo_angles.append(angle)

    # Aggregate into features
    topo_features = {}

    for name, angle_list in [("M-O-M", mom_angles), ("O-M-O", omo_angles)]:
        if not angle_list:
            # Fill with 0 or NaN if no angles found (e.g. isolated atoms)
            for p in percentiles:
                topo_features[f"topo_{name}_p{p}"] = 0.0
            topo_features[f"topo_{name}_std"] = 0.0
        else:
            for p in percentiles:
                topo_features[f"topo_{name}_p{p}"] = np.percentile(angle_list, p)
            topo_features[f"topo_{name}_std"] = np.std(angle_list)

    return topo_features


def aggregate_site_features(
    atoms, econ, rad_strain, ang_strain, percentiles=[0, 25, 50, 75, 100]
):
    """
    Aggregates atom-level metrics by element type using percentiles.
    """
    symbols = np.array(atoms.get_chemical_symbols())
    unique_elements = ["Al", "Ga", "In", "O"]

    features = {}

    for el in unique_elements:
        mask = symbols == el

        if np.any(mask):
            el_econ = econ[mask]
            el_rad = rad_strain[mask]
            el_ang = ang_strain[mask]

            for p in percentiles:
                features[f"site_{el}_econ_p{p}"] = np.percentile(el_econ, p)
                features[f"site_{el}_rad_strain_p{p}"] = np.percentile(el_rad, p)
                features[f"site_{el}_ang_strain_p{p}"] = np.percentile(el_ang, p)
        else:
            # Element not present in this structure
            for p in percentiles:
                features[f"site_{el}_econ_p{p}"] = 0.0
                features[f"site_{el}_rad_strain_p{p}"] = 0.0
                features[f"site_{el}_ang_strain_p{p}"] = 0.0

    return features


def extract_features(metadata_df, load_cached_data=True):
    """
    Main function to extract features for a given metadata DataFrame.
    Implements caching logic.
    """
    # Determine cache file path based on dataset size (train/val/test distinction)
    # We use a hash of the IDs to ensure uniqueness, or simply map based on length/content.
    # For simplicity in this environment, we rely on the caller passing specific dfs (train, val, test).
    # However, since the function signature is fixed, we infer the split name or use a generic naming scheme
    # if we were building a pipeline. Here, we will save based on the number of rows to distinguish
    # (heuristic) or just use a standard name if provided.
    # Given the prompt constraints, we will construct a filename based on the first and last ID to be safe.

    ids = metadata_df["id"].values
    cache_filename = f"features_{ids[0]}_{ids[-1]}.parquet"
    cache_path = os.path.join(WORKING_DIR, cache_filename)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Computing features for {len(metadata_df)} samples...")

    feature_rows = []

    for idx, row in metadata_df.iterrows():
        # Load geometry
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        try:
            atoms = ase.io.read(full_path, format="aims")

            # 1. Physical Descriptors
            vol = atoms.get_volume()
            mass = sum(atoms.get_masses())
            density = mass / vol if vol > 1e-6 else 0.0

            feats = {
                "phys_volume": vol,
                "phys_density": density,
                "phys_natoms": len(atoms),
            }

            # 2. Radial Fingerprints
            rdf = compute_rdf(
                atoms,
                cutoff=FEATURE_PARAMS["rdf_cutoff"],
                bins=FEATURE_PARAMS["rdf_bins"],
                elements=FEATURE_PARAMS["elements"],
            )
            feats.update(rdf)

            # 3. Site-Resolved Strain
            econ, rad_strain, ang_strain = compute_site_metrics(
                atoms, cutoff=FEATURE_PARAMS["neighbor_cutoff"]
            )
            site_feats = aggregate_site_features(
                atoms,
                econ,
                rad_strain,
                ang_strain,
                percentiles=FEATURE_PARAMS["percentiles"],
            )
            feats.update(site_feats)

            # 4. Interaction Topology
            topo_feats = compute_topology_features(
                atoms,
                cutoff=FEATURE_PARAMS["angle_cutoff"],
                percentiles=FEATURE_PARAMS["percentiles"],
            )
            feats.update(topo_feats)

            # 5. Add Metadata (Composition, Spacegroup)
            # These are already in metadata_df but good to keep aligned if needed.
            # We will merge later or just append to the feature dict if we want them in the X matrix.
            # The prompt says "Input Features: ... Tabular Metadata".
            feats["meta_spacegroup"] = row["spacegroup"]
            feats["meta_percent_al"] = row["percent_atom_al"]
            feats["meta_percent_ga"] = row["percent_atom_ga"]
            feats["meta_percent_in"] = row["percent_atom_in"]
            # Lattice parameters
            feats["meta_lattice_a"] = row["lattice_vector_1_ang"]
            feats["meta_lattice_b"] = row["lattice_vector_2_ang"]
            feats["meta_lattice_c"] = row["lattice_vector_3_ang"]
            feats["meta_angle_alpha"] = row["lattice_angle_alpha_degree"]
            feats["meta_angle_beta"] = row["lattice_angle_beta_degree"]
            feats["meta_angle_gamma"] = row["lattice_angle_gamma_degree"]

            # Add ID for merging/tracking
            feats["id"] = row["id"]

            feature_rows.append(feats)

        except Exception as e:
            print(f"Error processing {rel_path}: {e}")
            # Add a row of NaNs or zeros to maintain alignment?
            # Better to skip and let the user handle alignment via 'id'.
            continue

    # Create DataFrame
    features_df = pd.DataFrame(feature_rows)

    # Save to cache
    print(f"Saving features to {cache_path}")
    features_df.to_parquet(cache_path)

    return features_df
