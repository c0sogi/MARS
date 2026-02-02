import os
import numpy as np
import pandas as pd
from ase.io import read
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

# Import configuration
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    RDF_CUTOFF,
    RDF_NUM_BINS,
    RDF_SIGMA,
    CTM_BOND_CUTOFF,
    CTM_ELEMENTS,
    RANDOM_SEED,
)

# Suppress warnings
warnings.filterwarnings("ignore")


def get_physical_descriptors(atoms):
    """
    Calculates basic physical properties of the unit cell.
    """
    vol = atoms.get_volume()
    mass = sum(atoms.get_masses())
    density = mass / vol if vol > 0 else 0.0
    return {"vol_per_atom": vol / len(atoms), "density": density}


def make_mic(diff_vector, cell):
    """
    Apply Minimum Image Convention to a vector given a cell.
    """
    try:
        cell_inv = np.linalg.inv(cell)
        frac = np.dot(diff_vector, cell_inv)
        frac -= np.round(frac)
        cart_mic = np.dot(frac, cell)
        return cart_mic
    except:
        return diff_vector


def calculate_econ_and_distortion(atoms):
    """
    Calculates Continuous Topological Moments: ECoN and Polyhedral Distortion.
    Aggregates by element type (Mean, Std).
    """
    positions = atoms.get_positions()
    cell = atoms.get_cell()

    # Get all distances using ASE's built-in MIC handling
    dist_matrix = atoms.get_all_distances(mic=True)
    chemical_symbols = np.array(atoms.get_chemical_symbols())

    # Initialize containers
    econ_values = {el: [] for el in CTM_ELEMENTS}
    distortion_values = {el: [] for el in CTM_ELEMENTS}

    # Iterate over all atoms
    for i, el in enumerate(chemical_symbols):
        if el not in CTM_ELEMENTS:
            continue

        # Get neighbors within cutoff
        dists = dist_matrix[i]
        # Exclude self (distance 0)
        mask = (dists > 0) & (dists <= CTM_BOND_CUTOFF)
        neighbor_indices = np.where(mask)[0]
        neighbor_dists = dists[neighbor_indices]

        if len(neighbor_dists) == 0:
            econ_values[el].append(0.0)
            distortion_values[el].append(0.0)
            continue

        # 1. Calculate ECoN
        # Formula: sum(exp(1 - (d_j / d_avg)^6))
        d_avg = np.mean(neighbor_dists)
        if d_avg > 0:
            terms = np.exp(1 - (neighbor_dists / d_avg) ** 6)
            econ = np.sum(terms)
        else:
            econ = 0.0
        econ_values[el].append(econ)

        # 2. Calculate Polyhedral Distortion
        if len(neighbor_indices) < 2:
            distortion_values[el].append(0.0)
            continue

        # Calculate vectors to neighbors manually to ensure correct MIC
        vecs = []
        for n_idx in neighbor_indices:
            diff = positions[n_idx] - positions[i]
            diff_mic = make_mic(diff, cell)
            vecs.append(diff_mic)

        vecs = np.array(vecs)

        # Normalize vectors
        norms = np.linalg.norm(vecs, axis=1)
        valid_norm_mask = norms > 1e-6
        vecs = vecs[valid_norm_mask]
        norms = norms[valid_norm_mask]

        if len(vecs) < 2:
            distortion_values[el].append(0.0)
            continue

        unit_vecs = vecs / norms[:, np.newaxis]

        # Compute angles (dot products)
        cos_angles = np.dot(unit_vecs, unit_vecs.T)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)

        # Extract upper triangle off-diagonal elements
        triu_indices = np.triu_indices(len(cos_angles), k=1)
        valid_cosines = cos_angles[triu_indices]

        angles_deg = np.degrees(np.arccos(valid_cosines))

        # Distortion: Min mean absolute deviation from Ideal Tetra (109.47) or Octa (90.0)
        if len(angles_deg) > 0:
            mad_tetra = np.mean(np.abs(angles_deg - 109.47))
            mad_octa = np.mean(np.abs(angles_deg - 90.0))
            distortion = min(mad_tetra, mad_octa)
        else:
            distortion = 0.0

        distortion_values[el].append(distortion)

    # Aggregate features
    features = {}
    for el in CTM_ELEMENTS:
        e_vals = econ_values[el]
        d_vals = distortion_values[el]

        if e_vals:
            features[f"CTM_ECoN_Mean_{el}"] = np.mean(e_vals)
            features[f"CTM_ECoN_Std_{el}"] = np.std(e_vals)
        else:
            features[f"CTM_ECoN_Mean_{el}"] = 0.0
            features[f"CTM_ECoN_Std_{el}"] = 0.0

        if d_vals:
            features[f"CTM_Distortion_Mean_{el}"] = np.mean(d_vals)
            features[f"CTM_Distortion_Std_{el}"] = np.std(d_vals)
        else:
            features[f"CTM_Distortion_Mean_{el}"] = 0.0
            features[f"CTM_Distortion_Std_{el}"] = 0.0

    return features


def compute_elemental_rdf(atoms):
    """
    Computes element-resolved Radial Distribution Functions.
    Focuses on Metal-Oxygen pairs: Al-O, Ga-O, In-O.
    """
    features = {}

    # Define pairs of interest
    pairs = [("Al", "O"), ("Ga", "O"), ("In", "O")]

    # Get all distances and symbols
    dist_matrix = atoms.get_all_distances(mic=True)
    symbols = np.array(atoms.get_chemical_symbols())

    bins = np.linspace(0, RDF_CUTOFF, RDF_NUM_BINS + 1)

    for el1, el2 in pairs:
        # Find indices
        idx1 = np.where(symbols == el1)[0]
        idx2 = np.where(symbols == el2)[0]

        if len(idx1) == 0 or len(idx2) == 0:
            hist_smooth = np.zeros(RDF_NUM_BINS)
        else:
            # Extract submatrix of distances
            sub_dists = dist_matrix[np.ix_(idx1, idx2)].flatten()

            # Filter
            sub_dists = sub_dists[(sub_dists <= RDF_CUTOFF) & (sub_dists > 0.1)]

            # Histogram
            hist, _ = np.histogram(sub_dists, bins=bins)

            # Normalize by number of atoms of type 1
            hist = hist.astype(float) / len(idx1)

            # Gaussian Smearing
            hist_smooth = gaussian_filter1d(hist, sigma=RDF_SIGMA)

        # Store in features dict
        for b in range(RDF_NUM_BINS):
            features[f"RDF_{el1}_{el2}_bin_{b}"] = hist_smooth[b]

    return features


def process_single_geometry(file_path):
    """
    Worker function to process one geometry file.
    """
    full_path = os.path.join(INPUT_DIR, file_path)

    try:
        atoms = read(full_path)

        # 1. Physical Descriptors
        phys_feats = get_physical_descriptors(atoms)

        # 2. CTM Features
        ctm_feats = calculate_econ_and_distortion(atoms)

        # 3. RDF Features
        rdf_feats = compute_elemental_rdf(atoms)

        # Combine
        return {**phys_feats, **ctm_feats, **rdf_feats}

    except Exception as e:
        return {}


def generate_features(metadata_df, dataset_name, load_cached_data=True):
    """
    Main function to generate features for a dataset (train/val/test).
    Handles caching.
    """
    cache_path = os.path.join(WORKING_DIR, f"{dataset_name}_features.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features for {dataset_name} from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Generating features for {dataset_name}...")

    # Extract file paths
    file_paths = metadata_df["file_path"].tolist()

    # Parallel processing
    results = Parallel(n_jobs=4, backend="loky")(
        delayed(process_single_geometry)(fp)
        for fp in tqdm(file_paths, desc=f"Processing {dataset_name}")
    )

    # Convert to DataFrame
    features_df = pd.DataFrame(results)

    # Add ID column to ensure alignment
    features_df["id"] = metadata_df["id"].values

    # Add tabular features from metadata (composition, lattice)
    exclude = ["formation_energy_ev_natom", "bandgap_energy_ev", "file_path", "id"]
    meta_feats = [c for c in metadata_df.columns if c not in exclude]

    # Concatenate
    final_df = pd.concat([features_df, metadata_df[meta_feats]], axis=1)

    # Handle NaNs
    final_df = final_df.fillna(0.0)

    # Cache
    print(f"Saving features to {cache_path}...")
    final_df.to_parquet(cache_path, index=False)

    return final_df
