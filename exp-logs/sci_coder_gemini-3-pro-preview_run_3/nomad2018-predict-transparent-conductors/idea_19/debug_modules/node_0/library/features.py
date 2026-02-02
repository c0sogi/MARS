import os
import numpy as np
import pandas as pd
import ase.io
from ase.neighborlist import neighbor_list
from scipy.spatial.distance import pdist, squareform
import warnings

# Import configuration and constants
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    RANDOM_SEED,
    NEIGHBOR_CUTOFF,
    RDF_CUTOFF,
    RDF_BINS,
    INTERESTING_CNS,
    METALS,
    ATOMIC_SPECIES,
    get_dataset_size,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class FeatureExtractor:
    """
    Implements the Site-Specific Coordination Fingerprinting (SSCF) strategy
    along with Radial Distribution Functions and Physical Descriptors.
    """

    def __init__(self):
        self.rdf_bins = np.linspace(0, RDF_CUTOFF, RDF_BINS + 1)

    def get_global_descriptors(self, atoms):
        """
        Calculates global physical properties of the unit cell.
        """
        vol = atoms.get_volume()
        mass = sum(atoms.get_masses())
        density = mass / vol if vol > 0 else 0.0
        return {
            "cell_volume": vol,
            "mass_density": density,
            "number_of_atoms": len(atoms),
        }

    def get_rdf_features(self, atoms):
        """
        Computes element-resolved Radial Distribution Functions (RDF).
        Focuses on Metal-Oxygen pairs which are most structurally relevant.
        """
        features = {}

        # Ensure PBC is set for distance calculations
        atoms.set_pbc(True)

        # Get indices for each species
        indices = {s: [a.index for a in atoms if a.symbol == s] for s in ATOMIC_SPECIES}
        oxy_indices = indices.get("O", [])

        # Compute RDF for each Metal-Oxygen pair
        for metal in METALS:
            metal_indices = indices.get(metal, [])

            hist = np.zeros(RDF_BINS)
            if len(metal_indices) > 0 and len(oxy_indices) > 0:
                # Get distances from all metal atoms to all oxygen atoms
                # mic=True applies Minimum Image Convention
                dists = atoms.get_distances(
                    metal_indices, oxy_indices, mic=True
                ).flatten()

                # Filter by cutoff
                dists = dists[dists <= RDF_CUTOFF]

                # Compute histogram
                counts, _ = np.histogram(dists, bins=self.rdf_bins)

                # Normalize by the number of metal atoms to make it intensive
                hist = counts / len(metal_indices)

            # Store features
            for i, val in enumerate(hist):
                features[f"rdf_{metal}_O_bin_{i}"] = val

        return features

    def get_site_specific_fingerprints(self, atoms):
        """
        The Core Innovation: Site-Specific Coordination Fingerprinting (SSCF).
        Calculates the fraction of each metal species residing in specific
        coordination environments (CN=4, 5, 6).
        """
        features = {}

        # Ensure PBC
        atoms.set_pbc(True)

        # Calculate neighbor list once
        # i: central atom indices, j: neighbor indices
        # We only care about Metal-Oxygen bonds
        i_indices, j_indices = neighbor_list("ij", atoms, cutoff=NEIGHBOR_CUTOFF)

        # Map indices to symbols for fast lookup
        symbols = np.array(atoms.get_chemical_symbols())

        for metal in METALS:
            # Identify indices of this metal
            metal_mask = symbols == metal
            metal_idxs = np.where(metal_mask)[0]

            if len(metal_idxs) == 0:
                # If metal not present, all fractions are 0
                for cn in INTERESTING_CNS:
                    features[f"frac_{metal}_CN{cn}"] = 0.0
                continue

            # Filter neighbor list for this metal as center (i) and Oxygen as neighbor (j)
            # 1. Get neighbors of current metal atoms
            relevant_mask = np.isin(i_indices, metal_idxs)
            m_i = i_indices[relevant_mask]
            m_j = j_indices[relevant_mask]

            # 2. Check if neighbor is Oxygen
            # symbols[m_j] returns array of symbols for neighbors
            is_oxygen = symbols[m_j] == "O"

            # 3. Count Oxygen neighbors for each metal atom
            # We use bincount. The bins correspond to atom indices.
            # We need to be careful to index up to len(atoms)
            # valid_m_i contains indices of metal atoms that have at least one O neighbor
            valid_m_i = m_i[is_oxygen]

            if len(valid_m_i) > 0:
                coord_counts = np.bincount(valid_m_i, minlength=len(atoms))
                # Extract counts only for the metal atoms of interest
                metal_coord_counts = coord_counts[metal_idxs]
            else:
                metal_coord_counts = np.zeros(len(metal_idxs))

            # Calculate fractions for specific CNs
            total_metal_atoms = len(metal_idxs)
            for cn in INTERESTING_CNS:
                # Count how many atoms of this metal have this specific CN
                count_cn = np.sum(metal_coord_counts == cn)
                features[f"frac_{metal}_CN{cn}"] = count_cn / total_metal_atoms

        return features

    def get_distortion_metrics(self, atoms):
        """
        Calculates bond angle variance for metal atoms to capture polyhedral distortion.
        """
        features = {}
        atoms.set_pbc(True)

        # We need neighbors to compute angles
        # Using a slightly larger cutoff for angles can be safer, but let's stick to bonding cutoff
        nl = neighbor_list("ijD", atoms, cutoff=NEIGHBOR_CUTOFF)
        i_indices, j_indices, D_vectors = nl

        symbols = np.array(atoms.get_chemical_symbols())

        for metal in METALS:
            metal_idxs = np.where(symbols == metal)[0]

            if len(metal_idxs) == 0:
                features[f"var_angle_{metal}"] = 0.0
                continue

            variances = []

            for idx in metal_idxs:
                # Find neighbors of this atom
                mask = i_indices == idx
                neighbors = j_indices[mask]
                vectors = D_vectors[mask]

                # Filter for Oxygen neighbors
                is_oxy = symbols[neighbors] == "O"
                oxy_vectors = vectors[is_oxy]

                if len(oxy_vectors) < 2:
                    variances.append(0.0)
                    continue

                # Calculate all angles around this center
                # Cosine rule: a.b / (|a||b|)
                # Normalize vectors
                norms = np.linalg.norm(oxy_vectors, axis=1)
                # Avoid div by zero
                norms[norms == 0] = 1.0
                normalized_vecs = oxy_vectors / norms[:, np.newaxis]

                # Dot product of all pairs
                # (N, 3) @ (3, N) -> (N, N)
                cosine_matrix = np.dot(normalized_vecs, normalized_vecs.T)

                # Clip to valid range for arccos
                cosine_matrix = np.clip(cosine_matrix, -1.0, 1.0)

                # Get angles in degrees
                angles = np.degrees(np.arccos(cosine_matrix))

                # Extract upper triangle (unique pairs), excluding diagonal
                unique_angles = angles[np.triu_indices_from(angles, k=1)]

                if len(unique_angles) > 0:
                    variances.append(np.var(unique_angles))
                else:
                    variances.append(0.0)

            # Feature is the mean variance for this element type
            features[f"var_angle_{metal}"] = np.mean(variances) if variances else 0.0

        return features

    def compute_descriptors(self, atoms):
        """
        Aggregates all descriptors for a single atomic structure.
        """
        feats = {}
        # 1. Global
        feats.update(self.get_global_descriptors(atoms))
        # 2. RDF
        feats.update(self.get_rdf_features(atoms))
        # 3. Site-Specific Coordination
        feats.update(self.get_site_specific_fingerprints(atoms))
        # 4. Distortion
        feats.update(self.get_distortion_metrics(atoms))
        return feats


def generate_features(data_type, load_cached_data=True, debug=False):
    """
    Main function to generate or load features for a specific dataset split.

    Args:
        data_type (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to load from parquet if available.
        debug (bool): If True, process only a subset of data.

    Returns:
        pd.DataFrame: Feature matrix.
    """
    # Determine paths
    if data_type == "train":
        meta_path = TRAIN_METADATA_PATH
    elif data_type == "val":
        meta_path = VAL_METADATA_PATH
    elif data_type == "test":
        meta_path = TEST_METADATA_PATH
    else:
        raise ValueError("data_type must be 'train', 'val', or 'test'")

    cache_file = os.path.join(WORKING_DIR, f"{data_type}_features.parquet")

    # Try loading cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}...")
        return pd.read_parquet(cache_file)

    print(f"Generating features for {data_type} set...")

    # Load metadata
    df = pd.read_csv(meta_path)

    # Debug slicing
    if debug:
        limit = get_dataset_size(debug=True)
        if limit:
            df = df.iloc[:limit]
            print(f"Debug mode: processing {len(df)} samples.")

    extractor = FeatureExtractor()
    feature_list = []

    # Processing Loop
    for _, row in df.iterrows():
        # Construct full path to geometry file
        xyz_path = os.path.join(INPUT_DIR, row["file_path"])

        try:
            # Load atoms
            atoms = ase.io.read(xyz_path)

            # Compute features
            descriptors = extractor.compute_descriptors(atoms)

            # Add tabular metadata features
            # We exclude ID and targets here, they are managed separately or merged later
            descriptors["spacegroup"] = row["spacegroup"]
            descriptors["percent_atom_al"] = row["percent_atom_al"]
            descriptors["percent_atom_ga"] = row["percent_atom_ga"]
            descriptors["percent_atom_in"] = row["percent_atom_in"]
            descriptors["lattice_angle_alpha_degree"] = row[
                "lattice_angle_alpha_degree"
            ]
            descriptors["lattice_angle_beta_degree"] = row["lattice_angle_beta_degree"]
            descriptors["lattice_angle_gamma_degree"] = row[
                "lattice_angle_gamma_degree"
            ]

            # Add ID for merging safety
            descriptors["id"] = row["id"]

            # Add targets if available (train/val)
            if "formation_energy_ev_natom" in row:
                descriptors["formation_energy_ev_natom"] = row[
                    "formation_energy_ev_natom"
                ]
            if "bandgap_energy_ev" in row:
                descriptors["bandgap_energy_ev"] = row["bandgap_energy_ev"]

            feature_list.append(descriptors)

        except Exception as e:
            print(f"Error processing {xyz_path}: {e}")
            continue

    # Convert to DataFrame
    features_df = pd.DataFrame(feature_list)

    # Save to cache
    if not debug:
        print(f"Saving features to {cache_file}...")
        features_df.to_parquet(cache_file, index=False)

    return features_df
