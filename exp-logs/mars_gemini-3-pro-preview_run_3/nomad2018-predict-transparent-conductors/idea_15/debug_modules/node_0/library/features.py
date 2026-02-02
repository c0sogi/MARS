import os
import numpy as np
import pandas as pd
import ase.io
from ase.neighborlist import neighbor_list
from scipy.special import sph_harm
from library.config import Config


class StructureFeaturizer:
    """
    Extracts physical, radial, and symmetry-based features from atomic structures.
    Implements Steinhardt Bond Orientational Order Parameters and Element-Resolved RDFs.
    """

    def __init__(self):
        self.cations = Config.CATIONS
        self.anions = Config.ANIONS
        self.all_elements = Config.ALL_ELEMENTS
        self.rdf_cutoff = Config.RDF_CUTOFF
        self.rdf_bins = Config.RDF_NUM_BINS
        self.steinhardt_l = Config.STEINHARDT_L
        self.steinhardt_cutoff = Config.STEINHARDT_CUTOFF

        # Pre-compute RDF bin edges
        self.rdf_bin_edges = np.linspace(0, self.rdf_cutoff, self.rdf_bins + 1)
        self.rdf_bin_centers = (self.rdf_bin_edges[:-1] + self.rdf_bin_edges[1:]) / 2

    def process_single_structure(self, file_path):
        """
        Computes all features for a single geometry file.
        """
        try:
            full_path = os.path.join(Config.INPUT_DIR, file_path)
            atoms = ase.io.read(full_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

        features = {}

        # 1. Physical Properties
        phys_feats = self.compute_physical_properties(atoms)
        features.update(phys_feats)

        # 2. Radial Distribution Functions (RDF)
        rdf_feats = self.compute_rdf(atoms)
        features.update(rdf_feats)

        # 3. Steinhardt Symmetry Features
        steinhardt_feats = self.compute_steinhardt_ql(atoms)
        features.update(steinhardt_feats)

        return features

    def compute_physical_properties(self, atoms):
        """
        Computes basic physical properties: Volume, Density, Number of Atoms.
        """
        vol = atoms.get_volume()
        mass = sum(atoms.get_masses())
        # Density in amu / Angstrom^3
        density = mass / vol if vol > 0 else 0.0

        return {
            "vol_per_atom": vol / len(atoms),
            "density": density,
            "num_atoms": len(atoms),
        }

    def compute_rdf(self, atoms):
        """
        Computes element-resolved Radial Distribution Functions.
        Pairs: Cation-Anion (Al-O, Ga-O, In-O) and Cation-Cation.
        """
        features = {}

        # Get all neighbors within cutoff
        # i: center atom indices, j: neighbor indices, d: distances
        i_indices, j_indices, d_values = neighbor_list("ijd", atoms, self.rdf_cutoff)

        chemical_symbols = np.array(atoms.get_chemical_symbols())

        # Define pairs of interest
        # We focus on Metal-Oxygen and Metal-Metal interactions
        pairs_to_compute = []
        for cat in self.cations:
            pairs_to_compute.append((cat, "O"))  # Metal-Oxygen
            for cat2 in self.cations:
                # Lexicographical order to avoid duplicates (e.g. Al-Ga and Ga-Al)
                if cat <= cat2:
                    pairs_to_compute.append((cat, cat2))

        # Compute RDF for each pair
        for el1, el2 in pairs_to_compute:
            # Mask for specific element pair
            # We need bonds where atom i is el1 and atom j is el2 (or vice versa)

            mask_i_el1 = chemical_symbols[i_indices] == el1
            mask_j_el2 = chemical_symbols[j_indices] == el2

            # Since neighbor list is i->j, we consider directed pairs.
            # For A-B RDF, we want distances between A and B.
            # If el1 == el2, we just take mask_i & mask_j.
            # If el1 != el2, we take (i==A & j==B) | (i==B & j==A).

            if el1 == el2:
                pair_mask = mask_i_el1 & mask_j_el2
            else:
                mask_i_el2 = chemical_symbols[i_indices] == el2
                mask_j_el1 = chemical_symbols[j_indices] == el1
                pair_mask = (mask_i_el1 & mask_j_el2) | (mask_i_el2 & mask_j_el1)

            d_pair = d_values[pair_mask]

            # Histogram
            hist, _ = np.histogram(d_pair, bins=self.rdf_bin_edges)

            # Normalize by total number of atoms to make it intensive-like
            # (Standard RDF normalization by volume shell is good, but simple count/N is robust for ML)
            norm_hist = hist / len(atoms)

            # Store features
            for k, val in enumerate(norm_hist):
                features[f"RDF_{el1}_{el2}_bin_{k}"] = val

        return features

    def compute_steinhardt_ql(self, atoms):
        """
        Computes chemically aggregated Steinhardt Order Parameters (Q4, Q6).
        Focuses on the coordination environment of Cations (surrounded by Anions).
        """
        features = {}

        # Get neighbors for Steinhardt (usually 1st coordination shell ~3.0A)
        # i: center, j: neighbor, D: vector r_j - r_i
        i_idx, j_idx, D_vecs = neighbor_list("ijD", atoms, self.steinhardt_cutoff)

        chemical_symbols = np.array(atoms.get_chemical_symbols())

        # Containers for Ql values per element
        ql_values = {el: {l: [] for l in self.steinhardt_l} for el in self.cations}

        # Iterate over each atom in the unit cell
        num_atoms = len(atoms)
        for atom_i in range(num_atoms):
            symbol = chemical_symbols[atom_i]

            # We only care about Cation environments (geometry around Al, Ga, In)
            if symbol not in self.cations:
                continue

            # Find neighbors of atom_i
            # The neighbor list arrays are sorted by i, but not strictly grouped.
            # We use boolean mask.
            mask_neighbors = i_idx == atom_i

            if not np.any(mask_neighbors):
                # No neighbors within cutoff
                for l in self.steinhardt_l:
                    ql_values[symbol][l].append(0.0)
                continue

            # Vectors to neighbors
            vectors = D_vecs[mask_neighbors]
            neighbor_indices = j_idx[mask_neighbors]
            neighbor_symbols = chemical_symbols[neighbor_indices]

            # Filter neighbors: We define coordination polyhedron by surrounding Oxygens
            # (or all atoms if we want general environment, but O is physically motivated)
            mask_anions = neighbor_symbols == "O"
            if not np.any(mask_anions):
                # No Oxygen neighbors
                for l in self.steinhardt_l:
                    ql_values[symbol][l].append(0.0)
                continue

            vectors = vectors[mask_anions]

            # Convert to spherical coordinates (r, theta, phi)
            # scipy sph_harm takes (m, n, theta=azimuthal, phi=polar)
            # Physics convention: theta=polar (0..pi), phi=azimuthal (0..2pi)
            # Scipy convention: theta=azimuthal [0, 2pi], phi=polar [0, pi]

            x = vectors[:, 0]
            y = vectors[:, 1]
            z = vectors[:, 2]
            r = np.sqrt(x**2 + y**2 + z**2)

            # Polar angle (0 to pi)
            phi_polar = np.arccos(np.clip(z / r, -1.0, 1.0))
            # Azimuthal angle (0 to 2pi)
            theta_azimuthal = np.arctan2(y, x)
            # Map to [0, 2pi]
            theta_azimuthal[theta_azimuthal < 0] += 2 * np.pi

            # Compute Q_l
            for l in self.steinhardt_l:
                # Sum over m = -l to l
                q_lm_sum = 0.0
                for m in range(-l, l + 1):
                    # Average Y_lm over all neighbors k
                    # sph_harm(m, n, theta_azimuthal, phi_polar)
                    y_lm = sph_harm(m, l, theta_azimuthal, phi_polar)
                    y_lm_avg = np.mean(y_lm)
                    q_lm_sum += np.abs(y_lm_avg) ** 2

                Q_l = np.sqrt(4 * np.pi / (2 * l + 1) * q_lm_sum)
                ql_values[symbol][l].append(Q_l)

        # Aggregate features (Mean, Std) for each cation type
        for el in self.cations:
            for l in self.steinhardt_l:
                vals = ql_values[el][l]
                if len(vals) > 0:
                    features[f"Steinhardt_Q{l}_{el}_mean"] = np.mean(vals)
                    features[f"Steinhardt_Q{l}_{el}_std"] = np.std(vals)
                else:
                    features[f"Steinhardt_Q{l}_{el}_mean"] = 0.0
                    features[f"Steinhardt_Q{l}_{el}_std"] = 0.0

        return features


def process_dataset(
    metadata_path, cache_file, load_cached_data=True, debug_sample=None
):
    """
    Main function to process a dataset (train/val/test).
    Handles caching, feature extraction, and dataframe construction.
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

    # Debugging: sample subset
    if debug_sample is not None:
        df = df.head(debug_sample)
        print(f"Debug mode: processing first {debug_sample} rows.")

    featurizer = StructureFeaturizer()

    feature_list = []
    ids = []

    # Iterate and process
    for idx, row in df.iterrows():
        feats = featurizer.process_single_structure(row["file_path"])
        if feats is not None:
            # Add metadata features that are already in the csv but useful to keep
            # We keep everything from metadata and append new features
            # Actually, let's just append new features to a list and merge later
            feature_list.append(feats)
            ids.append(row["id"])
        else:
            # If processing fails, we might need to handle it.
            # For now, append empty or zeroed features to maintain alignment?
            # Better to skip and filter df later, but alignment is critical.
            # We will assume data quality allows processing.
            pass

    # Create DataFrame from new features
    features_df = pd.DataFrame(feature_list)
    features_df["id"] = ids

    # Merge with original metadata to keep targets and composition features
    # Ensure 'id' is the key
    merged_df = pd.merge(df, features_df, on="id", how="inner")

    # Save to cache
    print(f"Saving features to {cache_path}")
    merged_df.to_parquet(cache_path, index=False)

    return merged_df
