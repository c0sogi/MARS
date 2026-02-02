import os
import numpy as np
import pandas as pd
import ase.neighborlist
from ase.geometry import analysis
from library.config import (
    CACHE_DIR,
    RDF_CUTOFF,
    RDF_BIN_WIDTH,
    RDF_BINS,
    PERCENTILES,
    RANDOM_SEED,
)
from library.data_loader import read_geometry
from collections import defaultdict


class GeometricDescriptor:
    def __init__(self):
        self.elements = ["Al", "Ga", "In", "O"]
        # Pairs for RDF and Bonds: Al-O, Ga-O, In-O, etc.
        # We mostly care about Metal-Oxygen, but let's be generic.
        self.pairs = []
        for i, e1 in enumerate(self.elements):
            for e2 in self.elements[i:]:
                self.pairs.append(tuple(sorted((e1, e2))))

        # Triplets for angles: O-M-O is most important.
        # We will capture distributions for Metal-centered angles (O-M-O)
        # and Oxygen-centered angles (M-O-M).
        self.bond_cutoff = 3.0  # Angstroms, for defining local coordination

    def compute_global_features(self, atoms):
        """
        Computes global features: Volume, Density, Packing Fraction (proxy).
        """
        vol = atoms.get_volume()
        mass = sum(atoms.get_masses())
        density = mass / vol if vol > 0 else 0.0

        # Number of atoms per element
        chemical_formula = atoms.get_chemical_formula(mode="hill")
        # We can just use the counts from atoms object
        symbols = atoms.get_chemical_symbols()
        counts = {e: symbols.count(e) for e in self.elements}
        total_atoms = len(atoms)

        features = {
            "global_volume": vol,
            "global_density": density,
            "global_num_atoms": total_atoms,
            "global_vol_per_atom": vol / total_atoms if total_atoms > 0 else 0,
        }
        for e in self.elements:
            features[f"global_count_{e}"] = counts[e]
            features[f"global_frac_{e}"] = (
                counts[e] / total_atoms if total_atoms > 0 else 0
            )

        return features

    def compute_rdf_features(self, atoms):
        """
        Computes Element-Resolved Radial Distribution Functions.
        """
        # neighbor_list returns (i, j, d, D)
        # i: index of atom 1, j: index of atom 2, d: distance
        i_list, j_list, d_list, _ = ase.neighborlist.neighbor_list(
            "ijdD", atoms, cutoff=RDF_CUTOFF
        )

        symbols = np.array(atoms.get_chemical_symbols())

        # Initialize RDF bins
        rdf_features = {}
        # We want RDF for each pair type
        # Bins
        bins = np.linspace(0, RDF_CUTOFF, RDF_BINS + 1)

        # Pre-calculate histograms for each pair
        for e1, e2 in self.pairs:
            # Mask for this pair
            # We need to consider that i_list and j_list contain both (i,j) and (j,i)
            # We want unique pairs or just handle symmetric consistently.
            # neighbor_list 'ijdD' gives both directions.

            mask_e1_i = symbols[i_list] == e1
            mask_e2_j = symbols[j_list] == e2

            if e1 == e2:
                # Same element, avoid double counting if we were iterating unique pairs,
                # but histogramming all distances is fine as long as we normalize.
                mask = mask_e1_i & mask_e2_j
            else:
                # Different elements, e.g. Al-O.
                # We want distances where (i is Al and j is O) OR (i is O and j is Al)
                mask = (mask_e1_i & mask_e2_j) | (
                    (symbols[i_list] == e2) & (symbols[j_list] == e1)
                )

            dists = d_list[mask]

            hist, _ = np.histogram(dists, bins=bins)

            # Normalize by volume and shell volume (standard RDF def) or just by atom count
            # Simple normalization: density of pairs at distance r
            # g(r) ~ count(r) / (4*pi*r^2 * dr * rho)
            # Here we just use the raw histogram counts normalized by total atoms to make it intensive-ish
            # This acts as a fingerprint.

            norm_hist = hist / len(atoms)

            for b_idx, val in enumerate(norm_hist):
                # Feature name: rdf_Al_O_bin_0, etc.
                feat_name = f"rdf_{e1}_{e2}_bin_{b_idx}"
                rdf_features[feat_name] = val

        return rdf_features

    def compute_interaction_features(self, atoms):
        """
        Computes percentiles of bond lengths and bond angles.
        Focuses on Metal-Oxygen bonds for angles.
        """
        # Get neighbors for bonding
        i_list, j_list, d_list, vector_list = ase.neighborlist.neighbor_list(
            "ijdD", atoms, cutoff=self.bond_cutoff
        )
        symbols = np.array(atoms.get_chemical_symbols())

        features = {}

        # --- Bond Lengths ---
        # Group distances by pair type
        pair_dists = defaultdict(list)
        for k in range(len(d_list)):
            idx_i, idx_j = i_list[k], j_list[k]
            e1, e2 = sorted((symbols[idx_i], symbols[idx_j]))
            pair_dists[f"{e1}_{e2}"].append(d_list[k])

        for pair_name, dists in pair_dists.items():
            if not dists:
                vals = [0.0] * len(PERCENTILES)
            else:
                vals = np.percentile(dists, PERCENTILES)

            for p, v in zip(PERCENTILES, vals):
                features[f"bond_dist_{pair_name}_p{p}"] = v

        # Fill missing pairs with 0
        all_pair_names = [f"{p[0]}_{p[1]}" for p in self.pairs]
        for name in all_pair_names:
            if f"bond_dist_{name}_p0" not in features:
                for p in PERCENTILES:
                    features[f"bond_dist_{name}_p{p}"] = 0.0

        # --- Bond Angles ---
        # We focus on X-Center-Y angles.
        # Construct an adjacency list
        adj = defaultdict(list)
        for k in range(len(d_list)):
            adj[i_list[k]].append((j_list[k], vector_list[k]))

        # Collect angles by triplet type (Center element, Neighbor1 element, Neighbor2 element)
        # We sort neighbors to canonicalize: Center=Al, N1=O, N2=O -> Al_O_O
        angle_groups = defaultdict(list)

        for center_idx, neighbors in adj.items():
            center_elem = symbols[center_idx]
            n_neighbors = len(neighbors)
            if n_neighbors < 2:
                continue

            # Calculate angles for all pairs of neighbors
            # This is O(N_neighbors^2), usually small (<100)
            for i in range(n_neighbors):
                for j in range(i + 1, n_neighbors):
                    n_idx1, vec1 = neighbors[i]
                    n_idx2, vec2 = neighbors[j]

                    # Angle calculation
                    # vec1 is vector from center to n1
                    # vec2 is vector from center to n2
                    # angle = arccos(dot(v1, v2) / (|v1||v2|))
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    if norm1 > 1e-6 and norm2 > 1e-6:
                        cosine = np.dot(vec1, vec2) / (norm1 * norm2)
                        # Clip for numerical stability
                        cosine = np.clip(cosine, -1.0, 1.0)
                        angle_deg = np.degrees(np.arccos(cosine))

                        e1 = symbols[n_idx1]
                        e2 = symbols[n_idx2]
                        # Canonical triplet name: Center_E1_E2 (where E1 <= E2)
                        n_syms = sorted([e1, e2])
                        triplet_name = f"{center_elem}_{n_syms[0]}_{n_syms[1]}"
                        angle_groups[triplet_name].append(angle_deg)

        # Compute percentiles for interesting triplets
        # We primarily care about Metal centered (O-M-O) and Oxygen centered (M-O-M)
        # But let's just process whatever we found to be safe, filtering for common ones if needed.
        # Given the dataset, Al, Ga, In are metals, O is oxygen.
        # Potential Triplets: Al_O_O, Ga_O_O, In_O_O (Polyhedral angles)
        # And O_Al_Al, O_Al_Ga, etc. (Connectivity angles)

        # List of expected triplets to ensure feature vector consistency
        # Metals: Al, Ga, In. Non-metal: O
        metals = ["Al", "Ga", "In"]
        expected_triplets = []
        # Metal centered
        for m in metals:
            expected_triplets.append(f"{m}_O_O")
        # Oxygen centered
        for i, m1 in enumerate(metals):
            for m2 in metals[i:]:
                expected_triplets.append(f"O_{m1}_{m2}")

        for trip in expected_triplets:
            angles = angle_groups.get(trip, [])
            if not angles:
                vals = [0.0] * len(PERCENTILES)
            else:
                vals = np.percentile(angles, PERCENTILES)

            for p, v in zip(PERCENTILES, vals):
                features[f"angle_{trip}_p{p}"] = v

        return features

    def compute_site_features(self, atoms):
        """
        Computes atom-centric features: Coordination Number and Angle Variance.
        Aggregated by element type percentiles.
        """
        # Re-use neighbor list logic for efficiency?
        # For clarity, we'll re-compute or assume we could pass it.
        # Let's recompute to keep methods decoupled.
        i_list, j_list, d_list, vector_list = ase.neighborlist.neighbor_list(
            "ijdD", atoms, cutoff=self.bond_cutoff
        )
        symbols = np.array(atoms.get_chemical_symbols())

        # Data structures to hold site metrics per element
        # element -> list of CNs, list of AngleVars
        site_cn = defaultdict(list)
        site_ang_var = defaultdict(list)

        # Build adjacency for angle variance
        adj = defaultdict(list)
        for k in range(len(d_list)):
            adj[i_list[k]].append(vector_list[k])

        # Iterate over all atoms
        for idx in range(len(atoms)):
            elem = symbols[idx]
            neighbors = adj[idx]
            cn = len(neighbors)
            site_cn[elem].append(cn)

            # Angle Variance
            if cn < 2:
                var_angle = 0.0
            else:
                angles = []
                for i in range(cn):
                    for j in range(i + 1, cn):
                        v1 = neighbors[i]
                        v2 = neighbors[j]
                        n1 = np.linalg.norm(v1)
                        n2 = np.linalg.norm(v2)
                        if n1 > 1e-6 and n2 > 1e-6:
                            c = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                            angles.append(np.degrees(np.arccos(c)))

                if angles:
                    # Variance of angles around this atom
                    var_angle = np.var(angles)
                else:
                    var_angle = 0.0

            site_ang_var[elem].append(var_angle)

        features = {}
        for elem in self.elements:
            # Coordination Number Percentiles
            cns = site_cn.get(elem, [])
            if not cns:
                cns_vals = [0.0] * len(PERCENTILES)
                ang_vals = [0.0] * len(PERCENTILES)
            else:
                cns_vals = np.percentile(cns, PERCENTILES)
                ang_vals = np.percentile(site_ang_var[elem], PERCENTILES)

            for p, v in zip(PERCENTILES, cns_vals):
                features[f"site_cn_{elem}_p{p}"] = v
            for p, v in zip(PERCENTILES, ang_vals):
                features[f"site_angvar_{elem}_p{p}"] = v

        return features


def extract_features(metadata_df, split_name, load_cached_data=True):
    """
    Main function to extract features for a dataset.
    Handles caching.
    """
    cache_file = os.path.join(CACHE_DIR, f"{split_name}_features.parquet")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}")
        return pd.read_parquet(cache_file)

    print(f"Computing features for {split_name} ({len(metadata_df)} samples)...")

    descriptor = GeometricDescriptor()
    feature_rows = []

    for _, row in metadata_df.iterrows():
        try:
            atoms = read_geometry(row["file_path"])

            # 1. Global
            feats = descriptor.compute_global_features(atoms)

            # 2. RDF
            feats.update(descriptor.compute_rdf_features(atoms))

            # 3. Interaction
            feats.update(descriptor.compute_interaction_features(atoms))

            # 4. Site
            feats.update(descriptor.compute_site_features(atoms))

            # Add ID for merging
            feats["id"] = row["id"]

            feature_rows.append(feats)

        except Exception as e:
            print(f"Error processing id {row.get('id', 'unknown')}: {e}")
            # Add empty/nan row or handle appropriately.
            # For this competition, we assume data quality is decent or we skip.
            # Better to add a row with NaNs to keep alignment if possible, but list append implies alignment by ID later.
            continue

    # Create DataFrame
    features_df = pd.DataFrame(feature_rows)

    # Merge with original metadata to keep tabular features
    # We assume 'id' is the key.
    # Metadata has columns like 'spacegroup', 'percent_atom_al', etc.
    # We want to keep those.

    # Drop file_path from metadata before merge to save space
    meta_clean = metadata_df.drop(columns=["file_path"], errors="ignore")

    # Merge
    full_df = pd.merge(meta_clean, features_df, on="id", how="left")

    # Save to cache
    print(f"Saving features to {cache_file}")
    full_df.to_parquet(cache_file, index=False)

    return full_df
