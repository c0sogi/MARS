import numpy as np
import pandas as pd
from ase import Atoms
from ase.neighborlist import neighbor_list
from library.config import (
    ELECTRONEGATIVITY,
    BVS_PARAMS,
    RDF_CUTOFF,
    RDF_BINS,
    BOND_CUTOFF,
    PERCENTILES,
)


class StructureDescriptor:
    def __init__(self):
        self.elements = ["Al", "Ga", "In", "O"]
        self.metals = ["Al", "Ga", "In"]
        self.anion = "O"
        self.percentiles = PERCENTILES

    def extract(self, atoms: Atoms) -> dict:
        """
        Extracts all physics-based features for a given ASE Atoms object.
        """
        features = {}

        # 1. Macroscopic Features
        features.update(self._get_macroscopic(atoms))

        # 2. Radial Distribution Features (RDF)
        features.update(self._get_rdf(atoms))

        # Pre-compute neighbor list for interactions and atomic states
        # 'i': atom index, 'j': neighbor index, 'd': distance, 'D': vector
        # self_interaction=False to avoid counting atom with itself
        i_list, j_list, d_list, D_list = neighbor_list(
            "ijdD", atoms, cutoff=BOND_CUTOFF, self_interaction=False
        )

        # 3. Interaction Distributions (Bonds & Angles)
        features.update(
            self._get_interaction_dist(atoms, i_list, j_list, d_list, D_list)
        )

        # 4. Atomic State Distributions (BVS, ECoN)
        features.update(self._get_atomic_state_dist(atoms, i_list, j_list, d_list))

        return features

    def _get_macroscopic(self, atoms: Atoms) -> dict:
        vol = atoms.get_volume()
        mass = sum(atoms.get_masses())
        # Avoid division by zero
        density = mass / vol if vol > 1e-6 else 0.0

        # Composition
        chemical_symbols = np.array(atoms.get_chemical_symbols())
        n_atoms = len(chemical_symbols)
        comp_feats = {}
        for el in self.elements:
            comp_feats[f"comp_frac_{el}"] = (
                np.sum(chemical_symbols == el) / n_atoms if n_atoms > 0 else 0.0
            )

        return {
            "macro_volume_per_atom": vol / n_atoms if n_atoms > 0 else 0.0,
            "macro_density": density,
            **comp_feats,
        }

    def _get_rdf(self, atoms: Atoms) -> dict:
        """
        Computes element-resolved RDFs.
        Focuses on Metal-Oxygen pairs as they are the primary interactions.
        """
        feats = {}
        chemical_symbols = np.array(atoms.get_chemical_symbols())

        # Define pairs of interest (Metal-Oxygen)
        pairs = [(m, "O") for m in self.metals]

        # Get all distances up to RDF_CUTOFF
        # We use neighbor_list with a larger cutoff for RDF
        i_rdf, j_rdf, d_rdf = neighbor_list(
            "ijd", atoms, cutoff=RDF_CUTOFF, self_interaction=False
        )

        # Create histogram bins
        bins = np.linspace(0, RDF_CUTOFF, RDF_BINS + 1)

        if len(d_rdf) == 0:
            # Return zeros if no neighbors found
            for m, o in pairs:
                for b in range(RDF_BINS):
                    feats[f"rdf_{m}-{o}_bin{b}"] = 0.0
            return feats

        for m, o in pairs:
            # Mask for specific pair type
            # neighbor_list returns both i-j and j-i.
            # We filter where i is Metal and j is Oxygen to capture the M-O distribution.
            mask = (chemical_symbols[i_rdf] == m) & (chemical_symbols[j_rdf] == o)
            d_pair = d_rdf[mask]

            hist, _ = np.histogram(d_pair, bins=bins)

            # Normalize by number of Metal atoms to make it intensive
            n_m = np.sum(chemical_symbols == m)
            if n_m > 0:
                hist = hist / n_m
            else:
                hist = hist * 0.0

            for b in range(RDF_BINS):
                feats[f"rdf_{m}-{o}_bin{b}"] = hist[b]

        return feats

    def _get_interaction_dist(self, atoms, i_list, j_list, d_list, D_list) -> dict:
        feats = {}
        chemical_symbols = np.array(atoms.get_chemical_symbols())

        # --- Bond Lengths ---
        # Group by pair type
        for m in self.metals:
            # Filter for M-O bonds
            mask = (chemical_symbols[i_list] == m) & (chemical_symbols[j_list] == "O")
            d_mo = d_list[mask]

            prefix = f"bond_dist_{m}-O"
            if len(d_mo) > 0:
                res = np.percentile(d_mo, self.percentiles)
            else:
                res = np.zeros(len(self.percentiles))

            for p, val in zip(self.percentiles, res):
                feats[f"{prefix}_p{p}"] = val

        # --- Angles ---
        # We need to reconstruct adjacency to calculate angles efficiently
        # Adjacency list: atom_idx -> list of (neighbor_idx, vector_to_neighbor, distance)
        # Using a list of lists for adjacency
        n_atoms = len(atoms)
        adj = [[] for _ in range(n_atoms)]

        for idx, i in enumerate(i_list):
            adj[i].append((j_list[idx], D_list[idx], d_list[idx]))

        # O-M-O Angles (centered on Metal)
        # M-O-M Angles (centered on Oxygen)

        omo_angles = {m: [] for m in self.metals}
        mom_angles = []

        for center_idx in range(n_atoms):
            center_sym = chemical_symbols[center_idx]
            neighbors = adj[center_idx]

            if len(neighbors) < 2:
                continue

            # Check if center is Metal or Oxygen
            is_metal = center_sym in self.metals
            is_oxygen = center_sym == "O"

            if not (is_metal or is_oxygen):
                continue

            # Filter relevant neighbors
            # For O-M-O (Center M), neighbors must be O
            # For M-O-M (Center O), neighbors must be M
            valid_neighbors = []
            for n_idx, n_vec, n_dist in neighbors:
                n_sym = chemical_symbols[n_idx]
                if is_metal and n_sym == "O":
                    valid_neighbors.append((n_vec, n_dist))
                elif is_oxygen and n_sym in self.metals:
                    valid_neighbors.append((n_vec, n_dist))

            if len(valid_neighbors) < 2:
                continue

            # Compute angles for all unique pairs of valid neighbors
            # Iterate k < l
            for k in range(len(valid_neighbors)):
                for l in range(k + 1, len(valid_neighbors)):
                    v1, d1 = valid_neighbors[k]
                    v2, d2 = valid_neighbors[l]

                    # Cosine rule: cos(theta) = (v1 . v2) / (|v1| |v2|)
                    dot_prod = np.dot(v1, v2)
                    cosine = dot_prod / (d1 * d2)
                    # Clip for numerical stability
                    cosine = np.clip(cosine, -1.0, 1.0)
                    angle_deg = np.degrees(np.arccos(cosine))

                    if is_metal:
                        omo_angles[center_sym].append(angle_deg)
                    else:
                        mom_angles.append(angle_deg)

        # Compute percentiles for O-M-O
        for m in self.metals:
            angles = np.array(omo_angles[m])
            prefix = f"angle_dist_O-{m}-O"
            if len(angles) > 0:
                res = np.percentile(angles, self.percentiles)
            else:
                res = np.zeros(len(self.percentiles))
            for p, val in zip(self.percentiles, res):
                feats[f"{prefix}_p{p}"] = val

        # Compute percentiles for M-O-M
        angles = np.array(mom_angles)
        prefix = "angle_dist_M-O-M"
        if len(angles) > 0:
            res = np.percentile(angles, self.percentiles)
        else:
            res = np.zeros(len(self.percentiles))
        for p, val in zip(self.percentiles, res):
            feats[f"{prefix}_p{p}"] = val

        return feats

    def _get_atomic_state_dist(self, atoms, i_list, j_list, d_list) -> dict:
        feats = {}
        chemical_symbols = np.array(atoms.get_chemical_symbols())
        n_atoms = len(atoms)

        # Initialize arrays to store per-atom properties
        bvs_values = np.zeros(n_atoms)
        econ_values = np.zeros(n_atoms)

        # If no bonds, return zeros
        if len(i_list) == 0:
            for el in self.elements:
                for p in self.percentiles:
                    feats[f"atomic_BVS_{el}_p{p}"] = 0.0
                    feats[f"atomic_ECoN_{el}_p{p}"] = 0.0
            return feats

        # Create a dataframe for neighbor interactions to leverage pandas groupby
        df_neighbors = pd.DataFrame(
            {
                "i": i_list,
                "j": j_list,
                "d": d_list,
                "sym_i": chemical_symbols[i_list],
                "sym_j": chemical_symbols[j_list],
            }
        )

        # --- Bond Valence Sum (BVS) ---
        # Formula: V_i = sum_j exp((R0 - d_ij) / b)
        # We only calculate BVS for M-O interactions.

        # Pre-calculate R0 map
        r0_map = {}
        for m in self.metals:
            # R0 for Metal cation
            r0_map[f"{m}-O"] = BVS_PARAMS["R0"][m]
            # R0 for Oxygen anion (symmetric)
            r0_map[f"O-{m}"] = BVS_PARAMS["R0"][m]

        df_neighbors["pair_key"] = df_neighbors["sym_i"] + "-" + df_neighbors["sym_j"]
        df_neighbors["R0"] = df_neighbors["pair_key"].map(r0_map)

        # Filter valid BVS pairs (non-NaN R0)
        valid_bvs = df_neighbors.dropna(subset=["R0"]).copy()
        if not valid_bvs.empty:
            valid_bvs["valence"] = np.exp(
                (valid_bvs["R0"] - valid_bvs["d"]) / BVS_PARAMS["b"]
            )
            bvs_sums = valid_bvs.groupby("i")["valence"].sum()
            # Map back to the full atom array
            # bvs_sums index corresponds to atom index
            # We use reindex to ensure all atoms are present (fill missing with 0)
            bvs_values = bvs_sums.reindex(range(n_atoms), fill_value=0.0).values

        # --- Effective Coordination Number (ECoN) ---
        # Hoppe's definition: ECoN_i = sum_j exp(1 - (d_ij / d_min_i)^6)
        # First, find d_min for each atom i
        min_dists = df_neighbors.groupby("i")["d"].min()
        df_neighbors["d_min"] = df_neighbors["i"].map(min_dists)

        # Calculate term
        # exp(1 - (d/d_min)^6)
        # Note: d >= d_min, so ratio >= 1, term <= 1.
        # If d=d_min, term=1.
        df_neighbors["econ_term"] = np.exp(
            1 - (df_neighbors["d"] / df_neighbors["d_min"]) ** 6
        )

        econ_sums = df_neighbors.groupby("i")["econ_term"].sum()
        econ_values = econ_sums.reindex(range(n_atoms), fill_value=0.0).values

        # --- Aggregate Distributions per Element ---
        for el in self.elements:
            mask = chemical_symbols == el

            # BVS percentiles
            vals_bvs = bvs_values[mask]
            prefix_bvs = f"atomic_BVS_{el}"
            if len(vals_bvs) > 0:
                res = np.percentile(vals_bvs, self.percentiles)
            else:
                res = np.zeros(len(self.percentiles))
            for p, val in zip(self.percentiles, res):
                feats[f"{prefix_bvs}_p{p}"] = val

            # ECoN percentiles
            vals_econ = econ_values[mask]
            prefix_econ = f"atomic_ECoN_{el}"
            if len(vals_econ) > 0:
                res = np.percentile(vals_econ, self.percentiles)
            else:
                res = np.zeros(len(self.percentiles))
            for p, val in zip(self.percentiles, res):
                feats[f"{prefix_econ}_p{p}"] = val

        return feats
