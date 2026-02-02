import os
import numpy as np
import pandas as pd
import ase.io
from ase import Atoms
from ase.neighborlist import neighbor_list
from scipy.special import erfc
from scipy.spatial.distance import pdist, squareform

# Import configuration and helper functions
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    SUBMISSION_PATH,
    OXIDATION_STATES,
    BVS_R0,
    BVS_B,
    train_and_predict,
)
from library.data_loader import load_metadata


class ElectrostaticsCalculator:
    """
    Calculator for physics-based electrostatic and geometric descriptors.
    Implements Ewald summation for Madelung energy and Bond Valence Vector Sum analysis.
    """

    def __init__(self, ewald_eta=None, ewald_cutoff=12.0):
        """
        Args:
            ewald_eta (float): Screening parameter for Ewald summation. If None, estimated from volume.
            ewald_cutoff (float): Cutoff radius for real-space interactions in Angstrom.
        """
        self.ewald_eta = ewald_eta
        self.ewald_cutoff = ewald_cutoff

    def calculate_madelung_energy(self, atoms: Atoms) -> float:
        """
        Calculates the Madelung energy using Ewald summation.

        E_total = E_real + E_recip - E_self

        Returns:
            float: Madelung energy per atom (eV).
        """
        charges = np.array(
            [OXIDATION_STATES.get(s, 0.0) for s in atoms.get_chemical_symbols()]
        )
        positions = atoms.get_positions()
        cell = atoms.get_cell()
        volume = atoms.get_volume()
        n_atoms = len(atoms)

        if volume < 1e-3:
            return 0.0

        # Determine screening parameter eta if not provided
        # Heuristic: eta ~ sqrt(pi) / cutoff or similar.
        # Common choice: eta such that real space converges at cutoff.
        eta = self.ewald_eta if self.ewald_eta else 5.0 / (volume ** (1 / 3))

        # 1. Real Space Summation
        # Use ASE neighbor list for periodic boundary conditions
        nl_i, nl_j, nl_d = neighbor_list("ijd", atoms, cutoff=self.ewald_cutoff)

        # Filter out self-interactions (i == j and distance ~ 0)
        mask = nl_d > 1e-3
        nl_i = nl_i[mask]
        nl_j = nl_j[mask]
        nl_d = nl_d[mask]

        # E_real = 0.5 * sum (qi * qj * erfc(eta * rij) / rij)
        real_term = 0.0
        if len(nl_d) > 0:
            factors = (charges[nl_i] * charges[nl_j] * erfc(eta * nl_d)) / nl_d
            real_term = 0.5 * np.sum(factors)

        # 2. Reciprocal Space Summation
        # Generate G-vectors
        recip_cell = atoms.get_reciprocal_cell()
        # Cutoff for G-vectors: G^2 / (4 * eta^2) < tolerance
        g_cutoff = 2 * eta * np.sqrt(-np.log(1e-6))  # Tolerance 1e-6

        # Grid of integers for G vectors
        n_max = (
            int(np.ceil(g_cutoff * np.max(np.linalg.norm(cell, axis=1)) / (2 * np.pi)))
            + 1
        )
        gx = np.arange(-n_max, n_max + 1)
        gy = np.arange(-n_max, n_max + 1)
        gz = np.arange(-n_max, n_max + 1)

        g_grid = np.array(np.meshgrid(gx, gy, gz)).T.reshape(-1, 3)
        # Exclude (0,0,0)
        g_grid = g_grid[np.sum(g_grid**2, axis=1) > 0]

        # Convert to cartesian G vectors: G = 2*pi * (h*b1 + k*b2 + l*b3)
        # ASE get_reciprocal_cell returns vectors without 2*pi factor usually, check convention
        # ASE: recip_cell * 2 * pi are the actual G vectors
        G_vecs = g_grid @ recip_cell * 2 * np.pi
        G2 = np.sum(G_vecs**2, axis=1)

        # Structure factor S(G) = sum_j q_j * exp(i * G . r_j)
        # Vectorized dot product: (N_G, 3) . (3, N_atoms) -> (N_G, N_atoms)
        Gr = G_vecs @ positions.T
        structure_factors = np.sum(charges * np.exp(1j * Gr), axis=1)
        S_sq = (np.abs(structure_factors)) ** 2

        # E_recip = (2*pi / V) * sum ( exp(-G^2 / 4eta^2) / G^2 * |S(G)|^2 )
        recip_factors = (np.exp(-G2 / (4 * eta**2)) / G2) * S_sq
        recip_term = (2 * np.pi / volume) * np.sum(recip_factors)

        # 3. Self Interaction Correction
        # E_self = (eta / sqrt(pi)) * sum(q_i^2)
        self_term = (eta / np.sqrt(np.pi)) * np.sum(charges**2)

        # Total Energy
        total_energy = real_term + recip_term - self_term

        return total_energy / n_atoms

    def calculate_bvs_features(self, atoms: Atoms) -> dict:
        """
        Calculates Bond Valence Sum (BVS) and Bond Valence Vector Sum (BVVS) features.
        Returns percentiles for each element type.
        """
        features = {}
        symbols = np.array(atoms.get_chemical_symbols())
        unique_elements = ["Al", "Ga", "In", "O"]

        # Neighbor list for bonds
        cutoff = 4.0
        nl_i, nl_j, nl_D = neighbor_list("ijD", atoms, cutoff=cutoff)
        nl_d = np.linalg.norm(nl_D, axis=1)

        n_atoms = len(atoms)
        bvs = np.zeros(n_atoms)
        bvvs = np.zeros((n_atoms, 3))

        # Calculate BVS terms
        for k, (i, j) in enumerate(zip(nl_i, nl_j)):
            el_i = symbols[i]
            el_j = symbols[j]
            dist = nl_d[k]

            # Look up R0 parameter
            key = (el_i, el_j)
            if key in BVS_R0 and dist > 0.1:
                r0 = BVS_R0[key]
                val = np.exp((r0 - dist) / BVS_B)

                bvs[i] += val

                # Vector sum (weighted direction)
                # D points from j to i in ASE neighbor_list 'ijD' convention?
                # Actually D is vector pointing from atom i to atom j.
                # We want vector characterizing environment of i.
                bvvs[i] += val * (nl_D[k] / dist)

        bvvs_mag = np.linalg.norm(bvvs, axis=1)

        # Global Instability Index
        ideal_valences = np.array([abs(OXIDATION_STATES.get(s, 0)) for s in symbols])
        gii = np.sqrt(np.mean((bvs - ideal_valences) ** 2))
        features["GII"] = gii

        # Distributional Features (Percentiles)
        percentiles = [0, 25, 50, 75, 100]

        for elem in unique_elements:
            mask = symbols == elem
            if np.sum(mask) > 0:
                elem_bvs = bvs[mask]
                elem_bvvs = bvvs_mag[mask]

                # BVS Percentiles
                p_bvs = np.percentile(elem_bvs, percentiles)
                for p, val in zip(percentiles, p_bvs):
                    features[f"BVS_{elem}_p{p}"] = val

                # BVVS Percentiles
                p_bvvs = np.percentile(elem_bvvs, percentiles)
                for p, val in zip(percentiles, p_bvvs):
                    features[f"BVVS_{elem}_p{p}"] = val
            else:
                # Fill with NaN if element not present
                for p in percentiles:
                    features[f"BVS_{elem}_p{p}"] = np.nan
                    features[f"BVVS_{elem}_p{p}"] = np.nan

        return features

    def calculate_geometric_features(self, atoms: Atoms) -> dict:
        """
        Calculates geometric features like density, volume, and coordination environment stats.
        """
        features = {}
        vol = atoms.get_volume()
        mass = sum(atoms.get_masses())
        n_atoms = len(atoms)

        features["density"] = mass / vol if vol > 0 else 0
        features["vol_per_atom"] = vol / n_atoms

        # Cell parameters
        cell_params = atoms.get_cell_lengths_and_angles()
        for i, label in enumerate(["a", "b", "c", "alpha", "beta", "gamma"]):
            features[f"cell_{label}"] = cell_params[i]

        # Composition
        symbols = atoms.get_chemical_symbols()
        for elem in ["Al", "Ga", "In", "O"]:
            features[f"frac_{elem}"] = symbols.count(elem) / n_atoms

        return features


def extract_all_features(
    metadata_df: pd.DataFrame, split_name: str, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Extracts Electro-Geometric Distributional features for a dataset.
    Handles caching to parquet.
    """
    cache_path = os.path.join(WORKING_DIR, f"{split_name}_egdf_features.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached EGDF features for {split_name} from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Generating EGDF features for {split_name}...")

    calculator = ElectrostaticsCalculator()
    features_list = []

    for idx, row in metadata_df.iterrows():
        # Read atoms
        file_path = os.path.join(INPUT_DIR, row["file_path"])
        try:
            atoms = ase.io.read(file_path, format="aims")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Extract features
        feats = {"id": row["id"]}

        # 1. Madelung
        feats["madelung_energy"] = calculator.calculate_madelung_energy(atoms)

        # 2. BVS / BVVS
        feats.update(calculator.calculate_bvs_features(atoms))

        # 3. Geometry
        feats.update(calculator.calculate_geometric_features(atoms))

        features_list.append(feats)

    df_features = pd.DataFrame(features_list)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_features.to_parquet(cache_path)
    print(f"Saved EGDF features to {cache_path}")

    return df_features


def generate_submission(load_cached_data: bool = True):
    """
    Orchestrates the pipeline: Data Loading -> Feature Extraction -> Training -> Submission.
    """
    print("Starting EGDF Pipeline...")

    # 1. Load Metadata
    train_meta = load_metadata("train")
    val_meta = load_metadata("val")
    test_meta = load_metadata("test")

    # 2. Extract Features
    train_feats = extract_all_features(train_meta, "train", load_cached_data)
    val_feats = extract_all_features(val_meta, "val", load_cached_data)
    test_feats = extract_all_features(test_meta, "test", load_cached_data)

    # 3. Merge with Targets
    # Train and Val have targets in metadata
    train_full = pd.merge(train_meta, train_feats, on="id")
    val_full = pd.merge(val_meta, val_feats, on="id")
    test_full = pd.merge(test_meta, test_feats, on="id")

    # 4. Train and Predict
    print("\nTraining Model for Formation Energy...")
    pred_form, score_form = train_and_predict(
        train_full, val_full, test_full, "formation_energy_ev_natom"
    )

    print("\nTraining Model for Bandgap Energy...")
    pred_band, score_band = train_and_predict(
        train_full, val_full, test_full, "bandgap_energy_ev"
    )

    avg_rmsle = (score_form + score_band) / 2
    print(f"\nFinal Average RMSLE: {avg_rmsle:.6f}")

    # 5. Save Submission
    submission = pd.DataFrame(
        {
            "id": test_full["id"],
            "formation_energy_ev_natom": pred_form,
            "bandgap_energy_ev": pred_band,
        }
    )

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

    return submission
