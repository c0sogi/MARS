import os
import numpy as np
import pandas as pd
from ase.io import read
from ase.neighborlist import neighbor_list
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import library.config as config


class StructureFingerprinter:
    def __init__(self, atoms):
        self.atoms = atoms
        self.vol = atoms.get_volume()
        self.chemical_symbols = np.array(atoms.get_chemical_symbols())
        self.positions = atoms.get_positions()

        # Precompute neighbor list for efficiency
        # Cutoff covers RDF range. For bonds/angles we filter closer.
        # 'i': index of center atom
        # 'j': index of neighbor atom
        # 'd': distance
        # 'D': vector pointing from i to j
        self.i, self.j, self.d, self.D = neighbor_list(
            "ijdD", atoms, config.NEIGHBOR_CUTOFF
        )

    def get_macroscopic_features(self):
        mass = sum(self.atoms.get_masses())
        density = mass / self.vol if self.vol > 0 else 0.0
        return {"vol_per_atom": self.vol / len(self.atoms), "density": density}

    def get_radial_features(self):
        # Element-resolved RDFs
        features = {}
        bins = np.linspace(0, config.NEIGHBOR_CUTOFF, config.RDF_BINS + 1)

        # Create a dataframe of neighbors for easy filtering
        df_neigh = pd.DataFrame({"i": self.i, "j": self.j, "d": self.d})
        el_i = self.chemical_symbols[self.i]
        el_j = self.chemical_symbols[self.j]

        df_neigh["el_i"] = el_i
        df_neigh["el_j"] = el_j

        # Define pairs of interest: Metal-Oxygen and O-O
        pairs = []
        for m in config.METALS:
            pairs.append((m, "O"))
        pairs.append(("O", "O"))

        n_atoms = len(self.atoms)

        for e1, e2 in pairs:
            # Filter neighbors
            mask = (df_neigh["el_i"] == e1) & (df_neigh["el_j"] == e2)
            dists = df_neigh.loc[mask, "d"].values

            hist, _ = np.histogram(dists, bins=bins)
            # Normalize by total atoms to make it intensive-ish
            hist = hist.astype(float) / n_atoms

            for k, val in enumerate(hist):
                features[f"rdf_{e1}_{e2}_{k}"] = val

        return features

    def get_atomic_state_features(self):
        # BVS, ECoN, Local Anisotropy
        n_atoms = len(self.atoms)
        bvs = np.zeros(n_atoms)
        econ = np.zeros(n_atoms)
        anisotropy = np.zeros(n_atoms)

        # Prepare Dataframe for grouping
        df_neigh = pd.DataFrame(
            {
                "i": self.i,
                "j": self.j,
                "d": self.d,
                "Dx": self.D[:, 0],
                "Dy": self.D[:, 1],
                "Dz": self.D[:, 2],
            }
        )
        df_neigh["el_i"] = self.chemical_symbols[self.i]
        df_neigh["el_j"] = self.chemical_symbols[self.j]

        # 1. Bond Valence Sum (BVS)
        # Only for Metal-Oxygen pairs
        # Map R0 values
        # We construct a key column to map
        # Note: BVS_PARAMS keys are tuples.

        # Helper to get R0
        def get_r0(row):
            return config.BVS_PARAMS.get((row["el_i"], row["el_j"]), 0.0)

        # Apply is slow, but robust. For speed, we can map using a dict
        # Construct a look up dict for all combinations present
        unique_pairs = set(zip(df_neigh["el_i"], df_neigh["el_j"]))
        r0_map = {pair: config.BVS_PARAMS.get(pair, 0.0) for pair in unique_pairs}

        # Map R0
        # Create a tuple series to map
        pair_series = list(zip(df_neigh["el_i"], df_neigh["el_j"]))
        df_neigh["r0"] = [r0_map[p] for p in pair_series]

        # Calculate term: exp((R0 - d) / 0.37)
        # Only where R0 > 0
        mask_bvs = df_neigh["r0"] > 0
        df_neigh.loc[mask_bvs, "bvs_term"] = np.exp(
            (df_neigh.loc[mask_bvs, "r0"] - df_neigh.loc[mask_bvs, "d"]) / config.BVS_B
        )
        df_neigh.loc[~mask_bvs, "bvs_term"] = 0.0

        # Sum by atom i
        bvs_series = df_neigh.groupby("i")["bvs_term"].sum()
        bvs[bvs_series.index] = bvs_series.values

        # 2. Effective Coordination Number (ECoN)
        # We use a simple count of neighbors within a bond cutoff (3.0 A)
        # This is a robust proxy for steric packing in these oxides
        bond_cutoff = 3.0
        mask_bond = df_neigh["d"] < bond_cutoff
        df_bonded = df_neigh[mask_bond].copy()

        econ_series = df_bonded.groupby("i").size()
        econ[econ_series.index] = econ_series.values

        # 3. Local Anisotropy
        # Vector sum of normalized bond vectors
        # Normalize D vectors
        norms = np.sqrt(
            df_bonded["Dx"] ** 2 + df_bonded["Dy"] ** 2 + df_bonded["Dz"] ** 2
        )
        # Avoid division by zero
        norms[norms == 0] = 1.0

        df_bonded["nx"] = df_bonded["Dx"] / norms
        df_bonded["ny"] = df_bonded["Dy"] / norms
        df_bonded["nz"] = df_bonded["Dz"] / norms

        # Sum normalized vectors per atom
        vec_sum = df_bonded.groupby("i")[["nx", "ny", "nz"]].sum()

        # Calculate magnitude of the sum vector
        mags = np.sqrt(vec_sum["nx"] ** 2 + vec_sum["ny"] ** 2 + vec_sum["nz"] ** 2)

        # Assign to anisotropy array (handle missing atoms which have 0 anisotropy)
        anisotropy[mags.index] = mags.values

        # --- Aggregation by Element ---
        features = {}

        df_atoms = pd.DataFrame(
            {
                "element": self.chemical_symbols,
                "bvs": bvs,
                "econ": econ,
                "anisotropy": anisotropy,
            }
        )

        for el in config.ELEMENTS:
            sub = df_atoms[df_atoms["element"] == el]
            if len(sub) == 0:
                # If element not present, fill with 0
                for p in config.PERCENTILES:
                    features[f"bvs_{el}_p{p}"] = 0.0
                    features[f"econ_{el}_p{p}"] = 0.0
                    features[f"aniso_{el}_p{p}"] = 0.0
            else:
                for p in config.PERCENTILES:
                    features[f"bvs_{el}_p{p}"] = np.percentile(sub["bvs"], p)
                    features[f"econ_{el}_p{p}"] = np.percentile(sub["econ"], p)
                    features[f"aniso_{el}_p{p}"] = np.percentile(sub["anisotropy"], p)

        return features

    def get_topology_features(self):
        # Bond angles for M-O-M and O-M-O
        # Use bond cutoff 2.5 A (typical for these oxides)
        bond_cutoff = 2.5

        # Filter neighbors
        mask = self.d < bond_cutoff
        i_bonded = self.i[mask]
        j_bonded = self.j[mask]
        D_bonded = self.D[mask]

        if len(i_bonded) == 0:
            # No bonds found
            features = {}
            for name in ["mom", "omo"]:
                for p in config.PERCENTILES:
                    features[f"angle_{name}_p{p}"] = 0.0
            return features

        # Create DataFrame for grouping
        df_b = pd.DataFrame(
            {
                "i": i_bonded,
                "vx": D_bonded[:, 0],
                "vy": D_bonded[:, 1],
                "vz": D_bonded[:, 2],
            }
        )

        mom_angles = []
        omo_angles = []

        # Group by center atom 'i'
        grouped = df_b.groupby("i")

        for center_idx, group in grouped:
            center_el = self.chemical_symbols[center_idx]

            # Need at least 2 neighbors to form an angle
            if len(group) < 2:
                continue

            # Get vectors
            vecs = group[["vx", "vy", "vz"]].values

            # Normalize vectors
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            # Avoid divide by zero
            norms[norms == 0] = 1.0
            vecs_norm = vecs / norms

            # Compute cosine matrix: (N, 3) @ (3, N) -> (N, N)
            cos_matrix = np.dot(vecs_norm, vecs_norm.T)

            # Clip to valid domain for arccos
            cos_matrix = np.clip(cos_matrix, -1.0, 1.0)

            # Extract upper triangle indices (excluding diagonal)
            n = len(group)
            # Limit n to avoid explosion in dense structures (though 2.5A cutoff prevents this)
            if n > 12:
                continue

            iu = np.triu_indices(n, k=1)
            cosines = cos_matrix[iu]
            angles = np.degrees(np.arccos(cosines))

            if center_el == "O":
                # Center is Oxygen -> M-O-M angle
                mom_angles.extend(angles)
            elif center_el in config.METALS:
                # Center is Metal -> O-M-O angle
                omo_angles.extend(angles)

        features = {}

        for name, arr in [("mom", mom_angles), ("omo", omo_angles)]:
            if len(arr) == 0:
                for p in config.PERCENTILES:
                    features[f"angle_{name}_p{p}"] = 0.0
            else:
                for p in config.PERCENTILES:
                    features[f"angle_{name}_p{p}"] = np.percentile(arr, p)

        return features

    def fingerprint(self):
        f1 = self.get_macroscopic_features()
        f2 = self.get_radial_features()
        f3 = self.get_atomic_state_features()
        f4 = self.get_topology_features()
        return {**f1, **f2, **f3, **f4}


def process_dataset(metadata_path, load_cached_data=True):
    # Determine cache path
    base_name = os.path.basename(metadata_path).replace("_metadata.csv", "")
    cache_path = os.path.join(config.WORKING_DIR, f"{base_name}_features.parquet")

    # Ensure working dir exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing dataset from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    features_list = []

    # Iterate and compute
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {base_name}"):
        file_path = os.path.join(config.INPUT_DIR, row["file_path"])
        try:
            atoms = read(file_path)
            fingerprinter = StructureFingerprinter(atoms)
            feats = fingerprinter.fingerprint()
            feats["id"] = row["id"]
            features_list.append(feats)
        except Exception as e:
            # In case of read failure, we skip.
            # Ideally we should handle this, but for this task we assume data integrity.
            print(f"Error reading {file_path}: {e}")
            pass

    if not features_list:
        print("No features extracted!")
        return pd.DataFrame()

    feat_df = pd.DataFrame(features_list)

    # Merge with metadata
    merged_df = pd.merge(df, feat_df, on="id", how="left")

    # Save to cache
    print(f"Saving features to {cache_path}")
    merged_df.to_parquet(cache_path)

    return merged_df
