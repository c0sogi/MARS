import os
import numpy as np
import pandas as pd
import ase.io
from ase import Atoms
from ase.neighborlist import neighbor_list
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import pdist, squareform

# =============================================================================
# 1. GLOBAL CONFIGURATION & PATHS
# =============================================================================

INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_31"
SUBMISSION_DIR = "./submission"

TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

RANDOM_SEED = 42

# =============================================================================
# 2. PHYSICAL PARAMETERS
# =============================================================================

# Nominal Oxidation States for Electrostatics
OXIDATION_STATES = {"Al": 3.0, "Ga": 3.0, "In": 3.0, "O": -2.0}

# Bond Valence Parameters (R0)
# Form: {(Element1, Element2): R0}
# b is typically 0.37 Angstrom
BVS_B = 0.37
BVS_R0 = {
    ("Al", "O"): 1.651,
    ("O", "Al"): 1.651,
    ("Ga", "O"): 1.730,
    ("O", "Ga"): 1.730,
    ("In", "O"): 1.905,
    ("O", "In"): 1.905,
    # Self-interactions or cation-cation are usually ignored in simple BVS for oxides,
    # but we define them as 0 or handled by logic.
}

# Ewald Summation Parameters
EWALD_ACCURACY = 1e-5

# =============================================================================
# 3. HYPERPARAMETERS
# =============================================================================

XGB_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.01,
    "max_depth": 8,
    "subsample": 0.65,
    "colsample_bytree": 0.65,
    "min_child_weight": 5,
    "gamma": 0.1,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "objective": "reg:squarederror",
    "tree_method": "hist",
}

TRAIN_PARAMS = {"early_stopping_rounds": 100, "verbose": False}

# =============================================================================
# 4. FEATURE EXTRACTION LOGIC
# =============================================================================


def get_madelung_energy(atoms: Atoms) -> float:
    """
    Approximates the electrostatic (Madelung) energy of the crystal using Ewald summation.
    Returns the total electrostatic energy per atom.
    """
    # Assign charges
    charges = [OXIDATION_STATES.get(s, 0.0) for s in atoms.get_chemical_symbols()]
    atoms.set_initial_charges(charges)

    # Ewald summation requires a periodic cell.
    # We implement a simplified Ewald or use a library if available.
    # Here we use a screened Coulomb sum approximation via ASE's neighbor list for short range
    # and a simplified reciprocal term, or essentially a direct sum on a supercell for robustness.
    # For efficiency in this script, we use a direct summation on a sufficiently large supercell
    # with 1/r decay, which correlates well with Madelung energy for feature engineering.

    # Create a supercell to ensure convergence (e.g., 10 Angstrom cutoff context)
    # A 3x3x3 supercell is usually sufficient for feature extraction purposes.

    # Note: A full Ewald implementation is lengthy. We use a "pseudo-Madelung" feature:
    # Sum(q_i * q_j / r_ij) over a cutoff.

    cutoff = 10.0
    nl_i, nl_j, nl_d = neighbor_list("ijd", atoms, cutoff=cutoff)

    if len(nl_i) == 0:
        return 0.0

    q = atoms.get_initial_charges()

    # Energy = 0.5 * sum_{i!=j} (q_i * q_j / r_ij)
    # neighbor_list returns all pairs (i, j) within cutoff.

    # Filter out self-interaction if any (though neighbor_list usually handles i!=j for self image)
    # neighbor_list includes periodic images.

    energy_terms = (q[nl_i] * q[nl_j]) / nl_d
    total_energy = 0.5 * np.sum(energy_terms)

    return total_energy / len(atoms)


def get_bond_valence_features(atoms: Atoms) -> dict:
    """
    Computes Bond Valence Sums (BVS) and Bond Valence Vector Sums (BVVS).
    Returns statistical descriptors (min, max, mean, std) for each element type.
    """
    cutoff = 4.0  # Sufficient for M-O bonds
    nl_i, nl_j, nl_D = neighbor_list("ijD", atoms, cutoff=cutoff)
    nl_d = np.linalg.norm(nl_D, axis=1)

    symbols = atoms.get_chemical_symbols()
    n_atoms = len(atoms)

    # Initialize arrays
    bvs = np.zeros(n_atoms)
    bvvs = np.zeros((n_atoms, 3))

    for k, (i, j) in enumerate(zip(nl_i, nl_j)):
        elem_i = symbols[i]
        elem_j = symbols[j]
        dist = nl_d[k]

        # Only consider cation-anion bonds for BVS in oxides
        key = (elem_i, elem_j)
        if key in BVS_R0 and dist > 0.1:
            r0 = BVS_R0[key]
            s_ij = np.exp((r0 - dist) / BVS_B)

            bvs[i] += s_ij

            # Vector sum: weight unit vector by valence
            vec = nl_D[k] / dist
            bvvs[i] += s_ij * vec

    bvvs_mag = np.linalg.norm(bvvs, axis=1)

    # Global Instability Index
    # GII = sqrt( sum( (calc_valence - ideal_valence)^2 ) / N )
    # Ideal valence is oxidation state magnitude
    ideal_valences = np.array([abs(OXIDATION_STATES.get(s, 0)) for s in symbols])
    gii = np.sqrt(np.mean((bvs - ideal_valences) ** 2))

    features = {"GII": gii}

    # Per-element statistics
    for elem in ["Al", "Ga", "In", "O"]:
        indices = [i for i, s in enumerate(symbols) if s == elem]
        if not indices:
            # Fill with NaNs or 0s
            for stat in ["mean", "std", "max", "min"]:
                features[f"BVS_{elem}_{stat}"] = 0.0
                features[f"BVVS_{elem}_{stat}"] = 0.0
            continue

        elem_bvs = bvs[indices]
        elem_bvvs = bvvs_mag[indices]

        features[f"BVS_{elem}_mean"] = np.mean(elem_bvs)
        features[f"BVS_{elem}_std"] = np.std(elem_bvs)
        features[f"BVS_{elem}_max"] = np.max(elem_bvs)
        features[f"BVS_{elem}_min"] = np.min(elem_bvs)

        features[f"BVVS_{elem}_mean"] = np.mean(elem_bvvs)
        features[f"BVVS_{elem}_std"] = np.std(elem_bvvs)
        features[f"BVVS_{elem}_max"] = np.max(elem_bvvs)
        features[f"BVVS_{elem}_min"] = np.min(elem_bvvs)

    return features


def get_geometric_features(atoms: Atoms) -> dict:
    """
    Computes density, volume, and basic structural stats.
    """
    vol = atoms.get_volume()
    mass = sum(atoms.get_masses())
    density = mass / vol

    # Cell parameters
    cell = atoms.get_cell_lengths_and_angles()

    features = {
        "density": density,
        "volume_per_atom": vol / len(atoms),
        "cell_a": cell[0],
        "cell_b": cell[1],
        "cell_c": cell[2],
        "angle_alpha": cell[3],
        "angle_beta": cell[4],
        "angle_gamma": cell[5],
    }
    return features


def process_single_entry(row):
    """
    Process a single row from metadata.
    """
    file_path = os.path.join(INPUT_DIR, row["file_path"])
    try:
        atoms = ase.io.read(file_path)
    except Exception:
        # Fallback for empty or corrupt files
        return None

    feats = {}

    # 1. Global Electrostatics
    feats["madelung_energy"] = get_madelung_energy(atoms)

    # 2. Bond Valence
    bv_feats = get_bond_valence_features(atoms)
    feats.update(bv_feats)

    # 3. Geometry
    geo_feats = get_geometric_features(atoms)
    feats.update(geo_feats)

    # 4. Composition (from metadata directly or atoms)
    symbols = atoms.get_chemical_symbols()
    n = len(symbols)
    feats["comp_Al"] = symbols.count("Al") / n
    feats["comp_Ga"] = symbols.count("Ga") / n
    feats["comp_In"] = symbols.count("In") / n
    feats["comp_O"] = symbols.count("O") / n

    return feats


def process_dataset(
    metadata_df: pd.DataFrame, dataset_name: str, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Main processing function. Handles caching.
    """
    cache_path = os.path.join(WORKING_DIR, f"{dataset_name}_features.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features for {dataset_name} from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing {dataset_name} dataset...")
    features_list = []
    ids = []

    for idx, row in metadata_df.iterrows():
        f = process_single_entry(row)
        if f is not None:
            f["id"] = row["id"]
            features_list.append(f)
            ids.append(row["id"])

    df_features = pd.DataFrame(features_list)

    # Ensure ID is integer for merging
    df_features["id"] = df_features["id"].astype(int)

    # Save cache
    df_features.to_parquet(cache_path)
    print(f"Saved features to {cache_path}")

    return df_features


# =============================================================================
# 5. MODEL TRAINING & INFERENCE
# =============================================================================


def train_and_predict(train_df, val_df, test_df, target_col):
    """
    Trains an XGBoost model for a specific target and predicts on test set.
    """
    print(f"\n--- Training for Target: {target_col} ---")

    # Prepare Feature Matrix X and Target y
    # Drop non-feature columns
    drop_cols = ["id", "file_path", "formation_energy_ev_natom", "bandgap_energy_ev"]

    # Identify feature columns (intersection of columns in df and computed features)
    # We assume train_df already has the computed features merged.
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    X_test = test_df[feature_cols]

    # Log-transform targets to handle skew and enforce positivity (energy/bandgap > 0 usually)
    # y_trans = log(1 + y)
    # Note: formation energy can be 0, bandgap > 0.
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    model = xgb.XGBRegressor(**XGB_PARAMS)

    model.fit(
        X_train,
        y_train_log,
        eval_set=[(X_train, y_train_log), (X_val, y_val_log)],
        early_stopping_rounds=TRAIN_PARAMS.get("early_stopping_rounds", 50),
        verbose=TRAIN_PARAMS.get("verbose", False),
    )

    # Validation Score (RMSLE on original scale is RMSE on log scale)
    val_preds_log = model.predict(X_val)
    val_rmsle = np.sqrt(mean_squared_error(y_val_log, val_preds_log))
    print(f"Validation RMSLE ({target_col}): {val_rmsle:.6f}")

    # Predict on Test
    test_preds_log = model.predict(X_test)
    test_preds = np.expm1(test_preds_log)

    # Ensure non-negative
    test_preds = np.maximum(test_preds, 0)

    return test_preds, val_rmsle


def run_pipeline(load_cached_data=True):
    # 1. Load Metadata
    train_meta = pd.read_csv(TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(VAL_METADATA_PATH)
    test_meta = pd.read_csv(TEST_METADATA_PATH)

    # 2. Process Data (Feature Extraction)
    # We process them separately and then merge with metadata to get targets back
    df_train_feats = process_dataset(train_meta, "train", load_cached_data)
    df_val_feats = process_dataset(val_meta, "val", load_cached_data)
    df_test_feats = process_dataset(test_meta, "test", load_cached_data)

    # Merge features with targets
    # train_meta has 'id', 'formation...', 'bandgap...'
    train_full = pd.merge(train_meta, df_train_feats, on="id")
    val_full = pd.merge(val_meta, df_val_feats, on="id")
    test_full = pd.merge(test_meta, df_test_feats, on="id")

    # 3. Train & Predict
    # Target 1: Formation Energy
    pred_form, score_form = train_and_predict(
        train_full, val_full, test_full, "formation_energy_ev_natom"
    )

    # Target 2: Bandgap Energy
    pred_band, score_band = train_and_predict(
        train_full, val_full, test_full, "bandgap_energy_ev"
    )

    print(f"\nAverage RMSLE: {(score_form + score_band) / 2:.6f}")

    # 4. Create Submission
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
