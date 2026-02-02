import os
import numpy as np
import pandas as pd
import library.features
from library.config import WORKING_DIR, TARGET_COLS, BOND_CUTOFF, PERCENTILES
from ase.neighborlist import neighbor_list


# --- Monkey Patching to fix bug in library.features ---
def extract_interaction_fingerprints_fixed(atoms):
    """
    Extracts distributional fingerprints (percentiles) of bond lengths and bond angles
    for chemically resolved interactions.
    Patched version to fix variable name bug in the provided library.
    """
    i_indices, j_indices, d_indices, D_vectors = neighbor_list(
        "ijdD", atoms, BOND_CUTOFF
    )

    symbols = np.array(atoms.get_chemical_symbols())

    bond_lengths_pop = {}
    bond_angles_pop = {}

    # --- Collect Bond Lengths ---
    for k in range(len(i_indices)):
        idx_i = i_indices[k]
        idx_j = j_indices[k]
        dist = d_indices[k]

        sym_i = symbols[idx_i]
        sym_j = symbols[idx_j]

        pair_key = tuple(sorted((sym_i, sym_j)))
        pair_str = f"bond_{pair_key[0]}_{pair_key[1]}"

        if pair_str not in bond_lengths_pop:
            bond_lengths_pop[pair_str] = []
        bond_lengths_pop[pair_str].append(dist)

    # --- Collect Bond Angles ---
    adj_list = [[] for _ in range(len(atoms))]
    for k in range(len(i_indices)):
        adj_list[i_indices[k]].append((j_indices[k], D_vectors[k], d_indices[k]))

    for center_idx, neighbors in enumerate(adj_list):
        center_sym = symbols[center_idx]
        n_neighbors = len(neighbors)

        if n_neighbors < 2:
            continue

        for n1 in range(n_neighbors):
            for n2 in range(n1 + 1, n_neighbors):
                idx1, vec1, dist1 = neighbors[n1]
                idx2, vec2, dist2 = neighbors[n2]

                sym1 = symbols[idx1]
                sym2 = symbols[idx2]

                dot_prod = np.dot(vec1, vec2)
                denominator = dist1 * dist2

                if denominator < 1e-8:
                    continue

                cos_theta = np.clip(dot_prod / denominator, -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_theta))

                neigh_syms = sorted((sym1, sym2))
                # FIX: Use neigh_syms instead of neigh_key
                angle_key_str = f"angle_{neigh_syms[0]}_{center_sym}_{neigh_syms[1]}"

                if angle_key_str not in bond_angles_pop:
                    bond_angles_pop[angle_key_str] = []
                bond_angles_pop[angle_key_str].append(angle_deg)

    features = {}

    def compute_stats(name, values):
        res = {}
        if not values:
            return res
        try:
            pct_values = np.percentile(values, PERCENTILES)
            for p, val in zip(PERCENTILES, pct_values):
                res[f"{name}_p{p}"] = val
        except Exception:
            pass
        return res

    for name, values in bond_lengths_pop.items():
        features.update(compute_stats(name, values))

    for name, values in bond_angles_pop.items():
        features.update(compute_stats(name, values))

    return features


# Apply the patch
library.features.extract_interaction_fingerprints = (
    extract_interaction_fingerprints_fixed
)
# -------------------------------------------------------


def process_dataset(metadata_path, load_cached_data=True):
    """
    Wrapper around library.features.process_data to handle data loading.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Call the provided feature processing function
    # It handles loading from cache or computing from scratch and saving to cache
    df = library.features.process_data(metadata_path, load_cached_data=load_cached_data)
    return df


def load_datasets(load_cached_data=True):
    """
    Loads train, validation, and test datasets using metadata paths.
    """
    train_path = "./metadata/train_metadata.csv"
    val_path = "./metadata/val_metadata.csv"
    test_path = "./metadata/test_metadata.csv"

    print("Loading Training Data...")
    df_train = process_dataset(train_path, load_cached_data)
    print("Loading Validation Data...")
    df_val = process_dataset(val_path, load_cached_data)
    print("Loading Test Data...")
    df_test = process_dataset(test_path, load_cached_data)

    return df_train, df_val, df_test


def preprocess_targets(df, target_cols=None):
    """
    Applies log1p transformation to the target columns.
    """
    if target_cols is None:
        target_cols = TARGET_COLS

    df_out = df.copy()
    for col in target_cols:
        if col in df_out.columns:
            # Apply log1p: log(1 + y)
            df_out[col] = np.log1p(df_out[col])
    return df_out


def inverse_transform_targets(y_pred):
    """
    Applies expm1 transformation to predictions (inverse of log1p).
    """
    return np.expm1(y_pred)


def prepare_matrices(df_train, df_val, df_test, target_cols=None):
    """
    Prepares feature matrices and target vectors.
    Drops constant columns and aligns columns across splits.
    """
    if target_cols is None:
        target_cols = TARGET_COLS

    # Identify potential feature columns (exclude ID, file_path, and targets)
    ignore = ["id", "file_path"] + target_cols
    potential_features = [
        c
        for c in df_train.columns
        if c not in ignore and pd.api.types.is_numeric_dtype(df_train[c])
    ]

    # Drop constant features based on training set statistics
    std = df_train[potential_features].std()
    selected_features = std[std > 1e-9].index.tolist()

    print(
        f"Features selected: {len(selected_features)} (dropped {len(potential_features) - len(selected_features)} constants)"
    )

    # Prepare Train
    X_train = df_train[selected_features].fillna(0)
    y_train = df_train[target_cols]

    # Prepare Val (align columns to train, fill missing with 0)
    X_val = df_val.reindex(columns=selected_features, fill_value=0).fillna(0)
    y_val = df_val[target_cols]

    # Prepare Test (align columns to train, fill missing with 0)
    X_test = df_test.reindex(columns=selected_features, fill_value=0).fillna(0)

    return X_train, y_train, X_val, y_val, X_test, selected_features
