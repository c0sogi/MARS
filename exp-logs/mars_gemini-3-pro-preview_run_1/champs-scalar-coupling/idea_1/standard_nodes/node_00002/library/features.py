import os
import numpy as np
import pandas as pd
from library.config import Config

# Define consistent mappings for categorical encoding across all splits
# These are derived from the dataset description and analysis to ensure
# reproducibility without needing to save/load encoder objects.
COUPLING_TYPES = sorted(
    ["1JHC", "1JHN", "2JHC", "2JHH", "2JHN", "3JHC", "3JHH", "3JHN"]
)
ATOM_TYPES = sorted(["H", "C", "N", "O", "F"])


def process_data(
    split_name: str, load_cached_data: bool = True, debug_nrows: int = None
) -> pd.DataFrame:
    """
    Main function to load, process, and feature engineer the data for a specific split.
    Implements caching to Parquet to speed up subsequent runs.

    Args:
        split_name (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from cache first.
        debug_nrows (int, optional): If set, only loads this many rows for debugging.

    Returns:
        pd.DataFrame: The processed dataframe with features.
    """
    # 1. Resolve Paths based on split
    if split_name == "train":
        input_path = Config.TRAIN_PATH
        cache_path = Config.TRAIN_CACHE_PATH
    elif split_name == "val":
        input_path = Config.VAL_PATH
        cache_path = Config.VAL_CACHE_PATH
    elif split_name == "test":
        input_path = Config.TEST_PATH
        cache_path = Config.TEST_CACHE_PATH
    else:
        raise ValueError(
            f"Invalid split_name: {split_name}. Must be 'train', 'val', or 'test'."
        )

    # 2. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split_name} data from {cache_path}...")
        df = pd.read_parquet(cache_path)
        if debug_nrows is not None:
            df = df.head(debug_nrows)
        return df

    print(f"Processing {split_name} data from scratch...")

    # 3. Load Raw Data
    # Load metadata
    df = pd.read_csv(input_path, nrows=debug_nrows)

    # Load structures (always load full structures to ensure we find the atoms)
    structures = pd.read_csv(Config.STRUCTURES_PATH)

    # 4. Feature Engineering Pipeline
    df = merge_structures(df, structures)
    df = calculate_geometry(df)
    df = encode_categoricals(df)
    df = add_physicochemical_features(df)
    df = merge_atom_descriptors(df, structures)

    # 5. Save to Cache
    # We only cache if we processed the full dataset (debug_nrows is None)
    # to avoid overwriting a full cache with a partial debug run.
    if debug_nrows is None:
        print(f"Saving processed {split_name} data to {cache_path}...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path, index=False)

    return df


def merge_structures(df: pd.DataFrame, structures: pd.DataFrame) -> pd.DataFrame:
    """
    Merges atomic coordinates and element types from structures.csv into the main dataframe.
    """
    # Prepare structures dataframe for merging
    # structures columns: molecule_name, atom_index, atom, x, y, z

    # Select only necessary columns to keep merge clean
    struct_subset = structures[["molecule_name", "atom_index", "atom", "x", "y", "z"]]

    # --- Merge Atom 0 ---
    df = pd.merge(
        df,
        struct_subset,
        how="left",
        left_on=[Config.MOLECULE_COL, Config.ATOM_INDEX_0_COL],
        right_on=["molecule_name", "atom_index"],
    )

    # Rename and drop redundant columns for Atom 0
    df = df.rename(columns={"x": "x0", "y": "y0", "z": "z0", "atom": "atom_0"})
    df = df.drop(columns=["atom_index"])

    # --- Merge Atom 1 ---
    df = pd.merge(
        df,
        struct_subset,
        how="left",
        left_on=[Config.MOLECULE_COL, Config.ATOM_INDEX_1_COL],
        right_on=["molecule_name", "atom_index"],
        suffixes=("", "_1_dup"),
    )

    # Rename and drop redundant columns for Atom 1
    df = df.rename(columns={"x": "x1", "y": "y1", "z": "z1", "atom": "atom_1"})
    df = df.drop(columns=["atom_index"])

    return df


def calculate_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Euclidean distance and physics-informed inverse distance features.
    """
    # Extract coordinates as numpy arrays for vectorized calculation
    p0 = df[["x0", "y0", "z0"]].values
    p1 = df[["x1", "y1", "z1"]].values

    # Calculate Euclidean distance
    dist = np.linalg.norm(p0 - p1, axis=1)

    # Assign features
    df[Config.DIST_COL] = dist

    # Physics-informed inverse distances
    # Adding a tiny epsilon to ensure numerical stability, though atoms shouldn't overlap perfectly
    epsilon = 1e-9
    df[Config.DIST_INV_COL] = 1.0 / (dist + epsilon)
    df[Config.DIST_INV2_COL] = 1.0 / ((dist**2) + epsilon)
    df[Config.DIST_INV3_COL] = 1.0 / ((dist**3) + epsilon)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features (coupling type, atom types) using fixed mappings.
    """
    # Create mapping dictionaries
    type_map = {k: v for v, k in enumerate(COUPLING_TYPES)}
    atom_map = {k: v for v, k in enumerate(ATOM_TYPES)}

    # Apply mappings
    # We use map() which is efficient.
    # We fillna(-1) to handle potential unseen labels (safety check)

    if Config.TYPE_COL in df.columns:
        df[Config.TYPE_ENC_COL] = (
            df[Config.TYPE_COL].map(type_map).fillna(-1).astype(int)
        )

    # Encode atom types if they exist (created during merge_structures)
    if "atom_0" in df.columns:
        df[Config.ATOM_0_ENC_COL] = df["atom_0"].map(atom_map).fillna(-1).astype(int)

    if "atom_1" in df.columns:
        df[Config.ATOM_1_ENC_COL] = df["atom_1"].map(atom_map).fillna(-1).astype(int)

    return df


def add_physicochemical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds physics-informed features like electronegativity.
    Cite solution_lesson_node_00001: Physics-Informed Feature Engineering.
    """
    # Pauling Electronegativity Scale
    en_map = {"H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98}

    if "atom_0" in df.columns:
        df["en_0"] = df["atom_0"].map(en_map).fillna(0.0)

    if "atom_1" in df.columns:
        df["en_1"] = df["atom_1"].map(en_map).fillna(0.0)

    if "en_0" in df.columns and "en_1" in df.columns:
        df["en_diff"] = (df["en_0"] - df["en_1"]).abs()

    return df


def merge_atom_descriptors(df: pd.DataFrame, structures: pd.DataFrame) -> pd.DataFrame:
    """
    Generates and merges atom-level geometric descriptors (neighbors, bonds).
    Cite solution_lesson_node_00001: Relative geometric features.
    """
    atom_feats_cache = os.path.join(Config.WORKING_DIR, "atom_features.parquet")

    if os.path.exists(atom_feats_cache):
        atom_feats = pd.read_parquet(atom_feats_cache)
    else:
        print("Generating atom neighbor features (this may take a few minutes)...")
        # Self-join to find neighbors
        # Filter to reduce size: only keep necessary columns
        s = structures[["molecule_name", "atom_index", "x", "y", "z"]]

        merged = pd.merge(s, s, on="molecule_name", suffixes=("", "_neigh"))
        # Remove self-loops
        merged = merged[merged["atom_index"] != merged["atom_index_neigh"]]

        # Calculate distance
        merged["d"] = np.sqrt(
            (merged["x"] - merged["x_neigh"]) ** 2
            + (merged["y"] - merged["y_neigh"]) ** 2
            + (merged["z"] - merged["z_neigh"]) ** 2
        )

        # Define bond threshold (approximate for covalent)
        BOND_THRESHOLD = 1.6

        # Aggregations
        # n_bonds: count of neighbors within threshold
        # min_dist: distance to closest neighbor
        # mean_dist: average distance to all atoms in molecule (describes molecule size/density)

        merged["is_bond"] = (merged["d"] < BOND_THRESHOLD).astype(int)

        grp = merged.groupby(["molecule_name", "atom_index"])

        atom_feats = pd.DataFrame(
            {
                "n_bonds": grp["is_bond"].sum(),
                "min_dist_neigh": grp["d"].min(),
                "mean_dist_neigh": grp["d"].mean(),
            }
        ).reset_index()

        # Save to cache
        os.makedirs(os.path.dirname(atom_feats_cache), exist_ok=True)
        atom_feats.to_parquet(atom_feats_cache, index=False)

    # Merge for Atom 0
    df = (
        df.merge(
            atom_feats,
            left_on=[Config.MOLECULE_COL, Config.ATOM_INDEX_0_COL],
            right_on=["molecule_name", "atom_index"],
            how="left",
        )
        .rename(
            columns={
                "n_bonds": "n_bonds_0",
                "min_dist_neigh": "min_dist_neigh_0",
                "mean_dist_neigh": "mean_dist_neigh_0",
            }
        )
        .drop(columns=["atom_index"])
    )

    # Merge for Atom 1
    df = (
        df.merge(
            atom_feats,
            left_on=[Config.MOLECULE_COL, Config.ATOM_INDEX_1_COL],
            right_on=["molecule_name", "atom_index"],
            how="left",
        )
        .rename(
            columns={
                "n_bonds": "n_bonds_1",
                "min_dist_neigh": "min_dist_neigh_1",
                "mean_dist_neigh": "mean_dist_neigh_1",
            }
        )
        .drop(columns=["atom_index"])
    )

    return df
