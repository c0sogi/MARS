import os
import numpy as np
import pandas as pd
from library.config import GASEConfig


def load_structures():
    """
    Loads the structures.csv file containing atomic coordinates.

    Returns:
        pd.DataFrame: DataFrame with columns [molecule_name, atom_index, atom, x, y, z].
    """
    path = GASEConfig.STRUCTURES_CSV
    # Silent load
    df = pd.read_csv(path)
    return df


def load_metadata(split="train"):
    """
    Loads the metadata CSV for the specified split from the generated metadata directory.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded metadata DataFrame.
    """
    if split == "train":
        path = GASEConfig.TRAIN_CSV
    elif split == "val":
        path = GASEConfig.VAL_CSV
    elif split == "test":
        path = GASEConfig.TEST_CSV
    else:
        raise ValueError(f"Unknown split: {split}")

    return pd.read_csv(path)


def merge_structures(df, structures):
    """
    Merges atomic coordinates from the structures DataFrame into the main DataFrame
    for both atom_0 and atom_1 in the coupling pair.

    Args:
        df (pd.DataFrame): Main dataframe containing molecule_name, atom_index_0, atom_index_1.
        structures (pd.DataFrame): Structures dataframe.

    Returns:
        pd.DataFrame: Dataframe with added columns: x0, y0, z0, atom_0, x1, y1, z1, atom_1.
    """
    # Merge for Atom 0
    # We map (molecule_name, atom_index_0) -> (x, y, z, atom)
    df = pd.merge(
        df,
        structures[["molecule_name", "atom_index", "atom", "x", "y", "z"]],
        left_on=["molecule_name", "atom_index_0"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    # Rename columns to indicate Atom 0
    df = df.rename(columns={"atom": "atom_0", "x": "x0", "y": "y0", "z": "z0"})
    df = df.drop(columns=["atom_index"])

    # Merge for Atom 1
    # We map (molecule_name, atom_index_1) -> (x, y, z, atom)
    df = pd.merge(
        df,
        structures[["molecule_name", "atom_index", "atom", "x", "y", "z"]],
        left_on=["molecule_name", "atom_index_1"],
        right_on=["molecule_name", "atom_index"],
        how="left",
        suffixes=("", "_1"),  # Handle potential collisions if any
    )
    # Rename columns to indicate Atom 1
    # Note: If columns collided, they might have _1 suffix already, but we force rename for clarity
    rename_map = {"atom": "atom_1", "x": "x1", "y": "y1", "z": "z1"}
    # Handle pandas suffix behavior if necessary (though rename usually handles standard columns)
    df = df.rename(columns=rename_map)

    # Cleanup any remaining index columns or suffix issues if they arose
    if "atom_index" in df.columns:
        df = df.drop(columns=["atom_index"])

    return df


def calculate_geometry(df, structures=None):
    """
    Calculates essential geometric primitives and features for the atom pairs.
    Includes Euclidean distances and cosine angle features relative to the molecule's geometric center.

    Args:
        df (pd.DataFrame): Dataframe with coordinates x0, y0, z0, x1, y1, z1.
        structures (pd.DataFrame, optional): Full structures df used to calculate molecule-level
                                             properties like Center of Geometry.

    Returns:
        pd.DataFrame: Dataframe with added geometric features.
    """
    # 1. Basic Euclidean Distance & Components
    # Vector from Atom 1 to Atom 0
    df["dx"] = df["x0"] - df["x1"]
    df["dy"] = df["y0"] - df["y1"]
    df["dz"] = df["z0"] - df["z1"]

    # Euclidean Distance
    df["dist"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2 + df["dz"] ** 2)

    # Physics-informed Inverse Distances (Field decay)
    epsilon = 1e-6  # Prevent division by zero
    df["dist_inv"] = 1.0 / (df["dist"] + epsilon)
    df["dist_inv2"] = 1.0 / (df["dist"] ** 2 + epsilon)
    df["dist_inv3"] = 1.0 / (df["dist"] ** 3 + epsilon)

    # 2. Advanced: Center of Geometry (COG) Features
    # These serve as proxies for "bond angles" and global topological placement
    if structures is not None:
        # Calculate Center of Geometry per molecule
        cog = structures.groupby("molecule_name")[["x", "y", "z"]].mean().reset_index()
        cog = cog.rename(columns={"x": "cx", "y": "cy", "z": "cz"})

        # Merge COG into main dataframe
        df = pd.merge(df, cog, on="molecule_name", how="left")

        # Vector from COG to Atom 0 (r0)
        df["r0_x"] = df["x0"] - df["cx"]
        df["r0_y"] = df["y0"] - df["cy"]
        df["r0_z"] = df["z0"] - df["cz"]
        df["dist_c0"] = np.sqrt(df["r0_x"] ** 2 + df["r0_y"] ** 2 + df["r0_z"] ** 2)

        # Vector from COG to Atom 1 (r1)
        df["r1_x"] = df["x1"] - df["cx"]
        df["r1_y"] = df["y1"] - df["cy"]
        df["r1_z"] = df["z1"] - df["cz"]
        df["dist_c1"] = np.sqrt(df["r1_x"] ** 2 + df["r1_y"] ** 2 + df["r1_z"] ** 2)

        # Cosine Similarity between r0 and r1
        # Represents the angle subtended by the two atoms at the geometric center
        dot_r0_r1 = (
            df["r0_x"] * df["r1_x"] + df["r0_y"] * df["r1_y"] + df["r0_z"] * df["r1_z"]
        )
        df["cos_c0_c1"] = dot_r0_r1 / ((df["dist_c0"] * df["dist_c1"]) + epsilon)

        # Cosine Angle between Bond Vector (Atom1 -> Atom0) and r0 (COG -> Atom0)
        # Represents orientation of the bond relative to the center
        # Bond vector is (dx, dy, dz) calculated earlier (Atom0 - Atom1)
        # Note: Direction matters for cosine sign, consistency is key
        dot_bond_r0 = (
            df["dx"] * df["r0_x"] + df["dy"] * df["r0_y"] + df["dz"] * df["r0_z"]
        )
        df["cos_bond_r0"] = dot_bond_r0 / ((df["dist"] * df["dist_c0"]) + epsilon)

        # Cleanup intermediate columns to reduce memory footprint
        cols_to_drop = [
            "cx",
            "cy",
            "cz",
            "r0_x",
            "r0_y",
            "r0_z",
            "r1_x",
            "r1_y",
            "r1_z",
        ]
        df = df.drop(columns=cols_to_drop)

    return df


def process_and_cache_data(load_cached_data=True):
    """
    Orchestrates the data loading, processing, and caching pipeline.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed Parquet files.
                                 If False or files missing, processes from scratch.

    Returns:
        tuple: (df_train, df_val, df_test) processed DataFrames.
    """
    # Define file paths from config
    train_path = GASEConfig.PROCESSED_TRAIN_PATH
    val_path = GASEConfig.PROCESSED_VAL_PATH
    test_path = GASEConfig.PROCESSED_TEST_PATH

    # Ensure working directory exists
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    # 1. Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_path)
            and os.path.exists(val_path)
            and os.path.exists(test_path)
        ):
            print("Loading processed data from cache...")
            df_train = pd.read_parquet(train_path)
            df_val = pd.read_parquet(val_path)
            df_test = pd.read_parquet(test_path)
            return df_train, df_val, df_test
        else:
            print("Cache miss. Processing data from scratch...")
    else:
        print("Forcing data processing from scratch...")

    # 2. Process Data
    print("Loading raw structures...")
    structures = load_structures()

    # Process Train
    print("Processing Train set...")
    df_train = load_metadata("train")
    df_train = merge_structures(df_train, structures)
    df_train = calculate_geometry(df_train, structures)

    # Process Validation
    print("Processing Validation set...")
    df_val = load_metadata("val")
    df_val = merge_structures(df_val, structures)
    df_val = calculate_geometry(df_val, structures)

    # Process Test
    print("Processing Test set...")
    df_test = load_metadata("test")
    df_test = merge_structures(df_test, structures)
    df_test = calculate_geometry(df_test, structures)

    # 3. Save to Cache
    print(f"Saving processed data to {GASEConfig.WORKING_DIR}...")
    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    df_test.to_parquet(test_path, index=False)

    return df_train, df_val, df_test
