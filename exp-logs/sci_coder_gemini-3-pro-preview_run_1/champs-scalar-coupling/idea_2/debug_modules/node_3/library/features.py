import os
import numpy as np
import pandas as pd
import warnings

# Import helper functions and the main processing function from the provided library
from library.data import (
    process_data as _lib_process_data,
    merge_structures,
    _compute_bond_graph,
    _compute_angular_features,
    _compute_topological_features,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set fixed random seed for reproducibility
np.random.seed(42)


def generate_features(df, structures):
    """
    Orchestrates the creation of descriptors for the molecule coupling pairs.

    Args:
        df (pd.DataFrame): Dataframe containing coupling pairs (must have id, molecule_name, atom_index_0, atom_index_1).
        structures (pd.DataFrame): Dataframe containing atomic structures.

    Returns:
        pd.DataFrame: The input dataframe enriched with geometric, angular, and topological features.
    """
    # 1. Merge Coordinates
    # Adds x0, y0, z0, atom_0, x1, y1, z1, atom_1
    df = merge_structures(df, structures)

    # 2. Basic Distance Features
    df["dx"] = df["x0"] - df["x1"]
    df["dy"] = df["y0"] - df["y1"]
    df["dz"] = df["z0"] - df["z1"]
    df["dist"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2 + df["dz"] ** 2)

    # Inverse distance features for field decay modeling
    df["dist_inv"] = 1.0 / df["dist"]
    df["dist_inv2"] = 1.0 / (df["dist"] ** 2)
    df["dist_inv3"] = 1.0 / (df["dist"] ** 3)

    # 3. Compute Bond Graph (Full Molecule Context)
    # This identifies all bonded neighbors for every atom in the relevant molecules
    bonds = _compute_bond_graph(structures)

    # 4. Angular Features (Cosine Projections)
    # Define reference vectors (Coupling Axis)

    # Ref for Atom 0: Vector pointing from 0 to 1 (x1 - x0) = -dx
    df["v_ref_x"] = -df["dx"]
    df["v_ref_y"] = -df["dy"]
    df["v_ref_z"] = -df["dz"]

    # Ref for Atom 1: Vector pointing from 1 to 0 (x0 - x1) = dx
    df["v_ref_neg_x"] = df["dx"]
    df["v_ref_neg_y"] = df["dy"]
    df["v_ref_neg_z"] = df["dz"]

    # Compute cosine similarity features for neighbors of atom_0 and atom_1
    ang_feat_0 = _compute_angular_features(df, bonds, "atom_index_0", "v_ref")
    ang_feat_1 = _compute_angular_features(df, bonds, "atom_index_1", "v_ref_neg")

    # Merge angular features back to main dataframe
    df = df.merge(ang_feat_0, on="id", how="left")
    df = df.merge(ang_feat_1, on="id", how="left")

    # Fill NaNs for atoms with no bonded neighbors (isolated/terminal)
    ang_cols = [c for c in df.columns if c.startswith("cos_")]
    df[ang_cols] = df[ang_cols].fillna(0.0)

    # 5. Topological Features (Bag of Neighbors)
    # Counts of neighbor atom types (C, H, N, O, F) and degree
    topo_feat_0 = _compute_topological_features(df, bonds, "atom_index_0")
    topo_feat_1 = _compute_topological_features(df, bonds, "atom_index_1")

    df = df.merge(topo_feat_0, on="id", how="left")
    df = df.merge(topo_feat_1, on="id", how="left")

    # Fill NaNs for counts (0 neighbors)
    topo_cols = [c for c in df.columns if c.startswith("n_") or c.startswith("degree")]
    df[topo_cols] = df[topo_cols].fillna(0)

    # 6. Cleanup
    # Drop intermediate vector columns to save memory
    drop_cols = [
        "v_ref_x",
        "v_ref_y",
        "v_ref_z",
        "v_ref_neg_x",
        "v_ref_neg_y",
        "v_ref_neg_z",
        "dx",
        "dy",
        "dz",
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df


def load_train_data(load_cached_data=True):
    """
    Loads the processed training data.
    Delegates to library.data.process_data which handles caching and processing.
    """
    return _lib_process_data("train", load_cached_data=load_cached_data)


def load_val_data(load_cached_data=True):
    """
    Loads the processed validation data.
    Delegates to library.data.process_data which handles caching and processing.
    """
    return _lib_process_data("val", load_cached_data=load_cached_data)


def load_test_data(load_cached_data=True):
    """
    Loads the processed test data.
    Delegates to library.data.process_data which handles caching and processing.
    """
    return _lib_process_data("test", load_cached_data=load_cached_data)
