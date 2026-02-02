import os
import pandas as pd
import numpy as np
from library.config import ATOMIC_PROPERTIES, WORKING_DIR
from library.geometry_utils import extract_geometry_features


def compute_elemental_moments(df):
    """
    Computes composition-weighted mean and standard deviation of atomic properties.
    Assumes the input dataframe contains 'percent_atom_al', 'percent_atom_ga', 'percent_atom_in'
    which sum to 1.0, representing the cation site fractions in a generic M2O3 stoichiometry.
    Therefore, global atomic fractions are approximated as:
    f_cation = 0.4 * percent_cation
    f_oxygen = 0.6
    """
    # Define weights for the crystal (assuming M2O3 stoichiometry)
    w_cation_scale = 0.4
    w_oxygen = 0.6

    # Calculate global atomic fractions
    frac_al = df["percent_atom_al"] * w_cation_scale
    frac_ga = df["percent_atom_ga"] * w_cation_scale
    frac_in = df["percent_atom_in"] * w_cation_scale
    # frac_o is constant 0.6 for all rows effectively, but we handle it vectorially

    properties = ["electronegativity", "radius", "valence"]
    moments = {}

    for prop in properties:
        # Retrieve atomic property values
        p_al = ATOMIC_PROPERTIES["Al"][prop]
        p_ga = ATOMIC_PROPERTIES["Ga"][prop]
        p_in = ATOMIC_PROPERTIES["In"][prop]
        p_o = ATOMIC_PROPERTIES["O"][prop]

        # 1. Weighted Mean
        # mean = sum(f_i * p_i)
        mean_val = (
            (frac_al * p_al) + (frac_ga * p_ga) + (frac_in * p_in) + (w_oxygen * p_o)
        )
        moments[f"mean_{prop}"] = mean_val

        # 2. Weighted Standard Deviation
        # var = sum(f_i * (p_i - mean)^2)
        var_val = (
            (frac_al * (p_al - mean_val) ** 2)
            + (frac_ga * (p_ga - mean_val) ** 2)
            + (frac_in * (p_in - mean_val) ** 2)
            + (w_oxygen * (p_o - mean_val) ** 2)
        )

        moments[f"std_{prop}"] = np.sqrt(var_val)

    return pd.DataFrame(moments, index=df.index)


def build_feature_matrix(metadata_df, split_name, load_cached_data=True):
    """
    Constructs the final feature matrix for training or inference.

    Combines:
    1. Tabular metadata (lattice, composition, spacegroup)
    2. Extracted geometry features (density, volume, bond statistics)
    3. Computed elemental property moments

    Args:
        metadata_df (pd.DataFrame): The metadata dataframe loaded from CSV.
        split_name (str): 'train', 'val', or 'test' to identify the cache file.
        load_cached_data (bool): Whether to use cached feature files.

    Returns:
        pd.DataFrame: The complete feature matrix (X).
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache path for the final combined feature matrix
    cache_path = os.path.join(WORKING_DIR, f"{split_name}_combined_features.parquet")

    # 1. Try to load entire feature matrix from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading combined features from {cache_path}...")
        try:
            combined_df = pd.read_parquet(cache_path)
            # Verify index alignment
            if len(combined_df) == len(metadata_df) and np.all(
                combined_df.index == metadata_df.index
            ):
                return combined_df
            else:
                print("Cached combined data index mismatch. Recomputing...")
        except Exception as e:
            print(f"Failed to load combined cache: {e}. Recomputing...")

    print(f"Building feature matrix for {split_name}...")

    # 2. Extract Geometry Features (Handles its own caching)
    # We pass a unique cache name for the geometry part specifically
    geo_cache_name = f"{split_name}_geometry_features"
    geo_features = extract_geometry_features(
        metadata_df, load_cached_data=load_cached_data, cache_name=geo_cache_name
    )

    # 3. Compute Elemental Moments
    elemental_features = compute_elemental_moments(metadata_df)

    # 4. Select and Process Tabular Features
    # Lattice vectors and angles
    lattice_cols = [
        "lattice_vector_1_ang",
        "lattice_vector_2_ang",
        "lattice_vector_3_ang",
        "lattice_angle_alpha_degree",
        "lattice_angle_beta_degree",
        "lattice_angle_gamma_degree",
    ]

    # Atomic percentages (raw input features)
    comp_cols = ["percent_atom_al", "percent_atom_ga", "percent_atom_in"]

    # Spacegroup: One-Hot Encoding
    # We explicitly encode the known spacegroups to ensure consistent columns across splits
    # Common spacegroups for these oxides: 12, 33, 167, 194, 206, 227
    known_spacegroups = [12, 33, 167, 194, 206, 227]
    sg_features = pd.DataFrame(index=metadata_df.index)
    for sg in known_spacegroups:
        sg_features[f"sg_{sg}"] = (metadata_df["spacegroup"] == sg).astype(int)

    # Also include raw spacegroup as integer for tree-based models which can handle it
    sg_features["spacegroup_int"] = metadata_df["spacegroup"]

    # 5. Concatenate All Features
    # We use the index from metadata_df to ensure alignment
    tabular_subset = metadata_df[lattice_cols + comp_cols].copy()

    feature_matrix = pd.concat(
        [tabular_subset, sg_features, geo_features, elemental_features], axis=1
    )

    # 6. Save to Cache
    try:
        feature_matrix.to_parquet(cache_path)
        print(f"Saved combined features to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save combined cache to {cache_path}: {e}")

    return feature_matrix
