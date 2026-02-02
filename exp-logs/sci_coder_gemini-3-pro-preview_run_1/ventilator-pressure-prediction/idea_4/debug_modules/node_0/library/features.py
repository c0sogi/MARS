import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import generate_cache_hash, ensure_dir


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes physics-based features derived from the Equation of Motion.

    Adds the following columns:
    - cumulative_volume: Integral of u_in over time.
    - flow_interaction: u_in * R (Resistive Pressure component)
    - vol_interaction: cumulative_volume / C (Elastic Pressure component)
    - theoretical_pressure: flow_interaction + vol_interaction
    """
    # Ensure data is sorted for time-series operations
    # Reset index to ensure clean alignment
    df = df.sort_values(by=[Config.breath_id_col, Config.time_col]).reset_index(
        drop=True
    )

    # Calculate time delta (dt)
    # We group by breath_id to ensure the first time step of a new breath
    # doesn't calculate a diff with the last step of the previous breath.
    df["dt"] = df.groupby(Config.breath_id_col)[Config.time_col].diff().fillna(0)

    # Calculate Volume (Integral of Flow)
    # u_in is the control input for flow (0-100).
    # While not exactly in mL/s, it is proportional to flow.
    df["d_vol"] = df["u_in"] * df["dt"]
    df["cumulative_volume"] = df.groupby(Config.breath_id_col)["d_vol"].cumsum()

    # Interaction Terms (Physics-Informed)
    # Resistive Component: Flow * Resistance
    df["flow_interaction"] = df["u_in"] * df["R"]

    # Elastic Component: Volume / Compliance
    # Note: C is in mL/cmH2O. Higher C means easier to expand (lower pressure for same volume).
    df["vol_interaction"] = df["cumulative_volume"] / df["C"]

    # Theoretical Pressure (Approximation of Equation of Motion)
    # P_theoretical = P_resistive + P_elastic
    # This serves as the baseline for the residual connection in the model.
    df["theoretical_pressure"] = df["flow_interaction"] + df["vol_interaction"]

    # Cleanup intermediate columns not needed for the model or analysis
    # We keep the calculated features and the original identifiers/targets
    df.drop(columns=["dt", "d_vol"], inplace=True)

    return df


def get_processed_dataset(
    split_name: str, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Retrieves the processed dataset for a given split.
    Handles caching to avoid re-computing features on every run.

    Args:
        split_name (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from disk cache.

    Returns:
        pd.DataFrame: The processed dataframe with engineering features.
    """
    # 1. Determine Input File Path
    if split_name == "train":
        input_path = Config.train_file
    elif split_name == "val":
        input_path = Config.val_file
    elif split_name == "test":
        input_path = Config.test_file
    else:
        raise ValueError(
            f"Invalid split_name: {split_name}. Must be 'train', 'val', or 'test'."
        )

    # 2. Generate Cache Key
    # The cache key depends on the split, debug state, and the feature definition logic.
    # We include feature_cols to invalidate cache if we change which features we want.
    cache_config = {
        "split": split_name,
        "debug": Config.debug,
        "feature_cols": Config.feature_cols,
        "logic_version": "v1.0_physics_residual",
    }
    cache_hash = generate_cache_hash(cache_config)
    cache_filename = f"dataset_{split_name}_{cache_hash}.parquet"
    cache_path = os.path.join(Config.cache_dir, cache_filename)

    # 3. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split_name} dataset from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 4. Compute Features from Scratch
    print(f"Processing {split_name} dataset from {input_path}...")
    df = pd.read_csv(input_path)

    # Handle Debug Mode (Subsampling)
    if Config.debug:
        print(f"DEBUG MODE: Subsampling {split_name} dataset...")
        # Get first 200 breath_ids to maintain time-series integrity
        # We do not sample random rows because LSTM requires complete sequences.
        breath_ids = df[Config.breath_id_col].unique()
        subset_ids = breath_ids[:200]
        df = df[df[Config.breath_id_col].isin(subset_ids)].copy()

    # Apply Feature Engineering
    df = add_physics_features(df)

    # 5. Save to Cache
    ensure_dir(cache_path)
    print(f"Saving processed {split_name} dataset to {cache_path}...")
    df.to_parquet(cache_path, index=False)

    return df
