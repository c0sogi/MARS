import os
import pandas as pd
import numpy as np
from library.config import Config


def engineer_features(
    df: pd.DataFrame, name: str, config: Config = Config()
) -> pd.DataFrame:
    """
    Engineers features for the ventilator pressure prediction task based on the
    Logic-Gated Residual-Hybrid Network (LGRH-Net) design.

    This function computes physical state features (volume integration, time deltas),
    kinematic features (velocity, lookahead), and interactions, while handling
    caching to optimize runtime.

    Args:
        df (pd.DataFrame): The input dataframe containing raw ventilator data.
        name (str): The name of the dataset split (e.g., 'train', 'val', 'test') for caching.
        config (Config): Configuration object containing paths and hyperparameters.

    Returns:
        pd.DataFrame: The dataframe with engineered features.
    """
    # ==========================================
    # 1. Caching Logic
    # ==========================================
    # Ensure the cache directory exists
    os.makedirs(config.cache_dir, exist_ok=True)
    cache_path = os.path.join(config.cache_dir, f"{name}_features.parquet")

    # Check if cache exists and loading is enabled
    if config.load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features for {name} from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Engineering features for {name}...")

    # ==========================================
    # 2. Physical State Features
    # ==========================================
    # Calculate Time Delta (dt)
    # We group by breath_id to ensure we don't calculate diffs across different breaths.
    # The first time step of each breath will have NaN, which we fill with 0.
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)

    # Calculate Volume Proxy (Area)
    # Numerical integration of u_in over time: sum(u_in * dt)
    # This represents the total volume of air let into the lung so far.
    # We use a temporary column for the product to avoid modifying u_in yet.
    volume_step = df["u_in"] * df["dt"]
    df["area"] = volume_step.groupby(df["breath_id"]).cumsum()

    # ==========================================
    # 3. Kinematic Features
    # ==========================================
    # Backward Velocity: Rate of change of the inspiratory valve
    # u_in(t) - u_in(t-1)
    df["u_in_diff"] = df.groupby("breath_id")["u_in"].diff().fillna(0)

    # Forward Lookahead: Intent of the controller
    # u_in(t+1) ... u_in(t+k)
    # This gives the model visibility into the immediate future control signals.
    for i in range(1, config.lookahead_steps + 1):
        df[f"u_in_lookahead_{i}"] = df.groupby("breath_id")["u_in"].shift(-i).fillna(0)

    # ==========================================
    # 4. Interactions
    # ==========================================
    # Interaction features between control input and lung attributes
    # These help the model model the physical relationship P ~ R*Flow + V/C
    df["R_u_in"] = df["R"] * df["u_in"]
    df["C_u_in"] = df["C"] * df["u_in"]
    df["u_in_u_out"] = df["u_in"] * df["u_out"]

    # ==========================================
    # 5. Exclusions & Cleanup
    # ==========================================
    # Exclude raw time_step as per design to rely on dt and relative dynamics
    if "time_step" in df.columns:
        df = df.drop(columns=["time_step"])

    # Note: We retain 'u_out' (raw) here.
    # The separation into "Stream A (Scaled)" and "Stream B (Logic Gate)"
    # will be handled by the Dataset/DataLoader class, not here.

    # ==========================================
    # 6. Save to Cache
    # ==========================================
    print(f"Saving features for {name} to {cache_path}")
    df.to_parquet(cache_path, index=False)

    return df
