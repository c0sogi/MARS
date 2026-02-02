import os
import numpy as np
import pandas as pd
from library.config import Config


def compute_sample_weights(
    df: pd.DataFrame, load_cached_data: bool = True
) -> np.ndarray:
    """
    Computes sample weights for the training data to mitigate bias.

    Strategy:
    Assign higher weights to examples that mention identity subgroups.
    These correspond to the 'Subgroup Negative' (Non-toxic + Identity)
    and 'Subgroup Positive' (Toxic + Identity) categories, which are
    critical for the BPSN and BNSP bias metrics.

    Args:
        df: Training DataFrame containing identity columns and target.
        load_cached_data: Whether to load pre-computed weights from disk.

    Returns:
        Numpy array of sample weights corresponding to the rows of df.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = Config.TRAIN_WEIGHTS_PATH
    weights = None

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            cached_weights = np.load(cache_path)
            # Verify length matches dataframe (crucial if nrows was used in load_data)
            if len(cached_weights) == len(df):
                weights = cached_weights
            else:
                # Length mismatch indicates data subset or change, recompute
                weights = None
        except Exception:
            # Load failed, recompute
            weights = None

    # 2. Compute if needed
    if weights is None:
        # Initialize with base weight
        weights = np.full(len(df), Config.BASE_WEIGHT, dtype=np.float32)

        # Check if identity columns exist in DataFrame
        available_identity_cols = [
            c for c in Config.IDENTITY_COLUMNS if c in df.columns
        ]

        if available_identity_cols:
            # Fill NaNs with 0.0 (assuming NaN means identity not mentioned)
            # Use a temporary dataframe for calculation to avoid modifying input
            identities = df[available_identity_cols].fillna(0.0)

            # Determine if any identity is mentioned (threshold >= 0.5)
            # We use >= 0.5 as the standard convention for this dataset's fractional labels
            # This mask captures both:
            # - Subgroup Negative (Non-toxic + Identity)
            # - Subgroup Positive (Toxic + Identity)
            has_identity = (identities >= 0.5).any(axis=1)

            # Apply multiplier to rows with identities
            weights[has_identity] = Config.BIAS_WEIGHT_MULTIPLIER

        # Save to cache
        np.save(cache_path, weights)

    return weights
