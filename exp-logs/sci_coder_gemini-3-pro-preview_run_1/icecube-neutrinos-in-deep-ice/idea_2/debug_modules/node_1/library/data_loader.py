import pandas as pd
import numpy as np
from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    FEATURE_NAMES,
    DEBUG_N_ROWS,
)
from library.utils import spherical_to_cartesian
from library.features import generate_features


def load_train_dataset(load_cached_data=True, debug_n_rows=DEBUG_N_ROWS):
    """
    Loads the training dataset, generating features if necessary.

    Args:
        load_cached_data (bool): Whether to load from disk cache if available.
        debug_n_rows (int, optional): Number of rows to load for debugging.

    Returns:
        tuple: (X, y) where X is a DataFrame of features and y is a DataFrame of
               Cartesian target components (x, y, z).
    """
    # Generate or load features using the provided library function
    df = generate_features(
        meta_path=TRAIN_META_PATH,
        output_path=TRAIN_FEATURES_PATH,
        load_cached_data=load_cached_data,
        debug_n_rows=debug_n_rows,
    )

    # Extract Input Features
    X = df[FEATURE_NAMES].copy()

    # Extract and Transform Targets (Spherical -> Cartesian)
    # The models will predict the x, y, z components of the direction vector
    azimuth = df["azimuth"].values
    zenith = df["zenith"].values
    target_x, target_y, target_z = spherical_to_cartesian(azimuth, zenith)

    y = pd.DataFrame({"x": target_x, "y": target_y, "z": target_z})

    return X, y


def load_val_dataset(load_cached_data=True, debug_n_rows=DEBUG_N_ROWS):
    """
    Loads the validation dataset, generating features if necessary.

    Args:
        load_cached_data (bool): Whether to load from disk cache if available.
        debug_n_rows (int, optional): Number of rows to load for debugging.

    Returns:
        tuple: (X, y) where X is a DataFrame of features and y is a DataFrame of
               Cartesian target components (x, y, z).
    """
    # Generate or load features using the provided library function
    df = generate_features(
        meta_path=VAL_META_PATH,
        output_path=VAL_FEATURES_PATH,
        load_cached_data=load_cached_data,
        debug_n_rows=debug_n_rows,
    )

    # Extract Input Features
    X = df[FEATURE_NAMES].copy()

    # Extract and Transform Targets
    azimuth = df["azimuth"].values
    zenith = df["zenith"].values
    target_x, target_y, target_z = spherical_to_cartesian(azimuth, zenith)

    y = pd.DataFrame({"x": target_x, "y": target_y, "z": target_z})

    # We also return the original spherical targets for metric calculation if needed,
    # but the primary return is the transformed target for the regressor.
    # To keep the signature consistent with train, we return the Cartesian targets.
    # The evaluation loop can reconstruct angles or use the metadata if needed.

    return X, y


def load_test_dataset(load_cached_data=True, debug_n_rows=DEBUG_N_ROWS):
    """
    Loads the test dataset features.

    Args:
        load_cached_data (bool): Whether to load from disk cache if available.
        debug_n_rows (int, optional): Number of rows to load for debugging.

    Returns:
        tuple: (X, ids) where X is a DataFrame of features and ids is a Series of event_ids.
    """
    # Generate or load features using the provided library function
    df = generate_features(
        meta_path=TEST_META_PATH,
        output_path=TEST_FEATURES_PATH,
        load_cached_data=load_cached_data,
        debug_n_rows=debug_n_rows,
    )

    # Extract Input Features
    X = df[FEATURE_NAMES].copy()

    # Extract Event IDs for submission mapping
    ids = df["event_id"].copy()

    return X, ids
