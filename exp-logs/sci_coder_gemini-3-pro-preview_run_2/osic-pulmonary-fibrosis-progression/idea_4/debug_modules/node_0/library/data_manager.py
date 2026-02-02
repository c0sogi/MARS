import os
import numpy as np
import pandas as pd
from library.config import Config, process_tabular_data


def load_and_merge_data(mode="train", image_features=None, load_cached_data=True):
    """
    Loads clinical metadata, processes tabular features, and merges with image embeddings.
    Implements caching to disk using .npz format.

    Args:
        mode (str): Dataset split to load ('train', 'val', or 'test').
        image_features (np.array): PCA-reduced image features (N_samples, N_components).
                                   Required for merging.
        load_cached_data (bool): If True, attempts to load processed data from cache.

    Returns:
        dict: Dictionary containing:
            - 'X_static': Combined static clinical + image features.
            - 'weeks': Relative weeks vector.
            - 'y': Target FVC values (if available).
            - 'patient_weeks': IDs for submission (if available).
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, f"merged_data_{mode}.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # Load and convert to dict to ensure data remains accessible in memory
            with np.load(cache_path, allow_pickle=True) as data:
                return dict(data)
        except Exception:
            # If load fails (e.g. corrupt file), proceed to recompute
            pass

    # 2. Compute from scratch

    # Load Metadata
    meta_path = os.path.join(Config.METADATA_DIR, f"{mode}_metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at {meta_path}")

    df_meta = pd.read_csv(meta_path)

    # Process Tabular Data
    # Uses the provided library function which handles baseline extraction,
    # relative weeks calculation, and categorical encoding.
    df_proc = process_tabular_data(df_meta, mode=mode)

    # Extract Static Features
    # Ensure all required columns defined in Config are present
    missing_cols = [c for c in Config.STATIC_COLS if c not in df_proc.columns]
    if missing_cols:
        raise ValueError(
            f"Processed dataframe missing required static columns: {missing_cols}"
        )

    X_base = df_proc[Config.STATIC_COLS].values.astype(np.float32)

    # Merge with Image Features
    if image_features is not None:
        # Validate alignment
        if len(X_base) != len(image_features):
            raise ValueError(
                f"Data length mismatch: Tabular ({len(X_base)}) vs Image ({len(image_features)})"
            )
        X_static = np.hstack([X_base, image_features])
    else:
        # Fallback if no image features provided (e.g. for baseline testing)
        X_static = X_base

    # Extract Time Variable (Relative Weeks)
    if "Relative_Weeks" not in df_proc.columns:
        raise ValueError("Relative_Weeks column missing from processed data.")
    weeks = df_proc["Relative_Weeks"].values.astype(np.float32)

    # Extract Target (FVC)
    # Note: Test set has placeholder FVC values (2000), which is acceptable here.
    y = np.array([])
    if "FVC" in df_proc.columns:
        y = df_proc["FVC"].values.astype(np.float32)

    # Extract IDs for submission mapping
    patient_weeks = np.array([])
    if "Patient_Week" in df_proc.columns:
        patient_weeks = df_proc["Patient_Week"].values.astype(str)

    # 3. Save to cache
    np.savez(
        cache_path, X_static=X_static, weeks=weeks, y=y, patient_weeks=patient_weeks
    )

    # Return as dictionary
    return {
        "X_static": X_static,
        "weeks": weeks,
        "y": y,
        "patient_weeks": patient_weeks,
    }


def create_interaction_features(X_static, weeks):
    """
    Generates time-dependent interaction terms for the FVC predictor.
    Constructs the feature matrix [X, t, X*t] to allow for patient-specific slopes.

    Args:
        X_static (np.array): Static features (N, D).
        weeks (np.array): Relative weeks (N,) or (N, 1).

    Returns:
        np.array: Full feature matrix (N, 2*D + 1).
    """
    # Ensure weeks is column vector (N, 1)
    if weeks.ndim == 1:
        t = weeks.reshape(-1, 1)
    else:
        t = weeks

    # Interaction: Multiply every static feature by time t
    # This creates the varying coefficients (slope depends on static features)
    X_interaction = X_static * t

    # Stack: [Static Features, Time, Interaction Terms]
    X_full = np.hstack([X_static, t, X_interaction])

    return X_full


def get_static_features(data_container):
    """
    Isolates time-invariant features for the Uncertainty (ElasticNet) model.

    Args:
        data_container (dict or np.array): Output from load_and_merge_data or raw array.

    Returns:
        np.array: Static features matrix.
    """
    if isinstance(data_container, dict):
        if "X_static" not in data_container:
            raise KeyError("Key 'X_static' not found in data dictionary.")
        return data_container["X_static"]
    elif isinstance(data_container, np.ndarray):
        # Assuming the input is already the static feature matrix
        return data_container
    else:
        raise TypeError(
            "Input must be a dictionary (from load_and_merge_data) or numpy array."
        )
