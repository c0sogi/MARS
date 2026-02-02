import os
import numpy as np
import pandas as pd
import ase.io
from library.config import Config
from library.feature_extraction import extract_all_features


def transform_targets(y):
    """
    Applies log(1+x) transformation to targets to handle skewness and ensure positivity.
    """
    return np.log1p(y)


def inverse_transform_targets(y_log):
    """
    Applies exp(x)-1 transformation to revert predictions to original scale.
    """
    return np.expm1(y_log)


def process_geometries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Iterates over the dataframe, loads geometry files using ASE, and extracts features.
    Returns a DataFrame of extracted features corresponding to the input rows.
    """
    features_list = []

    # Iterate through metadata
    for idx, row in df.iterrows():
        # Construct full file path relative to the input directory
        # row['file_path'] comes from metadata, e.g., 'train/1/geometry.xyz'
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        try:
            # Load atoms object
            atoms = ase.io.read(full_path)

            # Extract features using the library function
            # This includes Macroscopic, RDF, Local Site, and Network Topology features
            feats = extract_all_features(atoms)
            features_list.append(feats)

        except Exception as e:
            print(f"Error processing {full_path}: {e}")
            # Append empty dict which results in a row of NaNs
            features_list.append({})

    # Convert list of dicts to DataFrame
    features_df = pd.DataFrame(features_list)
    return features_df


def _load_and_cache_data(
    metadata_path: str, cache_path: str, load_cached_data: bool
) -> pd.DataFrame:
    """
    Internal helper to handle loading metadata, processing geometries, and caching.
    Implements the logic: Load Cache -> If Fail/Force -> Compute -> Save Cache.
    """
    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing data from {metadata_path}...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Debug mode: sample subset to speed up development iteration
    if Config.DEBUG_MODE:
        print(f"Debug Mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    # Extract geometric features
    geo_features = process_geometries(df)

    # Reset indices to ensure alignment before concatenation
    df = df.reset_index(drop=True)
    geo_features = geo_features.reset_index(drop=True)

    # Concatenate original metadata with extracted features
    combined_df = pd.concat([df, geo_features], axis=1)

    # Drop 'file_path' as it is not a predictive feature
    if "file_path" in combined_df.columns:
        combined_df = combined_df.drop(columns=["file_path"])

    # Save to cache
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    combined_df.to_parquet(cache_path, index=False)
    print(f"Saved processed data to {cache_path}")

    return combined_df


def preprocess_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_cols: list,
):
    """
    Cleans features by dropping constant columns (based on training set statistics),
    separates features (X) from targets (y), and applies log transformation to targets.
    """
    # Identify feature columns (exclude targets and id)
    # 'id' is preserved in test_df output for submission creation
    exclude = target_cols + ["id"]
    feature_cols = [c for c in train_df.columns if c not in exclude]

    # 1. Drop constant columns based on TRAIN set statistics
    # We use dropna=True to ignore NaNs when counting unique values.
    # If nunique <= 1, the column is either all NaNs or has a single constant value.
    constant_cols = []
    for col in feature_cols:
        if train_df[col].nunique(dropna=True) <= 1:
            constant_cols.append(col)

    if constant_cols:
        print(f"Dropping {len(constant_cols)} constant columns found in training set.")
        train_df = train_df.drop(columns=constant_cols)
        val_df = val_df.drop(columns=constant_cols)
        test_df = test_df.drop(columns=constant_cols)

        # Update feature list
        feature_cols = [c for c in feature_cols if c not in constant_cols]

    # Prepare X and y
    # Training set
    X_train = train_df[feature_cols]
    y_train = train_df[target_cols]

    # Validation set
    X_val = val_df[feature_cols]
    y_val = val_df[target_cols]

    # Test set (no targets)
    X_test = test_df[feature_cols]
    test_ids = test_df["id"]

    # Apply Log Transformation to targets
    y_train_log = transform_targets(y_train)
    y_val_log = transform_targets(y_val)

    return (X_train, y_train_log), (X_val, y_val_log), (X_test, test_ids)


def get_datasets(load_cached_data=True):
    """
    Orchestrates the data loading, processing, and preprocessing pipeline.
    Returns prepared (X, y) tuples for train/val and (X, ids) for test.
    """
    # Define paths from Config
    train_meta = Config.TRAIN_METADATA_PATH
    val_meta = Config.VAL_METADATA_PATH
    test_meta = Config.TEST_METADATA_PATH

    train_cache = Config.TRAIN_FEATURES_PATH
    val_cache = Config.VAL_FEATURES_PATH
    test_cache = Config.TEST_FEATURES_PATH

    # Load and process each split (with caching)
    train_df = _load_and_cache_data(train_meta, train_cache, load_cached_data)
    val_df = _load_and_cache_data(val_meta, val_cache, load_cached_data)
    test_df = _load_and_cache_data(test_meta, test_cache, load_cached_data)

    # Define target columns
    targets = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    # Preprocess (clean features, split X/y, transform targets)
    (X_train, y_train), (X_val, y_val), (X_test, ids) = preprocess_features(
        train_df, val_df, test_df, targets
    )

    return (X_train, y_train), (X_val, y_val), (X_test, ids)
