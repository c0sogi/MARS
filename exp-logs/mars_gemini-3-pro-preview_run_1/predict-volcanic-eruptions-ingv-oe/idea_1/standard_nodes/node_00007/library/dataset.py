import pandas as pd
import library.config as config
import library.feature_engineering as fe


def get_train_data(load_cached_data=True):
    """
    Loads the training and validation data, generating features if necessary.

    Args:
        load_cached_data (bool): If True, attempts to load features from the cache directory.
                                 If False or cache missing, re-computes features.

    Returns:
        tuple: (X_train, y_train, X_val, y_val)
               X_train, X_val: pd.DataFrame containing sensor statistics.
               y_train, y_val: pd.Series containing 'time_to_eruption'.
    """
    # Load metadata
    train_meta = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(config.VAL_METADATA_PATH)

    # Generate or load feature matrices
    # The build_feature_matrix function handles the caching logic internally
    df_train = fe.build_feature_matrix(
        train_meta, "train", load_cached_data=load_cached_data
    )
    df_val = fe.build_feature_matrix(val_meta, "val", load_cached_data=load_cached_data)

    # Define columns to exclude from features (IDs and Targets)
    target_col = "time_to_eruption"
    drop_cols = ["segment_id", target_col]

    # Prepare Training Data
    y_train = df_train[target_col]
    X_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])

    # Prepare Validation Data
    y_val = df_val[target_col]
    X_val = df_val.drop(columns=[c for c in drop_cols if c in df_val.columns])

    return X_train, y_train, X_val, y_val


def get_test_data(load_cached_data=True):
    """
    Loads the test data, generating features if necessary.

    Args:
        load_cached_data (bool): If True, attempts to load features from the cache directory.

    Returns:
        tuple: (X_test, segment_ids)
               X_test: pd.DataFrame containing sensor statistics.
               segment_ids: pd.Series containing the segment IDs for submission.
    """
    # Load metadata
    test_meta = pd.read_csv(config.TEST_METADATA_PATH)

    # Generate or load feature matrix
    df_test = fe.build_feature_matrix(
        test_meta, "test", load_cached_data=load_cached_data
    )

    # Extract Segment IDs for the submission file
    segment_ids = df_test["segment_id"]

    # Prepare Feature Matrix
    # Metadata for test usually contains time_to_eruption=0, so we must drop it if present
    drop_cols = ["segment_id", "time_to_eruption"]
    X_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])

    return X_test, segment_ids
