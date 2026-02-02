import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import library.config as config


def load_raw_data():
    """
    Loads raw parquet files from the metadata directory defined in config.
    """
    print(f"Loading raw data from {config.INPUT_DIR}...")
    train_df = pd.read_parquet(config.TRAIN_PATH)
    val_df = pd.read_parquet(config.VAL_PATH)
    test_df = pd.read_parquet(config.TEST_PATH)
    return train_df, val_df, test_df


def create_interaction_features(df):
    """
    Generates interaction features based on INTERACTION_PAIRS defined in config.
    New features are named as '{col1}_x_{col2}'.
    """
    df_out = df.copy()
    for col1, col2 in config.INTERACTION_PAIRS:
        # Ensure columns exist before interaction
        if col1 in df_out.columns and col2 in df_out.columns:
            new_col_name = f"{col1}_x_{col2}"
            df_out[new_col_name] = df_out[col1] * df_out[col2]
    return df_out


def get_continuous_columns(df):
    """
    Identifies continuous columns.
    Heuristic: Exclude 'Id', 'Cover_Type', and binary columns (Soil_Type*, Wilderness_Area*).
    """
    cols = []
    for col in df.columns:
        if col in [config.ID_COL, config.TARGET_COL]:
            continue
        # Binary features in this dataset start with Soil_Type or Wilderness_Area
        if col.startswith("Soil_Type") or col.startswith("Wilderness_Area"):
            continue
        cols.append(col)
    return cols


def preprocess_for_nn(X_train, X_val, X_test):
    """
    Applies Standard Scaling to continuous features for Neural Network training.
    Fits on Train, transforms Train, Val, and Test.
    """
    scaler = StandardScaler()

    # Identify continuous columns (including newly created interaction terms if they are continuous-like)
    # Note: Interaction terms (Continuous * Binary) are treated as continuous for scaling purposes.
    cont_cols = get_continuous_columns(X_train)

    print(f"Scaling {len(cont_cols)} continuous features for Neural Network...")

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    # Fit on train, transform all
    X_train_scaled[cont_cols] = scaler.fit_transform(X_train[cont_cols])
    X_val_scaled[cont_cols] = scaler.transform(X_val[cont_cols])
    X_test_scaled[cont_cols] = scaler.transform(X_test[cont_cols])

    return X_train_scaled, X_val_scaled, X_test_scaled


def preprocess_data(load_cached_data=True):
    """
    Main data processing pipeline with caching.

    Returns:
        dict: Contains 'tree' and 'nn' keys, each mapping to a tuple:
              (X_train, y_train, X_val, y_val, X_test, test_ids)
    """
    # Define cache file paths
    cache_files = {
        "X_train_tree": os.path.join(config.CACHE_DIR, "X_train_tree.parquet"),
        "X_val_tree": os.path.join(config.CACHE_DIR, "X_val_tree.parquet"),
        "X_test_tree": os.path.join(config.CACHE_DIR, "X_test_tree.parquet"),
        "X_train_nn": os.path.join(config.CACHE_DIR, "X_train_nn.parquet"),
        "X_val_nn": os.path.join(config.CACHE_DIR, "X_val_nn.parquet"),
        "X_test_nn": os.path.join(config.CACHE_DIR, "X_test_nn.parquet"),
        "y_train": os.path.join(config.CACHE_DIR, "y_train.npy"),
        "y_val": os.path.join(config.CACHE_DIR, "y_val.npy"),
        "test_ids": os.path.join(config.CACHE_DIR, "test_ids.npy"),
    }

    # 1. Try to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(path) for path in cache_files.values())
        if all_exist:
            print("Loading processed data from cache...")
            try:
                X_train_tree = pd.read_parquet(cache_files["X_train_tree"])
                X_val_tree = pd.read_parquet(cache_files["X_val_tree"])
                X_test_tree = pd.read_parquet(cache_files["X_test_tree"])

                X_train_nn = pd.read_parquet(cache_files["X_train_nn"])
                X_val_nn = pd.read_parquet(cache_files["X_val_nn"])
                X_test_nn = pd.read_parquet(cache_files["X_test_nn"])

                y_train = np.load(cache_files["y_train"])
                y_val = np.load(cache_files["y_val"])
                test_ids = np.load(cache_files["test_ids"])

                return {
                    "tree": (
                        X_train_tree,
                        y_train,
                        X_val_tree,
                        y_val,
                        X_test_tree,
                        test_ids,
                    ),
                    "nn": (X_train_nn, y_train, X_val_nn, y_val, X_test_nn, test_ids),
                }
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
        else:
            print("Cache incomplete or missing. Recomputing...")

    # 2. Compute from scratch
    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Load raw data
    train_df, val_df, test_df = load_raw_data()

    # Extract IDs and Targets
    test_ids = test_df[config.ID_COL].values

    # Map targets using CLASS_MAP
    print("Mapping target classes...")
    y_train = train_df[config.TARGET_COL].map(config.CLASS_MAP).values.astype(np.int64)
    y_val = val_df[config.TARGET_COL].map(config.CLASS_MAP).values.astype(np.int64)

    # Drop Id and Target from features
    drop_cols_train = [config.ID_COL, config.TARGET_COL]
    drop_cols_test = [config.ID_COL]

    X_train = train_df.drop(columns=drop_cols_train, errors="ignore")
    X_val = val_df.drop(columns=drop_cols_train, errors="ignore")
    X_test = test_df.drop(columns=drop_cols_test, errors="ignore")

    # Feature Engineering (Interactions)
    print("Generating interaction features...")
    X_train = create_interaction_features(X_train)
    X_val = create_interaction_features(X_val)
    X_test = create_interaction_features(X_test)

    # Prepare Tree Data (Raw + Interactions)
    X_train_tree = X_train.copy()
    X_val_tree = X_val.copy()
    X_test_tree = X_test.copy()

    # Prepare NN Data (Skipped to save resources - Cite solution_lesson_node_00004)
    # X_train_nn, X_val_nn, X_test_nn = preprocess_for_nn(X_train, X_val, X_test)

    # Save to cache
    print("Saving processed data to cache...")
    X_train_tree.to_parquet(cache_files["X_train_tree"], index=False)
    X_val_tree.to_parquet(cache_files["X_val_tree"], index=False)
    X_test_tree.to_parquet(cache_files["X_test_tree"], index=False)

    # X_train_nn.to_parquet(cache_files["X_train_nn"], index=False)
    # X_val_nn.to_parquet(cache_files["X_val_nn"], index=False)
    # X_test_nn.to_parquet(cache_files["X_test_nn"], index=False)

    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["test_ids"], test_ids)

    return {
        "tree": (X_train_tree, y_train, X_val_tree, y_val, X_test_tree, test_ids),
        "nn": (None, None, None, None, None, None),
    }
