import os
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================

TRAIN_PATH = "./metadata/train.parquet"
VAL_PATH = "./metadata/val.parquet"
TEST_PATH = "./metadata/test.parquet"
SUBMISSION_PATH = "./submission/submission.csv"
CACHE_DIR = "./working/idea_7_v2/"

# Spatial Constraints (NYC Bounding Box)
NYC_BOUNDING_BOX = {
    "min_lat": 40.5,
    "max_lat": 40.9,
    "min_lon": -74.3,
    "max_lon": -73.7,
}

# Grid Discretization
# Precision for rounding coordinates to create grid cells
# 3 decimal places is approx 100m resolution
GRID_PRECISION = 3

# Target Sanitization
# Cite solution_lesson_node_00017: Sanitize target to stabilize L2 loss
# Cite solution_lesson_node_00018: Sanitize validation to avoid outlier dominance
FARE_MIN = 0
FARE_MAX = 500

# Target Encoding Smoothing
SMOOTHING_ALPHA = 10

# Data Subsampling
# Train on a stable subset to avoid L2 loss instability with outliers
SUBSAMPLE_SIZE = 5_000_000
RANDOM_STATE = 42

# XGBoost Hyperparameters
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "learning_rate": 0.02,
    "max_depth": 9,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": 12,
    "tree_method": "hist",
    "device": "cuda",
    "random_state": RANDOM_STATE,
}

TRAIN_ROUNDS = 10000
EARLY_STOPPING_ROUNDS = 100

# ==========================================
# IMPLEMENTATION LOGIC
# ==========================================


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates Haversine distance between two sets of coordinates."""
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def add_spatial_features(df):
    """Adds basic spatial features: distances, coordinate differences."""
    df["dist_km"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["abs_diff_lon"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df["abs_diff_lat"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
    return df


def add_time_features(df):
    """Adds temporal features from pickup_datetime."""
    if not np.issubdtype(df["pickup_datetime"].dtype, np.datetime64):
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    df["hour"] = df["pickup_datetime"].dt.hour
    df["year"] = df["pickup_datetime"].dt.year
    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek
    return df


def process_data(load_cached_data=True):
    """
    Implements the Two-Stage Global-Local strategy.
    Stage 1: Global Feature Extraction (OOF Target Encoding on full data).
    Stage 2: Subsampling and Feature Engineering.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_files = {
        "X_train": os.path.join(CACHE_DIR, "X_train.parquet"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "X_val": os.path.join(CACHE_DIR, "X_val.parquet"),
        "y_val": os.path.join(CACHE_DIR, "y_val.npy"),
        "global_map": os.path.join(CACHE_DIR, "global_route_map.parquet"),
    }

    if load_cached_data and all(os.path.exists(f) for f in cache_files.values()):
        print("Loading cached data...")
        X_train = pd.read_parquet(cache_files["X_train"])
        y_train = np.load(cache_files["y_train"])
        X_val = pd.read_parquet(cache_files["X_val"])
        y_val = np.load(cache_files["y_val"])
        global_route_map = pd.read_parquet(cache_files["global_map"])
        return X_train, y_train, X_val, y_val, global_route_map

    print("Processing data from scratch...")

    # Load Full Training Data (44M rows)
    train_full = pd.read_parquet(TRAIN_PATH)

    # --- STAGE 1: Global Feature Extraction ---
    print("Stage 1: Discretizing and OOF Encoding...")

    # Discretize coordinates for grid cells
    train_full["p_lat_r"] = np.round(train_full["pickup_latitude"], GRID_PRECISION)
    train_full["p_lon_r"] = np.round(train_full["pickup_longitude"], GRID_PRECISION)
    train_full["d_lat_r"] = np.round(train_full["dropoff_latitude"], GRID_PRECISION)
    train_full["d_lon_r"] = np.round(train_full["dropoff_longitude"], GRID_PRECISION)

    group_cols = ["p_lat_r", "p_lon_r", "d_lat_r", "d_lon_r"]

    # K-Fold OOF Encoding
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_full["oof_fare"] = np.nan

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_full)):
        # Calculate means on the training part of the fold
        fold_train = train_full.iloc[train_idx]
        route_stats = fold_train.groupby(group_cols)["fare_amount"].mean()

        # Map to the validation part of the fold
        fold_val = train_full.iloc[val_idx][group_cols]
        mapped_fares = fold_val.join(route_stats, on=group_cols, rsuffix="_mean")[
            "fare_amount"
        ]

        train_full.iloc[val_idx, train_full.columns.get_loc("oof_fare")] = mapped_fares

    # Fill NaNs with global mean
    global_mean = train_full["fare_amount"].mean()
    train_full["oof_fare"] = train_full["oof_fare"].fillna(global_mean)

    # Create Global Route Map for Test/Val sets (using all training data)
    print("Creating Global Route Map...")
    global_route_map = (
        train_full.groupby(group_cols)["fare_amount"].mean().reset_index()
    )
    global_route_map.rename(columns={"fare_amount": "global_avg_fare"}, inplace=True)

    # --- STAGE 2: Subsampling & Formatting ---
    print("Stage 2: Subsampling and Feature Engineering...")

    # Subsample for training stability
    if SUBSAMPLE_SIZE < len(train_full):
        train_df = train_full.sample(n=SUBSAMPLE_SIZE, random_state=RANDOM_STATE).copy()
    else:
        train_df = train_full.copy()

    del train_full
    gc.collect()

    # Load Validation Set
    val_df = pd.read_parquet(VAL_PATH)

    # Discretize Validation Set
    val_df["p_lat_r"] = np.round(val_df["pickup_latitude"], GRID_PRECISION)
    val_df["p_lon_r"] = np.round(val_df["pickup_longitude"], GRID_PRECISION)
    val_df["d_lat_r"] = np.round(val_df["dropoff_latitude"], GRID_PRECISION)
    val_df["d_lon_r"] = np.round(val_df["dropoff_longitude"], GRID_PRECISION)

    # Apply Global Map to Validation
    val_df = val_df.merge(global_route_map, on=group_cols, how="left")
    val_df["oof_fare"] = val_df["global_avg_fare"].fillna(global_mean)

    # Feature Engineering
    def engineer_features(df):
        df = add_spatial_features(df)
        df = add_time_features(df)
        return df

    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)

    features = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "dist_km",
        "abs_diff_lon",
        "abs_diff_lat",
        "hour",
        "year",
        "dayofweek",
        "oof_fare",
    ]

    X_train = train_df[features]
    y_train = train_df["fare_amount"].values
    X_val = val_df[features]
    y_val = val_df["fare_amount"].values

    # Cache
    print("Caching processed data...")
    X_train.to_parquet(cache_files["X_train"])
    np.save(cache_files["y_train"], y_train)
    X_val.to_parquet(cache_files["X_val"])
    np.save(cache_files["y_val"], y_val)
    global_route_map.to_parquet(cache_files["global_map"])

    return X_train, y_train, X_val, y_val, global_route_map


def train_model(X_train, y_train, X_val, y_val):
    """Trains the XGBoost model."""
    print(f"Training XGBoost on {len(X_train)} samples...")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=TRAIN_ROUNDS,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=50,
    )

    return model


def generate_submission(model, global_route_map):
    """Generates predictions for the test set."""
    print("Generating submission...")
    test_df = pd.read_parquet(TEST_PATH)

    # Discretize
    test_df["p_lat_r"] = np.round(test_df["pickup_latitude"], GRID_PRECISION)
    test_df["p_lon_r"] = np.round(test_df["pickup_longitude"], GRID_PRECISION)
    test_df["d_lat_r"] = np.round(test_df["dropoff_latitude"], GRID_PRECISION)
    test_df["d_lon_r"] = np.round(test_df["dropoff_longitude"], GRID_PRECISION)

    # Merge Global Map
    group_cols = ["p_lat_r", "p_lon_r", "d_lat_r", "d_lon_r"]
    test_df = test_df.merge(global_route_map, on=group_cols, how="left")

    # Fallback for unknown routes
    fallback_mean = global_route_map["global_avg_fare"].mean()
    test_df["oof_fare"] = test_df["global_avg_fare"].fillna(fallback_mean)

    # Features
    test_df = add_spatial_features(test_df)
    test_df = add_time_features(test_df)

    features = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "dist_km",
        "abs_diff_lon",
        "abs_diff_lat",
        "hour",
        "year",
        "dayofweek",
        "oof_fare",
    ]

    X_test = test_df[features]
    dtest = xgb.DMatrix(X_test)

    preds = model.predict(dtest)

    # Post-processing: Floor at $2.50
    preds = np.maximum(preds, 2.50)

    submission = pd.DataFrame({"key": test_df["key"], "fare_amount": preds})

    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
