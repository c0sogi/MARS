import os
import gc
import numpy as np
import pandas as pd
import joblib
from scipy import signal, stats
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


# ==========================================
# Configuration
# ==========================================
class Config:
    # Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_11"
    SUBMISSION_DIR = "./submission"

    # Metadata Paths
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Data Properties
    SAMPLING_RATE = 100  # Hz (60001 samples / 600 seconds)
    SENSORS = [f"sensor_{i}" for i in range(1, 11)]

    # Signal Processing
    SG_WINDOW = 51
    SG_POLY = 2
    N_TEMPORAL_WINDOWS = 10

    # Training
    SEED = 42
    N_FOLDS = 5
    EARLY_STOPPING = 50

    # Hyperparameters
    LGBM_PARAMS = {
        "objective": "regression_l1",
        "metric": "mae",
        "verbosity": -1,
        "learning_rate": 0.05,
        "n_estimators": 3000,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "seed": 42,
        "n_jobs": -1,
    }

    XGB_PARAMS = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "learning_rate": 0.03,
        "n_estimators": 3000,
        "max_depth": 8,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "tree_method": "hist",
        "device": "cuda",
        "random_state": 42,
        "n_jobs": -1,
    }

    CAT_PARAMS = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": 3000,
        "learning_rate": 0.03,
        "depth": 8,
        "task_type": "GPU",
        "random_seed": 42,
        "verbose": 0,
    }

    RIDGE_ALPHA = 10.0


# Ensure working directory exists
os.makedirs(Config.WORKING_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# ==========================================
# Feature Engineering (Dual-Stream)
# ==========================================
def _calculate_stats(x, prefix):
    """Helper to calculate standard statistics for a signal."""
    return {
        f"{prefix}_mean": np.mean(x),
        f"{prefix}_std": np.std(x),
        f"{prefix}_min": np.min(x),
        f"{prefix}_max": np.max(x),
        f"{prefix}_q01": np.quantile(x, 0.01),
        f"{prefix}_q05": np.quantile(x, 0.05),
        f"{prefix}_q95": np.quantile(x, 0.95),
        f"{prefix}_q99": np.quantile(x, 0.99),
        f"{prefix}_rms": np.sqrt(np.mean(x**2)),
        f"{prefix}_range": np.max(x) - np.min(x),
    }


def process_segment(file_path, segment_id):
    """
    Processes a single sensor file using the Dual-Stream pipeline.
    """
    try:
        # Load data, fill NaNs with column mean to preserve offsets
        df = pd.read_csv(file_path, dtype="float32")
        df = df.fillna(df.mean())

        features = {}
        features["segment_id"] = int(segment_id)

        for sensor in Config.SENSORS:
            if sensor not in df.columns:
                continue

            raw_sig = df[sensor].values

            # --- Stream A: Raw Data (Extrema & Spectral) ---
            # 1. Basic Extrema
            features[f"{sensor}_raw_min"] = np.min(raw_sig)
            features[f"{sensor}_raw_max"] = np.max(raw_sig)
            features[f"{sensor}_raw_range"] = np.max(raw_sig) - np.min(raw_sig)

            # 2. Spectral Features (Band Power)
            f, Pxx = signal.periodogram(raw_sig, fs=Config.SAMPLING_RATE)
            features[f"{sensor}_spec_low"] = np.sum(Pxx[(f >= 0.1) & (f < 2.0)])
            features[f"{sensor}_spec_mid"] = np.sum(Pxx[(f >= 2.0) & (f < 10.0)])
            features[f"{sensor}_spec_high"] = np.sum(Pxx[(f >= 10.0) & (f < 20.0)])

            # 3. Temporal Windows (Flattened)
            wins = np.array_split(raw_sig, Config.N_TEMPORAL_WINDOWS)
            for w_idx, w_data in enumerate(wins):
                features[f"{sensor}_win{w_idx}_rms"] = np.sqrt(np.mean(w_data**2))
                features[f"{sensor}_win{w_idx}_mean"] = np.mean(w_data)

            # --- Stream B: Smoothed Data (Kinematics) ---
            # Apply Savitzky-Golay filter
            smooth_sig = signal.savgol_filter(raw_sig, Config.SG_WINDOW, Config.SG_POLY)

            # Derivatives
            vel = np.gradient(smooth_sig)
            acc = np.gradient(vel)

            # Statistics on Kinematics
            features.update(_calculate_stats(smooth_sig, f"{sensor}_smooth"))
            features.update(_calculate_stats(vel, f"{sensor}_vel"))
            features.update(_calculate_stats(acc, f"{sensor}_acc"))

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def _process_wrapper(row):
    """Wrapper for parallel processing."""
    full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
    return process_segment(full_path, row["segment_id"])


# ==========================================
# Data Management
# ==========================================
def get_dataset(metadata_path, cache_name, load_cached_data=True):
    """
    Loads dataset from cache or computes it from scratch.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Generating data for {cache_name}...")
    meta_df = pd.read_csv(metadata_path)

    # Parallel processing
    results = joblib.Parallel(n_jobs=12, backend="loky")(
        joblib.delayed(_process_wrapper)(row) for _, row in meta_df.iterrows()
    )

    # Filter Nones
    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)

    # Merge target if available
    if "time_to_eruption" in meta_df.columns:
        df = df.merge(
            meta_df[["segment_id", "time_to_eruption"]], on="segment_id", how="left"
        )

    # Save to cache
    print(f"Saving cache to {cache_path}")
    df.to_parquet(cache_path, index=False)

    return df


# ==========================================
# Data Reshaping for Siamese Network
# ==========================================
def reshape_for_siamese(df, is_train=True):
    """
    Reshapes Wide format (1 row per segment) to Long format (10 rows per segment).
    Returns X, y (if train), groups.
    """
    long_rows = []
    targets = []
    groups = []

    # Identify feature suffixes based on sensor_1
    sensor_1_cols = [c for c in df.columns if c.startswith("sensor_1_")]
    suffixes = [c.replace("sensor_1_", "") for c in sensor_1_cols]

    for _, row in df.iterrows():
        seg_id = row["segment_id"]
        target = row["time_to_eruption"] if is_train else 0

        for i in range(1, 11):
            sensor = f"sensor_{i}"
            # Extract features for this sensor
            feat_dict = {}
            valid_sensor = True
            for suff in suffixes:
                col_name = f"{sensor}_{suff}"
                if col_name in df.columns:
                    feat_dict[suff] = row[col_name]
                else:
                    valid_sensor = False
                    break

            if valid_sensor:
                long_rows.append(feat_dict)
                if is_train:
                    targets.append(target)
                groups.append(seg_id)

    X_long = pd.DataFrame(long_rows)
    y_long = np.array(targets) if is_train else None
    groups = np.array(groups)

    return X_long, y_long, groups


# ==========================================
# Model Training Stages
# ==========================================
def train_stage_1_siamese(train_df, val_df):
    """
    Stage 1: Siamese Sensor Encoder (LightGBM).
    Uses GroupKFold to prevent leakage.
    """
    print("\n=== Training Stage 1: Siamese Sensor Encoder ===")

    # Prepare data
    X_train_long, y_train_long, groups_train = reshape_for_siamese(
        train_df, is_train=True
    )
    X_val_long, y_val_long, _ = reshape_for_siamese(val_df, is_train=True)

    # We use the provided validation set for early stopping, but we also need OOFs for the training set.
    # To get proper OOFs for the *training* set, we must cross-validate on it.
    # The provided val_df is used as a holdout to check generalization.

    # Setup CV on Train Data for OOF generation
    gkf = GroupKFold(n_splits=Config.N_FOLDS)
    oof_preds = np.zeros(len(y_train_long))
    models = []

    for fold, (trn_idx, val_idx) in enumerate(
        gkf.split(X_train_long, y_train_long, groups_train)
    ):
        X_t, y_t = X_train_long.iloc[trn_idx], y_train_long[trn_idx]
        X_v, y_v = X_train_long.iloc[val_idx], y_train_long[val_idx]

        model = lgb.LGBMRegressor(**Config.LGBM_PARAMS)
        model.fit(
            X_t,
            y_t,
            eval_set=[(X_v, y_v)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING),
                lgb.log_evaluation(0),
            ],
        )

        oof_preds[val_idx] = model.predict(X_v)
        models.append(model)

        score = mean_absolute_error(y_v, oof_preds[val_idx])
        print(f"Stage 1 Fold {fold} MAE: {score}")

    # Predict on holdout validation set (average of folds)
    val_preds = np.zeros(len(X_val_long))
    for model in models:
        val_preds += model.predict(X_val_long) / Config.N_FOLDS

    print(f"Stage 1 Holdout MAE: {mean_absolute_error(y_val_long, val_preds)}")

    # Add predictions back to dataframes for Stage 2
    # We need to reshape predictions back to wide format

    def add_preds_to_df(df, preds, groups):
        # Create a temp df
        tmp = pd.DataFrame({"segment_id": groups, "pred": preds})
        # We have 10 preds per segment. We need to assign them to sensor_1_pred, sensor_2_pred...
        # This relies on the order being preserved from reshape_for_siamese
        # The order in reshape was: Segment 1 (S1...S10), Segment 2 (S1...S10)

        # A safer way:
        # Reshape preds to (N_segments, 10)
        preds_reshaped = preds.reshape(-1, 10)
        unique_groups = groups.reshape(-1, 10)[:, 0]  # Should be identical rows

        pred_cols = [f"stage1_pred_s{i}" for i in range(1, 11)]
        pred_df = pd.DataFrame(preds_reshaped, columns=pred_cols)
        pred_df["segment_id"] = unique_groups

        return df.merge(pred_df, on="segment_id", how="left")

    train_df_aug = add_preds_to_df(train_df, oof_preds, groups_train)
    # For validation, we need to regenerate the groups array matching X_val_long
    _, _, groups_val = reshape_for_siamese(val_df, is_train=True)
    val_df_aug = add_preds_to_df(val_df, val_preds, groups_val)

    return models, train_df_aug, val_df_aug


def train_stage_2_stacking(train_df, val_df):
    """
    Stage 2: Spatially-Coupled Stacking.
    Input: Original Features + Stage 1 Preds + Aggregates.
    Models: LGBM, XGB, CatBoost.
    """
    print("\n=== Training Stage 2: Spatially-Coupled Stacking ===")

    # Feature Engineering for Stage 2 (Aggregates of Stage 1 preds)
    pred_cols = [c for c in train_df.columns if c.startswith("stage1_pred_s")]

    for df in [train_df, val_df]:
        df["s1_mean"] = df[pred_cols].mean(axis=1)
        df["s1_std"] = df[pred_cols].std(axis=1)
        df["s1_max"] = df[pred_cols].max(axis=1)
        df["s1_min"] = df[pred_cols].min(axis=1)
        df["s1_range"] = df["s1_max"] - df["s1_min"]

    features = [
        c for c in train_df.columns if c not in ["segment_id", "time_to_eruption"]
    ]
    target = "time_to_eruption"

    X = train_df[features]
    y = train_df[target]
    X_val = val_df[features]
    y_val = val_df[target]

    # Stratified Split based on target bins
    num_bins = 10
    y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Containers
    lgbm_models = []
    xgb_models = []
    cat_models = []

    oof_lgbm = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))

    val_pred_lgbm = np.zeros(len(X_val))
    val_pred_xgb = np.zeros(len(X_val))
    val_pred_cat = np.zeros(len(X_val))

    for fold, (trn_idx, v_idx) in enumerate(skf.split(X, y_bins)):
        X_t, y_t = X.iloc[trn_idx], y.iloc[trn_idx]
        X_v, y_v = X.iloc[v_idx], y.iloc[v_idx]

        # 1. LightGBM
        lgb_model = lgb.LGBMRegressor(**Config.LGBM_PARAMS)
        lgb_model.fit(
            X_t,
            y_t,
            eval_set=[(X_v, y_v)],
            callbacks=[
                lgb.early_stopping(Config.EARLY_STOPPING),
                lgb.log_evaluation(0),
            ],
        )
        oof_lgbm[v_idx] = lgb_model.predict(X_v)
        val_pred_lgbm += lgb_model.predict(X_val) / Config.N_FOLDS
        lgbm_models.append(lgb_model)

        # 2. XGBoost
        xgb_model = xgb.XGBRegressor(**Config.XGB_PARAMS)
        xgb_model.fit(
            X_t, y_t, eval_set=[(X_v, y_v)], verbose=False
        )  # Early stopping handled by n_estimators/learning rate usually, or explicit
        oof_xgb[v_idx] = xgb_model.predict(X_v)
        val_pred_xgb += xgb_model.predict(X_val) / Config.N_FOLDS
        xgb_models.append(xgb_model)

        # 3. CatBoost
        cat_model = CatBoostRegressor(**Config.CAT_PARAMS)
        cat_model.fit(
            X_t,
            y_t,
            eval_set=(X_v, y_v),
            early_stopping_rounds=Config.EARLY_STOPPING,
            verbose=False,
        )
        oof_cat[v_idx] = cat_model.predict(X_v)
        val_pred_cat += cat_model.predict(X_val) / Config.N_FOLDS
        cat_models.append(cat_model)

        print(
            f"Stage 2 Fold {fold} - LGBM: {mean_absolute_error(y_v, oof_lgbm[v_idx]):.4f}, XGB: {mean_absolute_error(y_v, oof_xgb[v_idx]):.4f}, CAT: {mean_absolute_error(y_v, oof_cat[v_idx]):.4f}"
        )

    # Create Meta-Features
    train_meta = pd.DataFrame({"lgbm": oof_lgbm, "xgb": oof_xgb, "cat": oof_cat})

    val_meta = pd.DataFrame(
        {"lgbm": val_pred_lgbm, "xgb": val_pred_xgb, "cat": val_pred_cat}
    )

    return (lgbm_models, xgb_models, cat_models), train_meta, val_meta


def train_stage_3_meta(X_train, y_train, X_val, y_val):
    """
    Stage 3: Ridge Meta-Learner.
    """
    print("\n=== Training Stage 3: Meta-Learner ===")
    model = Ridge(alpha=Config.RIDGE_ALPHA)
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    print(f"Final Train MAE: {mean_absolute_error(y_train, train_preds)}")
    print(f"Final Val MAE: {mean_absolute_error(y_val, val_preds)}")
    print(f"Meta Coefficients: {model.coef_}")

    return model


# ==========================================
# Inference Pipeline
# ==========================================
def generate_submission(load_cached_data=True):
    """
    Runs the full inference pipeline and generates submission.csv.
    """
    # 1. Load Data
    train_df = get_dataset(Config.TRAIN_META, "train_features", load_cached_data)
    val_df = get_dataset(Config.VAL_META, "val_features", load_cached_data)
    test_df = get_dataset(Config.TEST_META, "test_features", load_cached_data)

    # 2. Train Stage 1
    s1_models, train_aug, val_aug = train_stage_1_siamese(train_df, val_df)

    # 3. Train Stage 2
    s2_models, train_meta, val_meta = train_stage_2_stacking(train_aug, val_aug)

    # 4. Train Stage 3
    s3_model = train_stage_3_meta(
        train_meta, train_aug["time_to_eruption"], val_meta, val_aug["time_to_eruption"]
    )

    # 5. Inference on Test
    print("\n=== Generating Submission ===")

    # Stage 1 Inference
    X_test_long, _, groups_test = reshape_for_siamese(test_df, is_train=False)

    s1_preds = np.zeros(len(X_test_long))
    for model in s1_models:
        s1_preds += model.predict(X_test_long) / len(s1_models)

    # Reshape S1 preds to Wide
    # Reshape logic: (N_segments, 10)
    s1_preds_reshaped = s1_preds.reshape(-1, 10)
    unique_groups = groups_test.reshape(-1, 10)[:, 0]

    pred_cols = [f"stage1_pred_s{i}" for i in range(1, 11)]
    pred_df = pd.DataFrame(s1_preds_reshaped, columns=pred_cols)
    pred_df["segment_id"] = unique_groups

    test_aug = test_df.merge(pred_df, on="segment_id", how="left")

    # Stage 2 Features
    test_aug["s1_mean"] = test_aug[pred_cols].mean(axis=1)
    test_aug["s1_std"] = test_aug[pred_cols].std(axis=1)
    test_aug["s1_max"] = test_aug[pred_cols].max(axis=1)
    test_aug["s1_min"] = test_aug[pred_cols].min(axis=1)
    test_aug["s1_range"] = test_aug["s1_max"] - test_aug["s1_min"]

    features = [
        c for c in train_aug.columns if c not in ["segment_id", "time_to_eruption"]
    ]
    X_test = test_aug[features]

    # Stage 2 Inference
    lgbm_models, xgb_models, cat_models = s2_models

    pred_lgbm = np.zeros(len(X_test))
    for m in lgbm_models:
        pred_lgbm += m.predict(X_test) / len(lgbm_models)

    pred_xgb = np.zeros(len(X_test))
    for m in xgb_models:
        pred_xgb += m.predict(X_test) / len(xgb_models)

    pred_cat = np.zeros(len(X_test))
    for m in cat_models:
        pred_cat += m.predict(X_test) / len(cat_models)

    test_meta = pd.DataFrame({"lgbm": pred_lgbm, "xgb": pred_xgb, "cat": pred_cat})

    # Stage 3 Inference
    final_preds = s3_model.predict(test_meta)

    # Save Submission
    submission = pd.DataFrame(
        {"segment_id": test_aug["segment_id"], "time_to_eruption": final_preds}
    )

    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
