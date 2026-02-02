import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error

from library.config import Config
from library.utils import seed_everything, save_model, load_model
from library.data_loader import generate_dataset
from library.model_factory import get_stage1_model, get_stage2_models, get_stage3_model


# ==========================================
# Data Reshaping Utilities
# ==========================================
def reshape_for_siamese(df, is_train=True):
    """
    Reshapes Wide format (1 row per segment) to Long format (10 rows per segment).

    Args:
        df (pd.DataFrame): Input dataframe in wide format.
        is_train (bool): Whether the dataframe contains target labels.

    Returns:
        X_long (pd.DataFrame): Reshaped features.
        y_long (np.array or None): Targets repeated for each sensor.
        groups (np.array): Segment IDs for grouping.
    """
    long_rows = []
    targets = []
    groups = []

    # Identify feature suffixes based on sensor_1
    # We assume features are named like 'sensor_1_mean', 'sensor_1_std', etc.
    sensor_1_cols = [c for c in df.columns if c.startswith("sensor_1_")]
    suffixes = [c.replace("sensor_1_", "") for c in sensor_1_cols]

    # Pre-calculate column mappings for speed
    sensor_maps = []
    for i in range(1, 11):
        sensor = f"sensor_{i}"
        s_map = {f"{sensor}_{suff}": suff for suff in suffixes}
        sensor_maps.append((sensor, s_map))

    # Iterate and reshape
    # Using to_dict('records') is often faster than iterrows for large DFs
    records = df.to_dict("records")

    for row in records:
        seg_id = row["segment_id"]
        target = row.get("time_to_eruption", 0) if is_train else 0

        for sensor, s_map in sensor_maps:
            # Check if this sensor exists in the row (it should, based on data loader)
            # We construct the feature dict for this sensor instance
            feat_dict = {}
            valid_sensor = True

            for col_name, suff in s_map.items():
                if col_name in row:
                    feat_dict[suff] = row[col_name]
                else:
                    # Should not happen if data loader is consistent, but safety check
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


def pivot_predictions_to_wide(groups, preds):
    """
    Pivots long-format predictions back to wide format (1 row per segment).
    Assumes predictions are ordered by segment (S1, S2... S10).
    """
    # Reshape: (N_samples, ) -> (N_segments, 10)
    # We rely on the order being preserved from reshape_for_siamese
    # reshape_for_siamese iterates row by row, then sensor 1..10

    n_sensors = 10
    if len(preds) % n_sensors != 0:
        raise ValueError(
            f"Prediction length {len(preds)} is not divisible by {n_sensors}"
        )

    n_segments = len(preds) // n_sensors
    preds_reshaped = preds.reshape(n_segments, n_sensors)

    # Extract unique segment IDs (taking every 10th element)
    unique_groups = groups[::n_sensors]

    pred_cols = [f"stage1_pred_s{i}" for i in range(1, 11)]
    pred_df = pd.DataFrame(preds_reshaped, columns=pred_cols)
    pred_df["segment_id"] = unique_groups

    return pred_df


# ==========================================
# Feature Engineering for Stage 2
# ==========================================
def add_stage2_features(df):
    """
    Adds aggregate features based on Stage 1 predictions.
    """
    pred_cols = [c for c in df.columns if c.startswith("stage1_pred_s")]

    if not pred_cols:
        return df

    df["s1_mean"] = df[pred_cols].mean(axis=1)
    df["s1_std"] = df[pred_cols].std(axis=1)
    df["s1_max"] = df[pred_cols].max(axis=1)
    df["s1_min"] = df[pred_cols].min(axis=1)
    df["s1_range"] = df["s1_max"] - df["s1_min"]

    return df


# ==========================================
# Training Stages
# ==========================================
def train_stage1_siamese(train_df, val_df, debug=False):
    print("\n=== Training Stage 1: Siamese Sensor Encoder ===")

    # Reshape Data
    X_train_long, y_train_long, groups_train = reshape_for_siamese(
        train_df, is_train=True
    )
    X_val_long, y_val_long, groups_val = reshape_for_siamese(val_df, is_train=True)

    # Setup CV
    gkf = GroupKFold(n_splits=Config.N_FOLDS)
    oof_preds = np.zeros(len(y_train_long))
    models = []

    # Train Loop
    for fold, (trn_idx, val_idx) in enumerate(
        gkf.split(X_train_long, y_train_long, groups_train)
    ):
        X_t, y_t = X_train_long.iloc[trn_idx], y_train_long[trn_idx]
        X_v, y_v = X_train_long.iloc[val_idx], y_train_long[val_idx]

        model = get_stage1_model(n_estimators=100 if debug else None)

        model.fit(
            X_t,
            y_t,
            eval_set=[(X_v, y_v)],
            callbacks=[
                pd.io.common.os.sys.modules["lightgbm"].early_stopping(
                    stopping_rounds=Config.EARLY_STOPPING
                ),
                pd.io.common.os.sys.modules["lightgbm"].log_evaluation(0),
            ],
        )

        oof_preds[val_idx] = model.predict(X_v)
        models.append(model)

        score = mean_absolute_error(y_v, oof_preds[val_idx])
        print(f"Stage 1 Fold {fold} MAE: {score}")

    # Validation Predictions (Average of folds)
    val_preds = np.zeros(len(X_val_long))
    for model in models:
        val_preds += model.predict(X_val_long) / Config.N_FOLDS

    val_score = mean_absolute_error(y_val_long, val_preds)
    print(f"Stage 1 Holdout MAE: {val_score}")

    # Pivot predictions back to wide format
    train_preds_df = pivot_predictions_to_wide(groups_train, oof_preds)
    val_preds_df = pivot_predictions_to_wide(groups_val, val_preds)

    # Merge back to original dataframes
    train_aug = train_df.merge(train_preds_df, on="segment_id", how="left")
    val_aug = val_df.merge(val_preds_df, on="segment_id", how="left")

    return models, train_aug, val_aug


def train_stage2_stacking(train_df, val_df, debug=False):
    print("\n=== Training Stage 2: Spatially-Coupled Stacking ===")

    # Feature Engineering
    train_df = add_stage2_features(train_df)
    val_df = add_stage2_features(val_df)

    features = [
        c for c in train_df.columns if c not in ["segment_id", "time_to_eruption"]
    ]
    target = "time_to_eruption"

    X = train_df[features]
    y = train_df[target]
    X_val = val_df[features]
    y_val = val_df[target]

    # Stratified Split
    num_bins = 10
    y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Model Containers
    models_dict = {"lgbm": [], "xgb": [], "cat": []}

    # OOF Containers
    oof_preds = {k: np.zeros(len(X)) for k in models_dict.keys()}
    val_preds = {k: np.zeros(len(X_val)) for k in models_dict.keys()}

    for fold, (trn_idx, v_idx) in enumerate(skf.split(X, y_bins)):
        X_t, y_t = X.iloc[trn_idx], y.iloc[trn_idx]
        X_v, y_v = X.iloc[v_idx], y.iloc[v_idx]

        # Get fresh models
        current_models = get_stage2_models(n_estimators=100 if debug else None)

        # 1. LightGBM
        current_models["lgbm"].fit(
            X_t,
            y_t,
            eval_set=[(X_v, y_v)],
            callbacks=[
                pd.io.common.os.sys.modules["lightgbm"].early_stopping(
                    Config.EARLY_STOPPING
                ),
                pd.io.common.os.sys.modules["lightgbm"].log_evaluation(0),
            ],
        )
        oof_preds["lgbm"][v_idx] = current_models["lgbm"].predict(X_v)
        val_preds["lgbm"] += current_models["lgbm"].predict(X_val) / Config.N_FOLDS
        models_dict["lgbm"].append(current_models["lgbm"])

        # 2. XGBoost
        current_models["xgb"].fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
        oof_preds["xgb"][v_idx] = current_models["xgb"].predict(X_v)
        val_preds["xgb"] += current_models["xgb"].predict(X_val) / Config.N_FOLDS
        models_dict["xgb"].append(current_models["xgb"])

        # 3. CatBoost
        current_models["cat"].fit(
            X_t,
            y_t,
            eval_set=(X_v, y_v),
            early_stopping_rounds=Config.EARLY_STOPPING,
            verbose=False,
        )
        oof_preds["cat"][v_idx] = current_models["cat"].predict(X_v)
        val_preds["cat"] += current_models["cat"].predict(X_val) / Config.N_FOLDS
        models_dict["cat"].append(current_models["cat"])

        print(
            f"Stage 2 Fold {fold} - "
            f"LGBM: {mean_absolute_error(y_v, oof_preds['lgbm'][v_idx])} "
            f"XGB: {mean_absolute_error(y_v, oof_preds['xgb'][v_idx])} "
            f"CAT: {mean_absolute_error(y_v, oof_preds['cat'][v_idx])}"
        )

    # Prepare Meta-Features
    train_meta = pd.DataFrame(oof_preds)
    val_meta = pd.DataFrame(val_preds)

    return models_dict, train_meta, val_meta, features


def train_stage3_meta(train_meta, y_train, val_meta, y_val):
    print("\n=== Training Stage 3: Meta-Learner ===")

    model = get_stage3_model()
    model.fit(train_meta, y_train)

    train_preds = model.predict(train_meta)
    val_preds = model.predict(val_meta)

    print(f"Final Train MAE: {mean_absolute_error(y_train, train_preds)}")
    print(f"Final Val MAE: {mean_absolute_error(y_val, val_preds)}")
    print(f"Meta Coefficients: {model.coef_}")

    return model


# ==========================================
# Main Orchestrator
# ==========================================
def run_training(load_cached_data=True, debug=False):
    seed_everything(Config.SEED)

    # 1. Load Data
    train_df = generate_dataset(
        Config.TRAIN_META, "train_features", load_cached_data, debug
    )
    val_df = generate_dataset(Config.VAL_META, "val_features", load_cached_data, debug)

    # 2. Stage 1
    s1_models, train_aug, val_aug = train_stage1_siamese(train_df, val_df, debug)

    # 3. Stage 2
    s2_models, train_meta, val_meta, s2_features = train_stage2_stacking(
        train_aug, val_aug, debug
    )

    # 4. Stage 3
    s3_model = train_stage3_meta(
        train_meta, train_aug["time_to_eruption"], val_meta, val_aug["time_to_eruption"]
    )

    # 5. Save Artifacts
    print("\nSaving models...")
    save_model(s1_models, os.path.join(Config.WORKING_DIR, "stage1_models.pkl"))
    save_model(s2_models, os.path.join(Config.WORKING_DIR, "stage2_models.pkl"))
    save_model(s3_model, os.path.join(Config.WORKING_DIR, "stage3_model.pkl"))
    save_model(s2_features, os.path.join(Config.WORKING_DIR, "stage2_features.pkl"))

    return s1_models, s2_models, s3_model, s2_features, val_aug


def generate_submission_file(load_cached_data=True):
    print("\n=== Generating Submission ===")
    seed_everything(Config.SEED)

    # Load Models and Features
    try:
        s1_models = load_model(os.path.join(Config.WORKING_DIR, "stage1_models.pkl"))
        s2_models = load_model(os.path.join(Config.WORKING_DIR, "stage2_models.pkl"))
        s3_model = load_model(os.path.join(Config.WORKING_DIR, "stage3_model.pkl"))
        s2_features = load_model(
            os.path.join(Config.WORKING_DIR, "stage2_features.pkl")
        )
    except FileNotFoundError:
        print("Models not found. Running training first...")
        s1_models, s2_models, s3_model, s2_features, _ = run_training(
            load_cached_data=load_cached_data
        )

    # Load Test Data
    test_df = generate_dataset(Config.TEST_META, "test_features", load_cached_data)

    # Stage 1 Inference
    X_test_long, _, groups_test = reshape_for_siamese(test_df, is_train=False)

    s1_preds = np.zeros(len(X_test_long))
    for model in s1_models:
        s1_preds += model.predict(X_test_long) / len(s1_models)

    # Reshape and Merge
    pred_df = pivot_predictions_to_wide(groups_test, s1_preds)
    test_aug = test_df.merge(pred_df, on="segment_id", how="left")

    # Stage 2 Inference
    test_aug = add_stage2_features(test_aug)
    X_test = test_aug[s2_features]

    test_meta = pd.DataFrame()
    for name, models in s2_models.items():
        pred = np.zeros(len(X_test))
        for m in models:
            pred += m.predict(X_test) / len(models)
        test_meta[name] = pred

    # Stage 3 Inference
    final_preds = s3_model.predict(test_meta)

    # Save
    submission = pd.DataFrame(
        {"segment_id": test_aug["segment_id"], "time_to_eruption": final_preds}
    )

    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
