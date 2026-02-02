import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

from library.config import Config
from library.utils import seed_everything, mae_score
from library.feature_engineering import TabularFeatureEngineer


def run_lgbm_cv(train_df, val_df, test_df, load_cached_data=True):
    """
    Executes the LightGBM Cross-Validation pipeline (Branch A).

    1. Generates/Loads tabular features using TabularFeatureEngineer.
    2. Combines Train and Val sets for full K-Fold CV to maximize data usage.
    3. Trains LightGBM with Log-Target scaling (log1p) and MAE objective.
    4. Saves models and returns OOF and Test predictions.

    Args:
        train_df (pd.DataFrame): Training metadata.
        val_df (pd.DataFrame): Validation metadata.
        test_df (pd.DataFrame): Test metadata.
        load_cached_data (bool): Whether to use cached features.

    Returns:
        tuple: (oof_df, test_preds_df)
            oof_df: DataFrame with ['segment_id', 'time_to_eruption'] (predictions)
            test_preds_df: DataFrame with ['segment_id', 'time_to_eruption']
    """
    seed_everything(Config.SEED)

    # ---------------------------------------------------------
    # 1. Feature Engineering
    # ---------------------------------------------------------
    engineer = TabularFeatureEngineer()

    # Generate or Load Features
    # The engineer handles caching internally based on the subset name
    train_feats = engineer.create_tabular_dataset(
        train_df, "train", load_cached_data=load_cached_data
    )
    val_feats = engineer.create_tabular_dataset(
        val_df, "val", load_cached_data=load_cached_data
    )
    test_feats = engineer.create_tabular_dataset(
        test_df, "test", load_cached_data=load_cached_data
    )

    # ---------------------------------------------------------
    # 2. Prepare Datasets
    # ---------------------------------------------------------
    # Merge features with targets from metadata
    # train_df has ['segment_id', 'time_to_eruption', 'file_path']
    train_merged = train_df.merge(train_feats, on="segment_id", how="left")
    val_merged = val_df.merge(val_feats, on="segment_id", how="left")

    # Combine train and val for Cross-Validation
    full_df = pd.concat([train_merged, val_merged], axis=0).reset_index(drop=True)

    # Define columns
    target_col = "time_to_eruption"
    drop_cols = ["segment_id", "file_path", target_col]

    # Identify feature columns (all columns in features df except segment_id)
    # We use the columns from test_feats to ensure consistency
    feature_cols = [c for c in test_feats.columns if c != "segment_id"]

    # Prepare Training Data
    X = full_df[feature_cols]
    y = full_df[target_col].values
    segments = full_df["segment_id"].values

    # Prepare Test Data
    X_test = test_feats[feature_cols]
    test_segments = test_feats["segment_id"].values

    # ---------------------------------------------------------
    # 3. Cross-Validation Loop
    # ---------------------------------------------------------
    kf = KFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED)

    # Arrays to store results
    oof_preds = np.zeros(len(full_df))
    test_preds_accum = np.zeros((len(test_df), Config.NUM_FOLDS))

    print(f"Starting LightGBM {Config.NUM_FOLDS}-Fold CV on {len(full_df)} samples...")
    print(f"Number of features: {len(feature_cols)}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        # Split Data
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        # Log-Scale Targets (log1p) to handle large dynamic range
        y_train_log = np.log1p(y_train)
        y_val_log = np.log1p(y_val)

        # Create LightGBM Datasets
        lgb_train = lgb.Dataset(X_train, y_train_log, feature_name=feature_cols)
        lgb_val = lgb.Dataset(
            X_val, y_val_log, reference=lgb_train, feature_name=feature_cols
        )

        # Prepare Parameters
        params = Config.LGBM_PARAMS.copy()
        # Extract n_estimators to use as num_boost_round
        num_rounds = params.pop("n_estimators", 5000)

        # Callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),  # Silent training
        ]

        # Train Model
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=num_rounds,
            valid_sets=[lgb_train, lgb_val],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Predict Validation (Log Scale -> Original Scale)
        val_pred_log = model.predict(X_val, num_iteration=model.best_iteration)
        val_pred_raw = np.expm1(val_pred_log)
        val_pred_raw = np.maximum(val_pred_raw, 0)  # Clip negative predictions

        # Store OOF
        oof_preds[val_idx] = val_pred_raw

        # Calculate Fold Score
        fold_mae = mae_score(y_val, val_pred_raw)
        print(f"Fold {fold} MAE: {fold_mae}")

        # Predict Test
        test_pred_log = model.predict(X_test, num_iteration=model.best_iteration)
        test_pred_raw = np.expm1(test_pred_log)
        test_pred_raw = np.maximum(test_pred_raw, 0)
        test_preds_accum[:, fold] = test_pred_raw

        # Save Model
        model_path = os.path.join(Config.CACHE_DIR, f"lgb_model_fold_{fold}.txt")
        model.save_model(model_path)

    # ---------------------------------------------------------
    # 4. Results & Submission Construction
    # ---------------------------------------------------------
    # Overall OOF Score
    total_mae = mae_score(y, oof_preds)
    print(f"Overall LightGBM CV MAE: {total_mae}")

    # Create OOF DataFrame
    oof_df = pd.DataFrame({"segment_id": segments, "time_to_eruption": oof_preds})

    # Create Test DataFrame (Average across folds)
    avg_test_preds = np.mean(test_preds_accum, axis=1)
    test_preds_df = pd.DataFrame(
        {"segment_id": test_segments, "time_to_eruption": avg_test_preds}
    )

    return oof_df, test_preds_df
