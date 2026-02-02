import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from library.config import Config
from library.data_processing import DataManager
from library.utils import seed_everything


def run_tabular_training(debug=False):
    """
    Executes the training pipeline for the Tabular Branch (Branch A).

    1. Loads tabular features for Train, Val, and Test sets using DataManager.
    2. Combines Train and Val for 5-Fold Cross-Validation.
    3. Trains LightGBM models with Early Stopping.
    4. Generates OOF predictions and Test predictions.

    Args:
        debug (bool): If True, runs on a subset of data.

    Returns:
        tuple: (df_oof, df_test_preds)
            - df_oof: DataFrame containing [segment_id, pred_time_to_eruption, true_time_to_eruption]
            - df_test_preds: DataFrame containing [segment_id, time_to_eruption]
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    print("Initializing DataManager for Tabular Training...")
    dm = DataManager()

    # Load Data
    # We only need the tabular features (first return value) and targets (third return value)
    # The second return value (spectrograms) is ignored for this branch
    print("Loading Train data...")
    X_train_part, _, y_train_part = dm.get_data(
        "train", load_cached_data=True, debug=debug
    )

    print("Loading Val data...")
    X_val_part, _, y_val_part = dm.get_data("val", load_cached_data=True, debug=debug)

    print("Loading Test data...")
    X_test, _, _ = dm.get_data("test", load_cached_data=True, debug=debug)

    # Combine Train and Val for full Cross-Validation
    # Reset index to ensure KFold indexing works correctly
    X_full = pd.concat([X_train_part, X_val_part], axis=0).reset_index(drop=True)
    y_full = np.concatenate([y_train_part, y_val_part], axis=0)

    print(f"Combined Training Set Shape: {X_full.shape}")
    print(f"Test Set Shape: {X_test.shape}")

    # Identify feature columns (exclude metadata like segment_id)
    feature_cols = [col for col in X_full.columns if col != "segment_id"]
    print(f"Number of features: {len(feature_cols)}")

    # Prepare containers for predictions
    oof_preds = np.zeros(len(X_full))
    test_preds_accum = np.zeros(len(X_test))

    # Initialize KFold
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    # Training Loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full, y_full)):
        print(f"\n--- Starting Fold {fold + 1}/{Config.N_FOLDS} ---")

        # Split data
        X_tr, y_tr = X_full.iloc[train_idx][feature_cols], y_full[train_idx]
        X_va, y_va = X_full.iloc[val_idx][feature_cols], y_full[val_idx]

        # Create LightGBM Datasets
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

        # Define Callbacks
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=Config.LGBM_EARLY_STOPPING_ROUNDS, verbose=True
            ),
            lgb.log_evaluation(period=500),
        ]

        # Train Model
        # Note: n_estimators is passed as num_boost_round
        model = lgb.train(
            params=Config.LGBM_PARAMS,
            train_set=dtrain,
            num_boost_round=Config.LGBM_PARAMS["n_estimators"],
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Generate Predictions
        # num_iteration=model.best_iteration uses the best iteration found by early stopping
        val_pred = model.predict(X_va, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred

        test_pred = model.predict(
            X_test[feature_cols], num_iteration=model.best_iteration
        )
        test_preds_accum += test_pred

        # Calculate and Print Fold Metric
        fold_mae = mean_absolute_error(y_va, val_pred)
        print(f"Fold {fold + 1} MAE: {fold_mae}")

        # Save Model (optional, but good for persistence)
        model_save_path = os.path.join(Config.WORKING_DIR, f"lgb_model_fold_{fold}.txt")
        model.save_model(model_save_path)

        # Cleanup to save memory
        del X_tr, y_tr, X_va, y_va, dtrain, dval, model
        gc.collect()

    # Aggregate Test Predictions
    avg_test_preds = test_preds_accum / Config.N_FOLDS

    # Calculate Overall OOF Score
    total_mae = mean_absolute_error(y_full, oof_preds)
    print(f"\nOverall Tabular OOF MAE: {total_mae}")

    # Construct Output DataFrames
    df_oof = X_full[["segment_id"]].copy()
    df_oof["pred_time_to_eruption"] = oof_preds
    df_oof["true_time_to_eruption"] = y_full

    df_test = X_test[["segment_id"]].copy()
    df_test["time_to_eruption"] = avg_test_preds

    return df_oof, df_test
