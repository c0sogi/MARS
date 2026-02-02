import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import KFold

from library.config import (
    WORK_DIR,
    LGBM_PARAMS,
    SEED,
    N_FOLDS,
    DEBUG,
    MAX_DEBUG_SAMPLES,
)
from library.utils import seed_everything, calc_mae
from library.data_factory import get_tabular_dataset


def run_lgbm_cv(load_cached_data: bool = True):
    """
    Executes the 5-Fold Cross-Validation training for the LightGBM Tabular Branch.

    Args:
        load_cached_data (bool): Whether to load features from parquet cache.

    Returns:
        tuple: (oof_df, test_pred_df)
            oof_df: DataFrame containing Out-of-Fold predictions and targets.
            test_pred_df: DataFrame containing averaged Test set predictions.
    """
    seed_everything(SEED)

    print("Initializing LightGBM Tabular Branch Training...")

    # ---------------------------------------------------------
    # 1. Load and Prepare Data
    # ---------------------------------------------------------
    # Load separate splits
    df_train_part = get_tabular_dataset("train", load_cached_data=load_cached_data)
    df_val_part = get_tabular_dataset("val", load_cached_data=load_cached_data)
    df_test = get_tabular_dataset("test", load_cached_data=load_cached_data)

    # Combine train and val for proper K-Fold CV
    df_train_full = pd.concat([df_train_part, df_val_part], axis=0).reset_index(
        drop=True
    )

    # Debugging: Subsample if requested
    if DEBUG:
        print(f"DEBUG Mode: Subsampling data to {MAX_DEBUG_SAMPLES} samples.")
        df_train_full = df_train_full.sample(
            n=min(len(df_train_full), MAX_DEBUG_SAMPLES), random_state=SEED
        ).reset_index(drop=True)
        df_test = df_test.sample(
            n=min(len(df_test), MAX_DEBUG_SAMPLES), random_state=SEED
        ).reset_index(drop=True)

    # Identify Feature Columns (exclude metadata and target)
    exclude_cols = ["segment_id", "time_to_eruption", "file_path"]
    feature_cols = [c for c in df_train_full.columns if c not in exclude_cols]

    print(f"Training Data Shape: {df_train_full.shape}")
    print(f"Test Data Shape: {df_test.shape}")
    print(f"Number of Features: {len(feature_cols)}")

    # ---------------------------------------------------------
    # 2. Configure LightGBM
    # ---------------------------------------------------------
    # Prepare parameters
    params = LGBM_PARAMS.copy()

    # Extract early_stopping_rounds to use as callback
    early_stopping_rounds = params.pop("early_stopping_rounds", 100)

    # ---------------------------------------------------------
    # 3. Cross-Validation Loop
    # ---------------------------------------------------------
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Storage for OOF and Test predictions
    oof_preds = np.zeros(len(df_train_full))
    test_preds = np.zeros((len(df_test), N_FOLDS))

    # Store segment IDs for mapping
    train_segment_ids = df_train_full["segment_id"].values
    train_targets = df_train_full["time_to_eruption"].values
    test_segment_ids = df_test["segment_id"].values

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train_full)):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

        # Split Data
        X_train, y_train = (
            df_train_full.iloc[train_idx][feature_cols],
            train_targets[train_idx],
        )
        X_val, y_val = df_train_full.iloc[val_idx][feature_cols], train_targets[val_idx]

        # Create LGBM Datasets
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # Train
        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds, verbose=False
                ),
                lgb.log_evaluation(period=1000),
            ],
        )

        # Save Model
        model_path = os.path.join(WORK_DIR, f"lgb_model_fold_{fold}.txt")
        model.save_model(model_path)

        # Predict OOF
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred

        # Score
        fold_mae = calc_mae(y_val, val_pred)
        fold_scores.append(fold_mae)
        print(f"Fold {fold + 1} MAE: {fold_mae}")

        # Predict Test
        test_pred_fold = model.predict(
            df_test[feature_cols], num_iteration=model.best_iteration
        )
        test_preds[:, fold] = test_pred_fold

    # ---------------------------------------------------------
    # 4. Aggregation and Saving
    # ---------------------------------------------------------
    # Calculate Overall Metrics
    overall_mae = calc_mae(train_targets, oof_preds)
    avg_mae = np.mean(fold_scores)

    print("\n--- Training Complete ---")
    print(f"Average Fold MAE: {avg_mae}")
    print(f"Overall OOF MAE: {overall_mae}")

    # Prepare OOF DataFrame
    oof_df = pd.DataFrame(
        {
            "segment_id": train_segment_ids,
            "time_to_eruption": train_targets,
            "lgb_pred": oof_preds,
        }
    )

    # Prepare Test Prediction DataFrame (Average over folds)
    avg_test_preds = np.mean(test_preds, axis=1)
    test_df = pd.DataFrame({"segment_id": test_segment_ids, "lgb_pred": avg_test_preds})

    # Save to Working Directory
    oof_save_path = os.path.join(WORK_DIR, "lgbm_oof.csv")
    test_save_path = os.path.join(WORK_DIR, "lgbm_test.csv")

    oof_df.to_csv(oof_save_path, index=False)
    test_df.to_csv(test_save_path, index=False)

    print(f"Saved OOF predictions to {oof_save_path}")
    print(f"Saved Test predictions to {test_save_path}")

    return oof_df, test_df
