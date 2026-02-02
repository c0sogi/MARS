import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from library.config import METADATA_DIR, SUBMISSION_PATH, RANDOM_SEED, N_FOLDS, N_JOBS
from library.data_processor import generate_feature_matrix
from library.models import get_base_models, get_meta_model


def run_training_pipeline(debug_size=None, load_cached=True):
    """
    Orchestrates the Peak-Aware Stacked Kinematic Ensemble pipeline.

    This function performs the following steps:
    1. Loads and processes features for Train, Validation, and Test sets.
    2. Combines Train and Validation sets for robust Stacking.
    3. Level 0: Trains Base Learners (LGBM, XGB, CatBoost) using Stratified K-Fold CV
       to generate Out-of-Fold (OOF) predictions and determine optimal iteration counts.
    4. Level 1: Trains a Ridge Meta-Learner on the OOF predictions.
    5. Retrains Base Learners on the full dataset using averaged optimal iterations.
    6. Generates final predictions on the Test set using the stacked architecture.
    7. Saves the submission file.

    Args:
        debug_size (int, optional): Number of samples to use for debugging. If None, uses full dataset.
        load_cached (bool): Whether to attempt loading features from cache.
    """
    print("Starting Peak-Aware Stacked Kinematic Ensemble Pipeline...")

    # ==========================================
    # 1. Data Loading & Preparation
    # ==========================================

    # Load Metadata
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    train_meta = pd.read_csv(train_meta_path)
    val_meta = pd.read_csv(val_meta_path)
    test_meta = pd.read_csv(test_meta_path)

    # Handle Debug Slicing
    if debug_size is not None:
        print(f"Debug Mode: Slicing metadata to {debug_size} rows.")
        train_meta = train_meta.head(debug_size)
        val_meta = val_meta.head(debug_size)
        test_meta = test_meta.head(debug_size)

    # Generate Features
    print("Loading/Generating Features...")
    train_features = generate_feature_matrix(
        "train", load_cached=load_cached, debug_size=debug_size
    )
    val_features = generate_feature_matrix(
        "val", load_cached=load_cached, debug_size=debug_size
    )
    test_features = generate_feature_matrix(
        "test", load_cached=load_cached, debug_size=debug_size
    )

    # Merge Targets
    print("Merging features with targets...")
    train_df = train_features.merge(
        train_meta[["segment_id", "time_to_eruption"]], on="segment_id", how="inner"
    )
    val_df = val_features.merge(
        val_meta[["segment_id", "time_to_eruption"]], on="segment_id", how="inner"
    )

    # Combine Train and Val for Stacking
    # Filter only feature columns (exclude segment_id and target for X)
    feature_cols = [
        c for c in train_df.columns if c not in ["segment_id", "time_to_eruption"]
    ]

    X_full = pd.concat(
        [train_df[feature_cols], val_df[feature_cols]], axis=0
    ).reset_index(drop=True)
    y_full = pd.concat(
        [train_df["time_to_eruption"], val_df["time_to_eruption"]], axis=0
    ).reset_index(drop=True)

    X_test = test_features[feature_cols]

    print(f"Combined Training Data Shape: {X_full.shape}")
    print(f"Test Data Shape: {X_test.shape}")

    # ==========================================
    # 2. Level 0: Cross-Validation & OOF Generation
    # ==========================================
    print("\n--- Level 0 Training (Base Learners CV) ---")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # Get model names
    dummy_models = get_base_models()
    model_names = list(dummy_models.keys())

    oof_preds = pd.DataFrame(np.nan, index=X_full.index, columns=model_names)
    best_iterations = {name: [] for name in model_names}

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full, y_full)):
        X_tr, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]

        # Get fresh models for this fold
        fold_models = get_base_models(random_seed=RANDOM_SEED)

        fold_preds_dict = {}

        for name, model in fold_models.items():
            # Fit with Early Stopping
            if name == "cat":
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=50,
                    verbose=False,
                )
                # CatBoost get_best_iteration returns index (0-based)
                best_iter_idx = model.get_best_iteration()
                if best_iter_idx is None:
                    best_iter_idx = model.tree_count_ - 1
                best_iterations[name].append(best_iter_idx + 1)

            elif name == "lgbm":
                callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric="mae",
                    callbacks=callbacks,
                )
                # LGBM best_iteration_ is the count (1-based)
                best_iterations[name].append(model.best_iteration_)

            elif name == "xgb":
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                # XGB best_iteration is 0-based index
                best_iterations[name].append(model.best_iteration + 1)

            # Predict
            val_pred = model.predict(X_val)
            oof_preds.loc[val_idx, name] = val_pred
            fold_preds_dict[name] = val_pred

        # Fold Score (Average of base models for logging)
        avg_fold_pred = np.mean(list(fold_preds_dict.values()), axis=0)
        fold_mae = mean_absolute_error(y_val, avg_fold_pred)
        fold_scores.append(fold_mae)
        print(f"Fold {fold+1} Average MAE: {fold_mae}")

    avg_cv_mae = np.mean(fold_scores)
    print(f"Average CV MAE (Base Ensemble): {avg_cv_mae}")

    # ==========================================
    # 3. Level 1: Meta Learner Training
    # ==========================================
    print("\n--- Level 1 Training (Ridge Stacking) ---")

    meta_model = get_meta_model(random_seed=RANDOM_SEED)
    meta_model.fit(oof_preds, y_full)

    # Check OOF Score of Meta Model
    meta_oof_preds = meta_model.predict(oof_preds)
    meta_mae = mean_absolute_error(y_full, meta_oof_preds)
    print(f"Meta-Learner OOF MAE: {meta_mae}")

    # ==========================================
    # 4. Level 0: Refit on Full Data
    # ==========================================
    print("\n--- Retraining Base Learners on Full Data ---")

    final_base_models = get_base_models(random_seed=RANDOM_SEED)

    for name, model in final_base_models.items():
        # Calculate average optimal iterations
        avg_iter = int(np.mean(best_iterations[name]))
        print(f"Retraining {name} with {avg_iter} estimators.")

        if name == "lgbm":
            model.set_params(n_estimators=avg_iter)
            model.fit(X_full, y_full)
        elif name == "xgb":
            model.set_params(n_estimators=avg_iter)
            model.fit(X_full, y_full, verbose=False)
        elif name == "cat":
            model.set_params(iterations=avg_iter)
            model.fit(X_full, y_full, verbose=False)

    # ==========================================
    # 5. Inference
    # ==========================================
    print("\n--- Generating Predictions ---")

    # Base Predictions on Test Set
    test_base_preds = pd.DataFrame(index=X_test.index, columns=model_names)

    for name, model in final_base_models.items():
        test_base_preds[name] = model.predict(X_test)

    # Meta Prediction
    final_predictions = meta_model.predict(test_base_preds)

    # ==========================================
    # 6. Submission
    # ==========================================
    submission = pd.DataFrame(
        {
            "segment_id": test_features["segment_id"],
            "time_to_eruption": final_predictions,
        }
    )

    print(f"Saving submission to {SUBMISSION_PATH}...")
    submission.to_csv(SUBMISSION_PATH, index=False)
    print("Pipeline Completed Successfully.")
