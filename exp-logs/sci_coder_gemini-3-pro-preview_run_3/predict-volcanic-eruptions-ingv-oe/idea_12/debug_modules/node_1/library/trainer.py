import os
import copy
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone

from library.config import SEED, N_FOLDS, WORKING_DIR
from library.utils import seed_everything, save_submission
from library.feature_loader import build_feature_matrix
from library.model_zoo import get_base_models, get_meta_learner


def run_stacking_cv(debug: bool = False, load_cached_data: bool = True):
    """
    Executes the Two-Level Stacking Ensemble pipeline:
    1. Loads and prepares data.
    2. Performs Stratified K-Fold CV to train Base Learners and generate OOF preds.
    3. Trains Meta Learner on OOF preds.
    4. Retrains Base Learners on full data using average optimal iterations.
    5. Generates final predictions on Test set and saves submission.
    """
    seed_everything(SEED)

    # ==========================================
    # 1. Data Loading
    # ==========================================
    print("Loading datasets...")
    # Load train and val separately but combine them for CV to maximize data utility
    train_df = build_feature_matrix(
        "train", debug=debug, load_cached_data=load_cached_data
    )
    val_df = build_feature_matrix("val", debug=debug, load_cached_data=load_cached_data)
    test_df = build_feature_matrix(
        "test", debug=debug, load_cached_data=load_cached_data
    )

    # Combine for full training
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    # Identify Feature Columns (exclude metadata and target)
    feature_cols = [
        c for c in full_train_df.columns if c not in ["segment_id", "time_to_eruption"]
    ]

    X = full_train_df[feature_cols]
    y = full_train_df["time_to_eruption"]

    X_test = test_df[feature_cols]
    test_ids = test_df[["segment_id"]]

    print(f"Full Training Data Shape: {X.shape}")
    print(f"Test Data Shape: {X_test.shape}")

    # ==========================================
    # 2. Stratified K-Fold Setup
    # ==========================================
    # Create bins for stratification of continuous target
    num_bins = 10
    if len(y) < num_bins:
        num_bins = max(1, len(y) // 2)

    y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Initialize Base Models
    base_models_map = get_base_models()
    model_names = list(base_models_map.keys())

    # Storage for OOF predictions and Best Iterations
    oof_preds = pd.DataFrame(0.0, index=X.index, columns=model_names)
    best_iterations = {name: [] for name in model_names}

    # ==========================================
    # 3. Cross-Validation Loop (Level 0)
    # ==========================================
    print(f"\nStarting {N_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins)):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        for name, model_template in base_models_map.items():
            # Clone model to ensure fresh start for each fold
            try:
                model = clone(model_template)
            except Exception:
                model = copy.deepcopy(model_template)

            # Define fit parameters for Early Stopping
            fit_params = {}
            if name == "lgbm":
                fit_params = {
                    "eval_set": [(X_val_fold, y_val_fold)],
                    "eval_metric": "mae",
                }
            elif name == "xgb":
                fit_params = {"eval_set": [(X_val_fold, y_val_fold)], "verbose": False}
            elif name == "catboost":
                fit_params = {
                    "eval_set": (X_val_fold, y_val_fold),
                    "verbose": False,
                    "use_best_model": True,
                }

            # Train
            model.fit(X_train_fold, y_train_fold, **fit_params)

            # Predict OOF
            pred_val = model.predict(X_val_fold)
            oof_preds.loc[val_idx, name] = pred_val

            # Log Metric
            fold_mae = mean_absolute_error(y_val_fold, pred_val)
            print(f"[{name}] Fold MAE: {fold_mae}")

            # Record Best Iteration
            best_iter = None
            if name == "lgbm":
                best_iter = model.best_iteration_
            elif name == "xgb":
                # Handle different XGBoost versions
                try:
                    best_iter = model.get_booster().best_iteration
                except AttributeError:
                    best_iter = model.best_iteration
            elif name == "catboost":
                best_iter = model.get_best_iteration()

            if best_iter is not None:
                best_iterations[name].append(best_iter)

    # Report Global OOF Performance
    print("\n--- Base Model Global OOF Performance ---")
    for name in model_names:
        mae = mean_absolute_error(y, oof_preds[name])
        print(f"{name} Global OOF MAE: {mae}")

    # ==========================================
    # 4. Train Meta Learner (Level 1)
    # ==========================================
    print("\nTraining Meta Learner (Ridge) on OOF predictions...")
    meta_learner = get_meta_learner()
    meta_learner.fit(oof_preds, y)

    # Evaluate Meta Learner on OOF (Proxy for CV score)
    meta_oof_preds = meta_learner.predict(oof_preds)
    meta_mae = mean_absolute_error(y, meta_oof_preds)
    print(f"Meta Learner OOF MAE: {meta_mae}")

    # ==========================================
    # 5. Retrain Base Models on Full Data
    # ==========================================
    print(
        "\nRetraining Base Models on Full Dataset using average optimal iterations..."
    )
    final_base_models = {}

    for name, model_template in base_models_map.items():
        # Calculate average best iteration from CV
        if best_iterations[name]:
            avg_iter = int(np.mean(best_iterations[name]))
        else:
            avg_iter = 1000  # Fallback

        print(f"Retraining {name} with {avg_iter} estimators...")

        try:
            model = clone(model_template)
        except Exception:
            model = copy.deepcopy(model_template)

        # Update params: Set estimators and disable early stopping configuration
        # to prevent warnings/errors when no eval_set is provided.
        params_update = {}
        if name == "lgbm":
            params_update = {"n_estimators": avg_iter, "early_stopping_rounds": None}
        elif name == "xgb":
            params_update = {"n_estimators": avg_iter, "early_stopping_rounds": None}
        elif name == "catboost":
            params_update = {"iterations": avg_iter, "early_stopping_rounds": None}

        model.set_params(**params_update)

        # Fit on full data
        model.fit(X, y)
        final_base_models[name] = model

    # ==========================================
    # 6. Final Inference & Submission
    # ==========================================
    print("\nGenerating Final Predictions on Test Set...")

    # 1. Base Model Predictions
    test_base_preds = pd.DataFrame(index=X_test.index, columns=model_names)
    for name, model in final_base_models.items():
        test_base_preds[name] = model.predict(X_test)

    # 2. Meta Learner Prediction
    final_predictions = meta_learner.predict(test_base_preds)

    # 3. Save Submission
    save_submission(final_predictions, test_ids)

    # 4. Save Models (for persistence)
    models_path = os.path.join(WORKING_DIR, "stacked_models.pkl")
    joblib.dump(
        {"base_models": final_base_models, "meta_learner": meta_learner}, models_path
    )
    print(f"Trained models saved to {models_path}")
