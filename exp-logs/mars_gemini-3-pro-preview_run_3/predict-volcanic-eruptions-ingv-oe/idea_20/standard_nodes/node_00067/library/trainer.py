import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error

from library.config import Config
from library.data_loader import create_dataset
from library.model import VolcanoLGBM


def run_stratified_kfold(X, y, n_folds, seed, verbose=True):
    """
    Executes Stratified K-Fold Cross Validation.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        n_folds (int): Number of folds.
        seed (int): Random seed.
        verbose (bool): Whether to print progress.

    Returns:
        tuple: (oof_predictions, list_of_models, list_of_scores)
    """
    # Create stratification bins for continuous target
    # We use 15 bins to ensure fine-grained distribution matching
    # Handle edge case where dataset is smaller than bins (debug mode)
    num_bins = min(15, len(y) // n_folds) if len(y) > n_folds else 1

    if num_bins > 1:
        try:
            y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")
        except ValueError:
            # Fallback if qcut fails (e.g. too many duplicates)
            y_bins = y
    else:
        y_bins = np.zeros(len(y))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof_preds = np.zeros(len(y))
    models = []
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins)):
        if verbose:
            print(f"\n--- Fold {fold + 1} / {n_folds} ---")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Instantiate model wrapper
        model = VolcanoLGBM()

        # Update model path to save unique artifact for this fold
        # e.g. ./working/idea_20/lgbm_model.txt -> ./working/idea_20/lgbm_model_fold_0.txt
        base_path, ext = os.path.splitext(Config.MODEL_OUTPUT_PATH)
        model.model_path = f"{base_path}_fold_{fold}{ext}"

        # Train the model
        model.train(X_train, y_train, X_val, y_val)

        # Generate predictions for validation set
        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds

        # Calculate metric
        fold_mae = mean_absolute_error(y_val, val_preds)
        scores.append(fold_mae)
        models.append(model)

        if verbose:
            print(f"Fold {fold + 1} MAE: {fold_mae}")

        # Explicit garbage collection to manage memory with dense features
        del X_train, y_train, X_val, y_val
        gc.collect()

    return oof_preds, models, scores


def train_and_predict(load_cached_data=True, debug_limit=None):
    """
    Main orchestration function for the training and prediction pipeline.

    Args:
        load_cached_data (bool): Whether to load features from Parquet cache.
        debug_limit (int, optional): Limit dataset size for debugging.

    Returns:
        float: Overall Cross-Validation MAE.
    """
    # 1. Load and Combine Data
    print("Loading Training and Validation Data...")
    # We load both splits and combine them to perform our own Stratified K-Fold
    X_train_part, y_train_part = create_dataset(
        "train", load_cached_data=load_cached_data, debug_limit=debug_limit
    )
    X_val_part, y_val_part = create_dataset(
        "val", load_cached_data=load_cached_data, debug_limit=debug_limit
    )

    X = np.concatenate([X_train_part, X_val_part], axis=0)
    y = np.concatenate([y_train_part, y_val_part], axis=0)

    print(f"Combined Dataset Shape: X={X.shape}, y={y.shape}")

    # 2. Run Cross-Validation
    n_folds = Config.TRAIN_PARAMS.get("n_folds", 5)
    seed = Config.SEED

    print(f"Starting {n_folds}-Fold Stratified Cross-Validation...")
    oof_preds, models, scores = run_stratified_kfold(X, y, n_folds, seed)

    # 3. Report Results
    overall_mae = mean_absolute_error(y, oof_preds)
    print("\n=== Cross-Validation Results ===")
    print(f"Fold MAEs: {scores}")
    print(f"Overall MAE: {overall_mae}")

    # 4. Generate Submission
    print("\nLoading Test Data...")
    X_test, segment_ids = create_dataset(
        "test", load_cached_data=load_cached_data, debug_limit=debug_limit
    )

    print("Generating Test Predictions (Ensemble Average)...")
    test_preds = np.zeros(len(X_test))

    # Average predictions from all fold models
    for model in models:
        test_preds += model.predict(X_test)

    test_preds /= len(models)

    # 5. Save Submission
    submission_df = pd.DataFrame(
        {"segment_id": segment_ids, "time_to_eruption": test_preds}
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    return overall_mae
