import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from library.model_handler import VolcanoLGBM
from library.data_loader import generate_dataset
from library.utils import save_submission


def run_cv(n_splits=5, load_cached_data=True, debug_size=None, params=None):
    """
    Orchestrates Stratified K-Fold Cross-Validation.

    Loads both training and validation subsets defined in metadata, combines them,
    and performs stratified splitting based on the target variable.

    Args:
        n_splits (int): Number of folds for CV.
        load_cached_data (bool): Whether to attempt loading features from cache.
        debug_size (int, optional): Number of samples to load for debugging.
        params (dict, optional): Hyperparameters for the LightGBM model.

    Returns:
        tuple: (list of trained VolcanoLGBM models, float overall OOF MAE)
    """
    # 1. Load Data
    # We load both 'train' and 'val' sets from the data loader to form the full dataset for CV
    print("Loading training and validation data for CV...")
    X_train_part, y_train_part = generate_dataset("train", load_cached_data, debug_size)
    X_val_part, y_val_part = generate_dataset("val", load_cached_data, debug_size)

    # 2. Combine Datasets
    # Reset index to ensure proper alignment during splitting
    X = pd.concat([X_train_part, X_val_part], axis=0).reset_index(drop=True)
    y = pd.concat([y_train_part, y_val_part], axis=0).reset_index(drop=True)

    print(f"Combined dataset shape: {X.shape}")

    # 3. Binning for Stratification
    # StratifiedKFold requires discrete classes. We bin the continuous target.
    # We use qcut to create bins of roughly equal size.
    # duplicates='drop' handles cases where multiple samples have the exact same target value.
    num_bins = min(20, len(y) // n_splits)  # Ensure enough samples per bin
    if num_bins < 2:
        num_bins = 2

    y_binned = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

    # 4. Initialize CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    models = []
    oof_preds = np.zeros(len(y))
    fold_scores = []

    # 5. Training Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        # Split data
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

        # Instantiate and train model
        model = VolcanoLGBM(params)
        model.train(X_tr, y_tr, X_va, y_va)

        # Predict on validation fold
        preds = model.predict(X_va)
        oof_preds[val_idx] = preds

        # Calculate Fold Score
        score = mean_absolute_error(y_va, preds)
        fold_scores.append(score)
        models.append(model)

        print(f"Fold {fold + 1} MAE: {score}")

    # 6. Overall Evaluation
    overall_mae = mean_absolute_error(y, oof_preds)
    print(f"\nCross-Validation Completed.")
    print(f"Fold Scores: {fold_scores}")
    print(f"Overall OOF MAE: {overall_mae}")

    return models, overall_mae


def generate_final_submission(
    models,
    load_cached_data=True,
    debug_size=None,
    output_path="./submission/submission.csv",
):
    """
    Generates predictions for the test set using an ensemble of models and saves to CSV.

    Args:
        models (list): List of trained VolcanoLGBM models.
        load_cached_data (bool): Whether to use cached features.
        debug_size (int, optional): Limit data size for debugging.
        output_path (str): Path to save the submission file.
    """
    print("\nGenerating submission...")

    # Load Test Data
    # generate_dataset returns (df, None) for test mode. df includes segment_id.
    test_df, _ = generate_dataset("test", load_cached_data, debug_size)

    # Separate features and IDs
    feature_cols = [
        c for c in test_df.columns if c not in ["segment_id", "time_to_eruption"]
    ]
    X_test = test_df[feature_cols]
    segment_ids = test_df["segment_id"]

    print(f"Predicting on {len(X_test)} test samples with {len(models)} models...")

    # Ensemble Prediction (Average)
    avg_preds = np.zeros(len(X_test))
    for i, model in enumerate(models):
        preds = model.predict(X_test)
        avg_preds += preds

    avg_preds /= len(models)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"segment_id": segment_ids, "time_to_eruption": avg_preds}
    )

    # Save
    save_submission(submission_df, output_path)
