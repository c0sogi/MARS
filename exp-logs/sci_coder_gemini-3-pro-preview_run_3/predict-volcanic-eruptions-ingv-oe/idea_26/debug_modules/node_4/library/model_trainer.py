import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import joblib
from library import config


def train_kfold_ensemble(X, y, n_folds=None, save_models=True):
    """
    Trains a homogeneous ensemble of LightGBM models using Stratified K-Fold CV.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector (time_to_eruption).
        n_folds (int): Number of cross-validation folds.
        save_models (bool): Whether to save trained models to disk.

    Returns:
        list: A list of trained LightGBM Booster objects.
    """
    # Ensure reproducibility
    np.random.seed(config.SEED)

    # Resolve n_folds dynamically to pick up runtime config overrides (Cite debug_lesson_2)
    if n_folds is None:
        n_folds = config.N_FOLDS

    # Prepare for Stratified Split on Continuous Target
    # Bin the target into quantiles to simulate classes for StratifiedKFold
    # Dynamically adjust bins to ensure we have enough samples per bin for the splits
    max_possible_bins = max(1, int(len(y) // n_folds))
    num_bins = min(15, max_possible_bins)

    y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)

    models = []
    oof_preds = np.zeros(len(y))

    print(f"Starting {n_folds}-Fold Stratified Cross-Validation (Bins: {num_bins})...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        # Split Data
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Create LightGBM Datasets
        # We don't set categorical features explicitly as they are not present in this sensor data
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # Train Model
        # Note: 'early_stopping_rounds' is passed via callbacks in newer LightGBM versions,
        # but passing it as a parameter is often supported or handled via specific callback construction.
        # We use the standard python API parameter approach which is widely compatible.
        callbacks = [
            lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=config.VERBOSE_EVAL),
        ]

        model = lgb.train(
            config.LGBM_PARAMS,
            dtrain,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Save Model
        if save_models:
            model_path = os.path.join(config.WORKING_DIR, f"lgbm_fold_{fold}.txt")
            model.save_model(model_path)

        models.append(model)

        # Evaluate on Validation Fold
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_preds

        fold_mae = np.mean(np.abs(y_val - val_preds))
        print(f"Fold {fold + 1} MAE: {fold_mae}")

    # Overall Metric
    overall_mae = np.mean(np.abs(y - oof_preds))
    print(f"\nOverall OOF MAE: {overall_mae}")

    return models


def predict_with_ensemble(models, X):
    """
    Generates predictions by averaging outputs from all ensemble models.

    Args:
        models (list): List of trained LightGBM Booster objects.
        X (pd.DataFrame): Feature matrix for inference.

    Returns:
        np.array: Averaged predictions.
    """
    if not models:
        raise ValueError("No models provided for prediction.")

    print(f"Predicting with ensemble of {len(models)} models...")

    # Initialize accumulator
    final_preds = np.zeros(len(X))

    for i, model in enumerate(models):
        preds = model.predict(X, num_iteration=model.best_iteration)
        final_preds += preds

    # Average
    final_preds /= len(models)

    return final_preds


def save_submission(segment_ids, predictions, output_path):
    """
    Formats and saves the submission file.

    Args:
        segment_ids (pd.Series or list): The segment IDs.
        predictions (np.array): The predicted time_to_eruption values.
        output_path (str): Path to save the CSV.
    """
    submission_df = pd.DataFrame(
        {"segment_id": segment_ids, "time_to_eruption": predictions}
    )

    # Ensure segment_id is int64
    submission_df["segment_id"] = submission_df["segment_id"].astype(np.int64)

    print(f"Saving submission to {output_path}...")
    submission_df.to_csv(output_path, index=False)
    print("Submission saved successfully.")
