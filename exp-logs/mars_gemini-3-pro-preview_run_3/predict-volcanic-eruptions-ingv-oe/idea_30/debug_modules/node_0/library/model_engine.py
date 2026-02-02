import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import compute_metric


def train_fold_model(X_train, y_train, X_val, y_val, fold_id):
    """
    Trains a single LightGBM model for a specific fold.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training targets.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation targets.
        fold_id (int): ID of the current fold (for logging).

    Returns:
        lgb.Booster: The trained LightGBM model.
    """
    # Create LightGBM Datasets
    # Reference to train_set in val_set allows for proper memory usage and alignment
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    # Setup Callbacks
    # Early stopping based on the metric defined in Config (MAE)
    callbacks = [
        lgb.early_stopping(stopping_rounds=Config.LGBM_PARAMS["early_stopping_round"]),
        lgb.log_evaluation(period=100),
    ]

    print(f"--- Training Fold {fold_id} ---")

    # Train
    model = lgb.train(
        params=Config.LGBM_PARAMS,
        train_set=train_set,
        valid_sets=[train_set, val_set],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    return model


def run_cross_validation(X, y, save_models=True):
    """
    Executes Stratified K-Fold Cross-Validation to train an ensemble of models.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Targets.
        save_models (bool): Whether to save model artifacts to disk.

    Returns:
        list: A list of trained lgb.Booster objects.
        float: The overall OOF MAE.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Generate bins for Stratified K-Fold
    # We bin the continuous target to ensure representative distribution in each fold
    num_bins = 10
    y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    models = []
    oof_preds = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins)):
        fold_id = fold + 1

        # Split Data
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Train Model
        model = train_fold_model(X_train, y_train, X_val, y_val, fold_id)

        # Save Model
        if save_models:
            model_path = os.path.join(Config.WORKING_DIR, f"lgbm_model_fold_{fold}.txt")
            model.save_model(model_path)
            # print(f"Model for fold {fold_id} saved to {model_path}")

        models.append(model)

        # Generate OOF Predictions
        # best_iteration is automatically used by predict if model was trained with early stopping
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = preds

        # Fold Metric
        fold_mae = compute_metric(y_val, preds)
        print(f"Fold {fold_id} MAE: {fold_mae}")

    # Overall Metric
    total_mae = compute_metric(y, oof_preds)
    print(f"\nOverall CV MAE: {total_mae}")

    return models, total_mae


def load_ensemble_models():
    """
    Loads trained models from the working directory.

    Returns:
        list: List of lgb.Booster objects.
    """
    models = []
    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"lgbm_model_fold_{fold}.txt")
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            model = lgb.Booster(model_file=model_path)
            models.append(model)
        else:
            print(f"Warning: Model file {model_path} not found.")

    return models


def predict_ensemble(models, X_test):
    """
    Generates predictions using the ensemble of models.

    Args:
        models (list): List of trained lgb.Booster objects.
        X_test (pd.DataFrame): Test features.

    Returns:
        np.ndarray: Averaged predictions.
    """
    if not models:
        raise ValueError("No models provided for prediction.")

    print(f"Generating predictions using {len(models)} models...")

    # Initialize predictions
    final_preds = np.zeros(len(X_test))

    for model in models:
        # Predict
        preds = model.predict(X_test, num_iteration=model.best_iteration)
        final_preds += preds

    # Average
    final_preds /= len(models)

    return final_preds
