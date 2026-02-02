import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import setup_logger, set_seed
from library.data_loader import load_dataset
from library.feature_engineering import prepare_feature_matrices
from library.model_components import build_pipeline

# Initialize logger
logger = setup_logger("execution")


def train_model(X, y):
    """
    Performs Stratified K-Fold Cross Validation with Grid Search for hyperparameter tuning.

    The strategy involves:
    1. Splitting the full training data into 5 stratified folds.
    2. For each fold, iterating through the hyperparameter grid defined in Config.
    3. Training the Differentially-Regularized Bagged Linear Ensemble pipeline.
    4. Selecting the best hyperparameters based on validation AUC for that fold.
    5. Saving the best model and recording OOF predictions.

    Args:
        X (np.ndarray): Feature matrix (Text Embeddings + Tabular Features).
        y (np.ndarray): Target labels.

    Returns:
        list: A list of the best trained pipeline objects from each fold.
    """
    set_seed()

    # Initialize Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_SEED
    )

    fold_models = []
    oof_preds = np.zeros(len(y))

    # Create Parameter Grid
    param_grid = list(ParameterGrid(Config.GRID_SEARCH_PARAMS))
    logger.info(f"Hyperparameter tuning with {len(param_grid)} combinations per fold.")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"--- Starting Fold {fold + 1}/{Config.N_FOLDS} ---")

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        best_fold_auc = -1.0
        best_fold_model = None
        best_fold_params = None

        # Grid Search within the fold
        for params in param_grid:
            # Build pipeline with current parameters
            # The pipeline includes DifferentialScaler and BaggingClassifier(LogisticRegression)
            pipeline = build_pipeline(
                C=params["C"],
                alpha=params["alpha"],
                class_weight=params["class_weight"],
            )

            # Train
            pipeline.fit(X_train_fold, y_train_fold)

            # Validate
            # Predict probabilities for the positive class (1)
            y_pred_val = pipeline.predict_proba(X_val_fold)[:, 1]
            auc = roc_auc_score(y_val_fold, y_pred_val)

            # Track best model for this fold
            if auc > best_fold_auc:
                best_fold_auc = auc
                best_fold_model = pipeline
                best_fold_params = params

        # Log results for the fold (Full precision)
        logger.info(f"Fold {fold + 1} Best AUC: {best_fold_auc}")
        logger.info(f"Fold {fold + 1} Best Params: {best_fold_params}")

        # Store best model
        fold_models.append(best_fold_model)

        # Generate OOF predictions with best model for global metric calculation
        oof_preds[val_idx] = best_fold_model.predict_proba(X_val_fold)[:, 1]

        # Save model checkpoint
        Config.ensure_directories()
        model_path = os.path.join(Config.WORKING_DIR, f"fold_{fold}_pipeline.joblib")
        joblib.dump(best_fold_model, model_path)

    # Calculate and log Overall Metric
    total_auc = roc_auc_score(y, oof_preds)
    logger.info(f"Overall CV AUC: {total_auc}")

    return fold_models


def generate_submission(models, X_test, df_test):
    """
    Generates predictions using the ensemble of fold models (CV-Bagging) and saves the submission file.

    Args:
        models (list): List of trained pipeline objects.
        X_test (np.ndarray): Test feature matrix.
        df_test (pd.DataFrame): Test metadata DataFrame containing request_ids.
    """
    logger.info("Generating predictions for test set...")

    # Initialize array to store predictions from each model
    # Shape: (n_samples, n_models)
    preds_matrix = np.zeros((X_test.shape[0], len(models)))

    for i, model in enumerate(models):
        # Predict probabilities
        preds_matrix[:, i] = model.predict_proba(X_test)[:, 1]

    # Average predictions across all fold models (CV-Bagging)
    avg_preds = preds_matrix.mean(axis=1)

    # Create submission DataFrame
    submission = pd.DataFrame(
        {Config.ID_COL: df_test[Config.ID_COL], Config.TARGET_COL: avg_preds}
    )

    # Save submission
    Config.ensure_directories()
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_pipeline():
    """
    Main execution orchestrator.
    1. Loads data and metadata.
    2. Prepares feature matrices (Text Embeddings + Tabular).
    3. Merges Train and Validation sets for 5-Fold CV.
    4. Trains models using Stratified CV and Grid Search.
    5. Generates and saves the submission file.
    """
    set_seed()

    # 1. Load Data
    # df_train_meta and df_val_meta are used to align features, df_test_meta for submission IDs
    df_train_meta, df_val_meta, df_test_meta = load_dataset(load_cached_data=True)

    # 2. Feature Engineering
    # Returns numpy arrays for features and targets
    X_train_part, y_train_part, X_val_part, y_val_part, X_test = (
        prepare_feature_matrices(
            df_train_meta, df_val_meta, df_test_meta, load_cached_data=True
        )
    )

    # 3. Merge Train and Validation sets for full Cross-Validation
    # We combine the provided train and val splits to perform our own 5-Fold Stratified CV
    logger.info("Merging Train and Validation sets for 5-Fold Stratified CV...")
    X_full = np.vstack([X_train_part, X_val_part])
    y_full = np.concatenate([y_train_part, y_val_part])

    logger.info(f"Full Training Set Shape: {X_full.shape}")

    # 4. Train
    fold_models = train_model(X_full, y_full)

    # 5. Submit
    generate_submission(fold_models, X_test, df_test_meta)
