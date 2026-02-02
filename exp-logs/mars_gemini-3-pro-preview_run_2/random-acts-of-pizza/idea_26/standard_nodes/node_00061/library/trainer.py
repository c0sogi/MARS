import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import set_seed, save_joblib, ensure_directory, save_npy


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    pipeline_creator: callable,
    embedding_dim: int,
    meta_dim: int,
    model_name_prefix: str,
    n_folds: int = Config.N_FOLDS,
    param_grid: dict = None,
    debug_sample_size: int = None,
):
    """
    Executes Stratified K-Fold Cross Validation using the provided pipeline creator.

    This function manages the training lifecycle:
    1. Subsets data for debugging if requested.
    2. Splits data using StratifiedKFold.
    3. Instantiates and fits the model pipeline (including GridSearchCV).
    4. Generates OOF (Out-Of-Fold) and Test predictions.
    5. Calculates and prints high-precision AUC metrics.
    6. Saves trained models and prediction arrays to disk.

    Args:
        X (np.ndarray): Training features matrix (concatenated embeddings + metadata).
        y (np.ndarray): Training labels vector.
        X_test (np.ndarray): Test features matrix.
        pipeline_creator (callable): Factory function from library.pipeline_factory to create the GridSearchCV object.
        embedding_dim (int): Dimension of the embedding portion of the feature matrix.
        meta_dim (int): Dimension of the metadata portion of the feature matrix.
        model_name_prefix (str): Identifier for the model branch (e.g., 'branch_a'), used for file naming.
        n_folds (int, optional): Number of cross-validation folds. Defaults to Config.N_FOLDS.
        param_grid (dict, optional): Hyperparameter grid to override defaults. Defaults to None.
        debug_sample_size (int, optional): If set, trains on a random subset of this size for rapid debugging.

    Returns:
        tuple: (models, oof_preds, avg_test_preds, scores)
            - models (list): List of fitted best estimators from each fold.
            - oof_preds (np.ndarray): Aggregated out-of-fold probability predictions.
            - avg_test_preds (np.ndarray): Test set probability predictions averaged across folds.
            - scores (list): List of ROC AUC scores for each fold.
    """
    set_seed(Config.SEED)

    # ---------------------------------------------------------
    # Debugging: Subset data if requested
    # ---------------------------------------------------------
    if debug_sample_size is not None and debug_sample_size < len(X):
        print(f"DEBUG: Subsetting training data to {debug_sample_size} samples.")
        # Use random choice for subsetting, ensuring reproducibility with set_seed
        indices = np.random.choice(len(X), debug_sample_size, replace=False)
        X = X[indices]
        y = y[indices]
        # Note: While this might slightly perturb stratification, it is sufficient for debugging pipelines.

    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------
    oof_preds = np.zeros(len(y), dtype=np.float64)
    test_preds_accum = np.zeros(len(X_test), dtype=np.float64)
    models = []
    scores = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)

    print(f"\nStarting {n_folds}-Fold Cross-Validation for {model_name_prefix}...")

    # ---------------------------------------------------------
    # Cross-Validation Loop
    # ---------------------------------------------------------
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Split Data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create Pipeline
        # The pipeline_creator returns a GridSearchCV object configured with the specific architecture
        gs_pipeline = pipeline_creator(embedding_dim, meta_dim, param_grid)

        # Fit Pipeline (Executes Grid Search internally)
        print(f"Fold {fold}: Fitting pipeline...")
        gs_pipeline.fit(X_train, y_train)

        # Retrieve Best Estimator
        best_model = gs_pipeline.best_estimator_
        models.append(best_model)

        # Inference
        # predict_proba returns [prob_class_0, prob_class_1], we take index 1
        val_probs = best_model.predict_proba(X_val)[:, 1]
        test_probs = best_model.predict_proba(X_test)[:, 1]

        # Store Predictions
        oof_preds[val_idx] = val_probs
        test_preds_accum += test_probs

        # Evaluation
        fold_auc = roc_auc_score(y_val, val_probs)
        scores.append(fold_auc)

        # Reporting (Full Precision)
        print(f"Fold {fold} AUC: {fold_auc:.16f}")
        print(f"Fold {fold} Best Params: {gs_pipeline.best_params_}")

        # Save Model Artifact
        model_dir = os.path.join(Config.CACHE_DIR, "models")
        ensure_directory(model_dir)
        model_path = os.path.join(model_dir, f"{model_name_prefix}_fold_{fold}.joblib")
        save_joblib(best_model, model_path)

    # ---------------------------------------------------------
    # Aggregation and Final Metrics
    # ---------------------------------------------------------
    avg_test_preds = test_preds_accum / n_folds
    overall_auc = roc_auc_score(y, oof_preds)

    print(f"\n{model_name_prefix} - Overall OOF AUC: {overall_auc:.16f}")
    print(f"{model_name_prefix} - Average Fold AUC: {np.mean(scores):.16f}")

    # Save Predictions for Ensemble/Analysis
    preds_dir = os.path.join(Config.CACHE_DIR, "predictions")
    ensure_directory(preds_dir)

    oof_path = os.path.join(preds_dir, f"oof_preds_{model_name_prefix}.npy")
    test_path = os.path.join(preds_dir, f"test_preds_{model_name_prefix}.npy")

    save_npy(oof_preds, oof_path)
    save_npy(avg_test_preds, test_path)

    return models, oof_preds, avg_test_preds, scores
