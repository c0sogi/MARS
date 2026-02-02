import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
import joblib
import os

import library.config as config
from library.feature_pipeline import ADBEFTransformer


def optimize_logistic_regression(X_train, y_train):
    """
    Performs Grid Search to find the best hyperparameters for Logistic Regression.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.

    Returns:
        estimator: The best fitted LogisticRegression estimator.
        dict: The best hyperparameters found.
    """
    base_lr = LogisticRegression(
        random_state=config.SEED, max_iter=2000  # Increased to ensure convergence
    )

    grid_search = GridSearchCV(
        estimator=base_lr,
        param_grid=config.LR_PARAM_GRID,
        scoring="roc_auc",
        cv=3,  # Inner CV for hyperparameter tuning
        n_jobs=-1,
        verbose=0,
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def train_bagging_ensemble(base_estimator, X_train, y_train):
    """
    Trains a BaggingClassifier ensemble using the provided base estimator.

    Args:
        base_estimator: The optimized base estimator.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.

    Returns:
        BaggingClassifier: The fitted ensemble model.
    """
    bagging_clf = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=config.N_BAGGING_ESTIMATORS,
        random_state=config.SEED,
        n_jobs=-1,
    )

    bagging_clf.fit(X_train, y_train)

    return bagging_clf


def run_cross_validation(X_primary, X_aux, X_meta, y):
    """
    Executes the 5-Fold Stratified Cross-Validation following the ADBEF strategy.

    Args:
        X_primary (np.ndarray): Primary backbone embeddings (MiniLM).
        X_aux (np.ndarray): Auxiliary backbone embeddings (MPNet).
        X_meta (pd.DataFrame or np.ndarray): Numerical metadata features.
        y (pd.Series or np.ndarray): Target labels.

    Returns:
        list: A list of dictionaries, each containing the 'transformer' and 'model' for a fold.
        np.ndarray: Out-of-Fold (OOF) predictions.
    """
    # Ensure inputs are in a format indexable by sklearn splits
    if isinstance(X_meta, pd.DataFrame):
        X_meta = X_meta.values
    if isinstance(y, pd.Series):
        y = y.values

    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    )

    oof_preds = np.zeros(len(y))
    trained_pipelines = []

    print(f"Starting {config.N_FOLDS}-Fold Stratified Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta, y)):
        # 1. Split Data
        X_p_train, X_p_val = X_primary[train_idx], X_primary[val_idx]
        X_a_train, X_a_val = X_aux[train_idx], X_aux[val_idx]
        X_m_train, X_m_val = X_meta[train_idx], X_meta[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 2. Feature Engineering (ADBEF Pipeline)
        # Initialize fresh transformer for this fold to prevent leakage
        transformer = ADBEFTransformer()

        # Fit on TRAIN, Transform TRAIN
        X_train_fused = transformer.fit_transform(X_p_train, X_a_train, X_m_train)

        # Transform VAL (using stats from TRAIN)
        X_val_fused = transformer.transform(X_p_val, X_a_val, X_m_val)

        # 3. Hyperparameter Optimization
        best_lr, best_params = optimize_logistic_regression(X_train_fused, y_train)

        # 4. Ensemble Training
        model = train_bagging_ensemble(best_lr, X_train_fused, y_train)

        # 5. Evaluation
        # Predict probability of class 1 (Success)
        val_preds = model.predict_proba(X_val_fused)[:, 1]
        oof_preds[val_idx] = val_preds

        fold_auc = roc_auc_score(y_val, val_preds)
        print(f"Fold {fold} AUC: {fold_auc}")

        # Store pipeline components
        trained_pipelines.append(
            {"fold": fold, "transformer": transformer, "model": model}
        )

    # Overall Metric
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"Overall CV AUC: {overall_auc}")

    return trained_pipelines, oof_preds


def generate_submission(
    trained_pipelines, X_primary_test, X_aux_test, X_meta_test, test_ids
):
    """
    Generates predictions for the test set using CV-Bagging (averaging across folds).
    Saves the result to the submission file.

    Args:
        trained_pipelines (list): List of trained fold dictionaries.
        X_primary_test (np.ndarray): Test primary embeddings.
        X_aux_test (np.ndarray): Test auxiliary embeddings.
        X_meta_test (pd.DataFrame or np.ndarray): Test metadata.
        test_ids (pd.Series or list): Request IDs for the test set.
    """
    print("Generating predictions for test set...")

    if isinstance(X_meta_test, pd.DataFrame):
        X_meta_test = X_meta_test.values

    n_samples = len(test_ids)
    accumulated_preds = np.zeros(n_samples)

    # Iterate through each trained fold pipeline
    for pipe in trained_pipelines:
        transformer = pipe["transformer"]
        model = pipe["model"]

        # Transform test data using the fold-specific transformer
        # This applies the specific PCA projection and Quantile transform learned in that fold
        X_test_fused = transformer.transform(X_primary_test, X_aux_test, X_meta_test)

        # Predict probabilities
        preds = model.predict_proba(X_test_fused)[:, 1]

        # Accumulate
        accumulated_preds += preds

    # Average predictions
    avg_preds = accumulated_preds / len(trained_pipelines)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"request_id": test_ids, "requester_received_pizza": avg_preds}
    )

    # Save to CSV
    submission_path = config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print(submission_df.head())
