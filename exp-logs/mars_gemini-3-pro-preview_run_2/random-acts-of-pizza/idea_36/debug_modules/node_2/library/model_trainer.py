import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import joblib

from library.config import Config
from library.utils import setup_logger, save_pickle, load_pickle
from library.feature_engineering import ContextAwareFusionTransformer, assemble_features

# Initialize Logger
logger = setup_logger("model_trainer")


def train_model(load_cached_data=True):
    """
    Executes the training pipeline:
    1. Loads and combines train/val data.
    2. Performs Stratified K-Fold CV.
    3. Within each fold:
       - Fits ContextAwareFusionTransformer.
       - Tunes BaggingClassifier(LogisticRegression) via GridSearchCV.
       - Generates OOF predictions.
       - Predicts on Test set.
    4. Aggregates Test predictions.
    5. Saves submission file.

    Args:
        load_cached_data (bool): Whether to use cached features/embeddings.

    Returns:
        float: The overall OOF ROC AUC score.
    """
    logger.info("Starting model training pipeline...")

    # ---------------------------------------------------------
    # 1. Data Preparation
    # ---------------------------------------------------------
    # Load Train and Val sets and combine them for Cross-Validation
    # We combine them because the metadata split (80/20) was for static validation,
    # but for K-Fold we want to utilize the entire labeled dataset.
    X_train_part, y_train_part = assemble_features(
        "train", load_cached_data=load_cached_data
    )
    X_val_part, y_val_part = assemble_features("val", load_cached_data=load_cached_data)

    # Stack features and targets
    X = np.vstack([X_train_part, X_val_part])
    y = np.hstack([y_train_part, y_val_part])

    logger.info(f"Combined Training Data Shape: {X.shape}")
    logger.info(f"Target Distribution: {np.bincount(y)}")

    # Load Test Data (for inference during fold loop)
    X_test, _ = assemble_features("test", load_cached_data=load_cached_data)

    # ---------------------------------------------------------
    # 2. Cross-Validation Setup
    # ---------------------------------------------------------
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Storage for OOF predictions and Test predictions
    oof_preds = np.zeros(len(y))
    test_preds_accum = np.zeros((len(X_test), Config.N_FOLDS))

    # Prepare Parameter Grid
    # Map 'base_estimator' (Config) to 'estimator' (sklearn 1.7+)
    # Config keys are like 'base_estimator__C', we convert to 'estimator__C'
    param_grid = {}
    for k, v in Config.PARAM_GRID.items():
        new_key = k.replace("base_estimator", "estimator")
        param_grid[new_key] = v

    logger.info(f"Mapped Parameter Grid: {param_grid}")

    # ---------------------------------------------------------
    # 3. Training Loop
    # ---------------------------------------------------------
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"\n{'='*20} Fold {fold + 1} / {Config.N_FOLDS} {'='*20}")

        # Split Data
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        # --- Feature Engineering (Inside Fold) ---
        # We fit the transformer here to ensure PCA and Quantile stats
        # are learned only from the training fold (prevent leakage).
        logger.info("Fitting ContextAwareFusionTransformer...")
        transformer = ContextAwareFusionTransformer(
            pca_components=Config.AUX_PCA_COMPONENTS,
            interaction_top_k=Config.INTERACTION_TOP_K,
            random_state=Config.SEED,
        )

        # Fit on Train, Transform Train & Val
        X_tr_trans = transformer.fit_transform(X_tr)
        X_va_trans = transformer.transform(X_va)
        X_test_trans = transformer.transform(X_test)

        # --- Model Tuning & Training ---
        # Base Estimator: Logistic Regression
        base_clf = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,  # Increased max_iter for convergence
            random_state=Config.SEED,
        )

        # Ensemble: Bagging Classifier
        # We use n_jobs=1 here to avoid oversubscription when nested in GridSearchCV
        bagging_clf = BaggingClassifier(
            estimator=base_clf,
            n_estimators=Config.N_BAGGING_ESTIMATORS,
            random_state=Config.SEED,
            n_jobs=1,
        )

        # Hyperparameter Tuning
        logger.info("Starting GridSearchCV...")
        grid_search = GridSearchCV(
            estimator=bagging_clf,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=3,  # Internal CV for tuning
            n_jobs=-1,
            verbose=0,
        )

        grid_search.fit(X_tr_trans, y_tr)

        best_model = grid_search.best_estimator_
        logger.info(f"Best Params: {grid_search.best_params_}")
        logger.info(f"Best Internal CV AUC: {grid_search.best_score_}")

        # --- Evaluation (OOF) ---
        val_probs = best_model.predict_proba(X_va_trans)[:, 1]
        oof_preds[val_idx] = val_probs

        fold_auc = roc_auc_score(y_va, val_probs)
        logger.info(f"Fold {fold + 1} OOF AUC: {fold_auc}")

        # --- Inference (Test) ---
        test_probs = best_model.predict_proba(X_test_trans)[:, 1]
        test_preds_accum[:, fold] = test_probs

        # --- Save Pipeline ---
        # Construct a pipeline with the fitted transformer and model for portability
        full_pipeline = Pipeline([("transformer", transformer), ("model", best_model)])

        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.joblib")
        save_pickle(full_pipeline, model_path)
        logger.info(f"Saved pipeline to {model_path}")

    # ---------------------------------------------------------
    # 4. Final Metrics & Submission
    # ---------------------------------------------------------
    total_auc = roc_auc_score(y, oof_preds)
    # Print full precision as requested
    print(f"Overall OOF AUC: {total_auc}")

    # Average predictions across folds (CV-Bagging)
    avg_test_preds = np.mean(test_preds_accum, axis=1)

    # Create Submission DataFrame
    # Load Test Metadata to ensure correct ID alignment
    df_test_meta = pd.read_csv(Config.TEST_META)

    # Handle Debug Mode for Metadata
    if Config.DEBUG:
        logger.info(
            f"DEBUG mode active: Sampling metadata to {Config.MAX_SAMPLES} rows."
        )
        df_test_meta = df_test_meta.head(Config.MAX_SAMPLES)

    submission = pd.DataFrame(
        {Config.ID_COL: df_test_meta[Config.ID_COL], Config.TARGET_COL: avg_test_preds}
    )

    logger.info(f"Saving submission file to {Config.SUBMISSION_PATH}")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    return total_auc
