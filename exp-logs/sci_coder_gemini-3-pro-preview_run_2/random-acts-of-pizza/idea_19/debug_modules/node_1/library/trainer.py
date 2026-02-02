import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import get_logger, set_seed
from library.data_manager import load_dataset
from library.feature_extractor import generate_sbert_embeddings
from library.pipeline_manager import LPADFPipelineManager

logger = get_logger("Trainer")


def run_cross_validation():
    """
    Executes the 5-Fold Stratified Cross-Validation loop.
    Performs GridSearchCV within each fold, trains the ensemble,
    evaluates on validation sets, and generates test predictions.
    """
    set_seed(Config.RANDOM_SEED)
    logger.info("Starting Cross-Validation Workflow...")

    # 1. Load Data
    # load_dataset handles caching internally
    df_train, df_val, df_test = load_dataset()

    # 2. Generate/Load Embeddings
    # generate_sbert_embeddings handles caching internally
    train_emb, val_emb, test_emb = generate_sbert_embeddings(df_train, df_val, df_test)

    # 3. Initialize Pipeline Manager for helper methods
    manager = LPADFPipelineManager()

    # 4. Merge Features
    logger.info("Merging metadata, text embeddings, and user history...")
    X_train_full = manager.merge_features(df_train, train_emb)
    y_train_full = df_train["requester_received_pizza"].values.astype(int)

    X_val_full = manager.merge_features(df_val, val_emb)
    y_val_full = df_val["requester_received_pizza"].values.astype(int)

    X_test_full = manager.merge_features(df_test, test_emb)

    # 5. Combine Train and Val for Stratified CV
    # We combine them to perform a fresh 5-fold split on the entire labeled dataset
    X_all = pd.concat([X_train_full, X_val_full], axis=0).reset_index(drop=True)
    y_all = np.concatenate([y_train_full, y_val_full], axis=0)

    logger.info(f"Combined Labeled Data Shape: {X_all.shape}")

    # 6. Stratified Cross-Validation Setup
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_SEED
    )

    oof_preds = np.zeros(len(X_all))
    test_preds = np.zeros(len(X_test_full))
    fold_aucs = []

    # Parameter Grid for GridSearchCV
    # Targeting the LogisticRegression estimator inside the BaggingClassifier
    param_grid = {
        "classifier__estimator__C": Config.LR_C_RANGE,
        "classifier__estimator__class_weight": Config.LR_CLASS_WEIGHTS,
    }

    # 7. CV Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        logger.info(f"Processing Fold {fold + 1}/{Config.N_FOLDS}...")

        # Split Data
        X_tr, X_va = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_tr, y_va = y_all[train_idx], y_all[val_idx]

        # Create fresh pipeline for this fold
        pipeline = manager.create_pipeline()

        # Grid Search
        # Using 3-fold inner CV for hyperparameter tuning
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=3, scoring="roc_auc", n_jobs=4, verbose=0
        )

        logger.info("  Tuning hyperparameters...")
        grid_search.fit(X_tr, y_tr)

        best_model = grid_search.best_estimator_
        logger.info(f"  Best Params: {grid_search.best_params_}")

        # Validation Prediction
        val_probs = best_model.predict_proba(X_va)[:, 1]
        oof_preds[val_idx] = val_probs

        # Metric Calculation
        fold_auc = roc_auc_score(y_va, val_probs)
        fold_aucs.append(fold_auc)
        logger.info(f"  Fold {fold + 1} AUC: {fold_auc:.10f}")

        # Test Prediction (Accumulate)
        test_probs = best_model.predict_proba(X_test_full)[:, 1]
        test_preds += test_probs / Config.N_FOLDS

    # 8. Overall Evaluation
    overall_auc = roc_auc_score(y_all, oof_preds)
    mean_auc = np.mean(fold_aucs)

    logger.info("=" * 40)
    logger.info(f"Overall OOF AUC: {overall_auc:.10f}")
    logger.info(f"Mean Fold AUC:   {mean_auc:.10f}")
    logger.info("=" * 40)

    # 9. Save Submission
    submission = pd.DataFrame(
        {"request_id": df_test["request_id"], "requester_received_pizza": test_preds}
    )

    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
