import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import setup_logger, set_seed, save_model
from library.embedding_manager import generate_embeddings
from library.pipeline_factory import create_model_pipeline

logger = setup_logger("trainer")


def train_model():
    """
    Orchestrates the 5-Fold Stratified Cross-Validation loop for the MF-ADBE model.
    Performs Grid Search within each fold, evaluates performance, and generates
    submission predictions.
    """
    # 1. Setup
    set_seed(Config.SEED)
    logger.info("Starting training workflow...")

    # 2. Load Data and Embeddings
    # generate_embeddings handles caching and computing if necessary
    X_train_part, y_train_part, X_val_part, y_val_part, X_test, schema = (
        generate_embeddings(load_cached_data=True)
    )

    # Combine the fixed train/val splits into a single dataset for 5-Fold CV
    # This ensures we use all available labeled data for training/validation
    X_full = np.vstack([X_train_part, X_val_part])
    y_full = np.concatenate([y_train_part, y_val_part])

    logger.info(f"Full Training Data Shape: {X_full.shape}")
    logger.info(f"Test Data Shape: {X_test.shape}")

    # 3. Initialize Cross-Validation
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)

    # Storage for predictions
    # Accumulate test probabilities (will average later)
    test_preds_sum = np.zeros(len(X_test), dtype=np.float64)
    # Store OOF predictions for global evaluation
    oof_preds = np.zeros(len(X_full), dtype=np.float64)

    fold_aucs = []

    # 4. Prepare Hyperparameter Grid
    # The pipeline structure is: preprocessor -> classifier (Bagging) -> estimator (LogReg)
    # Config.PARAM_GRID keys are like 'estimator__C'
    # We need to prefix them with 'classifier__' to target the BaggingClassifier's estimator
    param_grid = {f"classifier__{k}": v for k, v in Config.PARAM_GRID.items()}
    logger.info(f"Grid Search Parameters: {param_grid}")

    # 5. Training Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        logger.info(f"--- Starting Fold {fold + 1}/{n_folds} ---")

        # Split data
        X_fold_train, y_fold_train = X_full[train_idx], y_full[train_idx]
        X_fold_val, y_fold_val = X_full[val_idx], y_full[val_idx]

        # Create fresh pipeline for this fold
        pipeline = create_model_pipeline(schema)

        # Configure Grid Search
        # We use n_jobs=1 for GridSearchCV because the BaggingClassifier inside the pipeline
        # is already configured with n_jobs=-1. Nested parallelism can cause contention.
        # Inner CV=3 is sufficient for hyperparameter tuning.
        gs = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=3,
            n_jobs=1,  # Avoid oversubscription
            verbose=0,
        )

        # Fit Grid Search
        logger.info(f"Fitting Grid Search for Fold {fold + 1}...")
        gs.fit(X_fold_train, y_fold_train)

        # Retrieve best model
        best_model = gs.best_estimator_
        best_params = gs.best_params_
        logger.info(f"Fold {fold + 1} Best Params: {best_params}")

        # Evaluate on Fold Validation Set
        val_probs = best_model.predict_proba(X_fold_val)[:, 1]
        fold_auc = roc_auc_score(y_fold_val, val_probs)

        # Log full precision as requested
        logger.info(f"Fold {fold + 1} AUC: {fold_auc}")
        fold_aucs.append(fold_auc)

        # Store OOF predictions
        oof_preds[val_idx] = val_probs

        # Predict on Test Set (Accumulate)
        test_probs = best_model.predict_proba(X_test)[:, 1]
        test_preds_sum += test_probs

        # Save the best model for this fold
        model_filename = f"model_fold_{fold}.joblib"
        model_path = os.path.join(Config.WORKING_DIR, model_filename)
        save_model(best_model, model_path)

    # 6. Global Evaluation
    global_auc = roc_auc_score(y_full, oof_preds)
    mean_auc = np.mean(fold_aucs)

    logger.info("-" * 30)
    logger.info(f"Mean Fold AUC: {mean_auc}")
    logger.info(f"Global OOF AUC: {global_auc}")
    logger.info("-" * 30)

    # 7. Generate Submission
    # Average the test predictions
    avg_test_preds = test_preds_sum / n_folds

    # Load Test Metadata to get request_ids
    if not os.path.exists(Config.METADATA_TEST):
        raise FileNotFoundError(f"Test metadata not found at {Config.METADATA_TEST}")

    df_test_meta = pd.read_csv(Config.METADATA_TEST)

    # Ensure alignment
    if len(df_test_meta) != len(avg_test_preds):
        raise ValueError(
            f"Mismatch between metadata rows ({len(df_test_meta)}) "
            f"and predictions ({len(avg_test_preds)})"
        )

    # Create submission DataFrame
    submission = pd.DataFrame(
        {
            "request_id": df_test_meta["request_id"],
            "requester_received_pizza": avg_test_preds,
        }
    )

    # Save submission
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    logger.info("Training workflow completed successfully.")


if __name__ == "__main__":
    train_model()
