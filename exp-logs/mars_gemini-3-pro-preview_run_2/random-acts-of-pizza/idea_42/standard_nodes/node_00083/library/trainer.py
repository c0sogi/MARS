import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from library import config
from library import utils
from library import data_loader
from library import embedding_manager
from library import pipeline_factory

# Initialize Logger
logger = utils.setup_logger("trainer", os.path.join(config.WORKING_DIR, "trainer.log"))


def build_feature_matrix(df, embeddings, split_prefix):
    """
    Constructs the fused feature matrix for a given split.
    Structure: [Title (384) | Body (384) | Global (768) | Metadata (N)]

    Args:
        df (pd.DataFrame): The dataframe containing metadata.
        embeddings (dict): Dictionary containing embedding arrays.
        split_prefix (str): Prefix key for embeddings (e.g., 'train', 'val', 'test').

    Returns:
        np.ndarray: The concatenated feature matrix.
    """
    # 1. Retrieve Embeddings
    # Ensure we copy to avoid modifying the cached array if any in-place ops occur (unlikely but safe)
    title_emb = embeddings[f"{split_prefix}_title_emb"]
    body_emb = embeddings[f"{split_prefix}_body_emb"]
    global_emb = embeddings[f"{split_prefix}_global_emb"]

    # 2. Retrieve Metadata
    # We rely on the order defined in config.NUMERIC_FEATURES
    # data_loader.extract_metadata ensures these columns exist and are numeric
    meta_cols = config.NUMERIC_FEATURES
    meta_data = df[meta_cols].values.astype(np.float32)

    # 3. Concatenate
    # Axis 1 = columns
    X = np.hstack([title_emb, body_emb, global_emb, meta_data])

    return X


def run_cv_training(load_cached_data=True, debug_mode=config.DEBUG_MODE):
    """
    Executes the Stratified Cross-Validation training loop.

    Args:
        load_cached_data (bool): Whether to load processed data from cache.
        debug_mode (bool): Whether to run in debug mode (smaller data).
    """
    utils.set_seed(config.SEED)

    # =========================================================================
    # 1. Load Data and Embeddings
    # =========================================================================
    logger.info("Starting data loading...")
    train_df, val_df, test_df = data_loader.load_dataset(
        load_cached_data=load_cached_data, debug_mode=debug_mode
    )

    logger.info("Retrieving embeddings...")
    embeddings = embedding_manager.get_embeddings(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # =========================================================================
    # 2. Prepare Feature Matrices
    # =========================================================================
    logger.info("Building feature matrices...")

    # We construct matrices for the original splits
    X_train_split = build_feature_matrix(train_df, embeddings, "train")
    y_train_split = train_df[config.TARGET_COL].values.astype(int)

    X_val_split = build_feature_matrix(val_df, embeddings, "val")
    y_val_split = val_df[config.TARGET_COL].values.astype(int)

    X_test = build_feature_matrix(test_df, embeddings, "test")

    # Cite debug_lesson_9: Enforce Strict Separation Between Training and Hold-Out Validation Sets
    # We strictly use the training split for CV to ensure the validation set remains unseen.
    X_full = X_train_split
    y_full = y_train_split

    logger.info(f"Full Training Data Shape: {X_full.shape}")
    logger.info(f"Test Data Shape: {X_test.shape}")

    # =========================================================================
    # 3. Stratified Cross-Validation Loop
    # =========================================================================
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    )

    oof_preds = np.zeros(len(y_full))
    models_dir = os.path.join(config.WORKING_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    fold_scores = []

    # Define Parameter Grid for GridSearchCV
    # We tune the base estimator inside the BaggingClassifier
    # Note: scikit-learn >= 1.2 uses 'estimator', older uses 'base_estimator'.
    # The pipeline_factory uses 'estimator'.
    param_grid = {
        "classifier__estimator__C": config.GRID_C,
        "classifier__estimator__class_weight": config.GRID_CLASS_WEIGHT,
        # Penalty and solver are fixed in factory, but can be added here if needed
    }

    logger.info(f"Starting {config.N_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        logger.info(f"\n--- Fold {fold + 1}/{config.N_FOLDS} ---")

        X_train_fold, y_train_fold = X_full[train_idx], y_full[train_idx]
        X_val_fold, y_val_fold = X_full[val_idx], y_full[val_idx]

        # Create fresh pipeline
        # Dimensions are implicit based on config/factory defaults, but we pass metadata dim
        meta_dim = len(config.NUMERIC_FEATURES)
        pipeline = pipeline_factory.create_model_pipeline(meta_dim=meta_dim)

        # Grid Search for Hyperparameter Tuning
        # We use a smaller CV inside the GridSearch or just fit on the fold?
        # Standard practice: Nested CV is expensive. We usually do GridSearch on the fold's train data
        # using an internal CV (e.g., 3-fold) or just use the fold's train/val if manual.
        # Here we use GridSearchCV with internal 3-fold CV on X_train_fold.
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=0
        )

        logger.info("Running Grid Search...")
        grid_search.fit(X_train_fold, y_train_fold)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        logger.info(f"Best Parameters: {best_params}")

        # Predict on Validation Fold
        # Probability of class 1
        val_preds = best_model.predict_proba(X_val_fold)[:, 1]

        # Calculate Score
        fold_auc = roc_auc_score(y_val_fold, val_preds)
        logger.info(f"Fold {fold + 1} ROC AUC: {fold_auc}")
        fold_scores.append(fold_auc)

        # Store OOF predictions
        oof_preds[val_idx] = val_preds

        # Save Model
        model_path = os.path.join(models_dir, f"model_fold_{fold}.joblib")
        joblib.dump(best_model, model_path)

    # =========================================================================
    # 4. Overall Evaluation
    # =========================================================================
    overall_auc = roc_auc_score(y_full, oof_preds)
    logger.info("\n========================================")
    logger.info(f"Overall OOF ROC AUC: {overall_auc}")
    logger.info(f"Average Fold AUC: {np.mean(fold_scores)}")
    logger.info("========================================")

    # =========================================================================
    # 5. Test Inference & Submission
    # =========================================================================
    logger.info("Generating Test Predictions...")

    test_preds_accum = np.zeros(X_test.shape[0])

    # Load each model and predict
    for fold in range(config.N_FOLDS):
        model_path = os.path.join(models_dir, f"model_fold_{fold}.joblib")
        model = joblib.load(model_path)

        # Predict
        preds = model.predict_proba(X_test)[:, 1]
        test_preds_accum += preds

    # Average predictions
    avg_test_preds = test_preds_accum / config.N_FOLDS

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {
            "request_id": test_df["request_id"],
            "requester_received_pizza": avg_test_preds,
        }
    )

    # Save Submission
    logger.info(f"Saving submission to {config.SUBMISSION_PATH}...")
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)

    logger.info("Training and Inference Complete.")
