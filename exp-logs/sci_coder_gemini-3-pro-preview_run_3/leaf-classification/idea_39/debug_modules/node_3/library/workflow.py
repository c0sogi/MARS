import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library.config import (
    CACHE_DIR,
    SUBMISSION_DIR,
    N_FOLDS,
    SEED,
)
from library.utils import setup_logger, seed_everything
from library.data_processor import load_dataset
from library.model_builder import create_pipeline, aggregate_predictions

# Initialize logger
logger = setup_logger("workflow.log")


def train_ensemble(debug_sample_size=None, load_cached_data=True):
    """
    Executes Stratified K-Fold Cross-Validation training.

    Args:
        debug_sample_size (int, optional): Limit dataset size for debugging.
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    seed_everything(SEED)

    # 1. Load Data
    logger.info("Loading dataset for training...")
    data = load_dataset(
        load_cached_data=load_cached_data, debug_sample_size=debug_sample_size
    )

    # Extract densified data
    X_train_full = data["train"]["X"]
    y_train_full = data["train"]["y"]
    ids_train_full = data["train"]["ids"]

    X_val_provided = data["val"]["X"]
    y_val_provided = data["val"]["y"]
    ids_val_provided = data["val"]["ids"]

    # Combine Train and Val for full K-Fold CV
    X_dev = np.concatenate([X_train_full, X_val_provided], axis=0)
    y_dev = np.concatenate([y_train_full, y_val_provided], axis=0)
    ids_dev = np.concatenate([ids_train_full, ids_val_provided], axis=0)

    classes = data["classes"]
    feature_indices = data["feature_indices"]

    logger.info(f"Development Set Shape: {X_dev.shape}")

    # 2. Setup Directory for Models
    models_dir = os.path.join(CACHE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Save classes for inference consistency
    joblib.dump(classes, os.path.join(models_dir, "classes.pkl"))

    # 3. Stratified K-Fold Setup
    # We must split based on unique IDs to avoid leakage of centroids
    unique_ids, unique_indices = np.unique(ids_dev, return_index=True)
    unique_labels = y_dev[unique_indices]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_log_losses = []

    logger.info(f"Starting {N_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx_unique, val_idx_unique) in enumerate(
        skf.split(unique_ids, unique_labels)
    ):
        logger.info(f"--- Fold {fold} ---")

        # Get IDs for this fold
        fold_train_ids = unique_ids[train_idx_unique]
        fold_val_ids = unique_ids[val_idx_unique]

        # Mask the densified dataset
        train_mask = np.isin(ids_dev, fold_train_ids)
        val_mask = np.isin(ids_dev, fold_val_ids)

        X_fold_train = X_dev[train_mask]
        y_fold_train = y_dev[train_mask]

        X_fold_val = X_dev[val_mask]
        y_fold_val = y_dev[val_mask]
        ids_fold_val = ids_dev[val_mask]

        # Build and Train Pipeline
        pipeline = create_pipeline(feature_indices)
        pipeline.fit(X_fold_train, y_fold_train)

        # Evaluate on Validation (Aggregated)
        # Predict on all centroids
        val_probs_dense = pipeline.predict_proba(X_fold_val)

        # Aggregate to image level
        val_ids_agg, val_probs_agg = aggregate_predictions(
            val_probs_dense, ids_fold_val
        )

        # Get true labels for aggregated IDs
        id_to_label = dict(zip(ids_fold_val, y_fold_val))
        y_val_agg = np.array([id_to_label[i] for i in val_ids_agg])

        # Calculate Metric
        # Clip probabilities to avoid log(0)
        val_probs_agg = np.clip(val_probs_agg, 1e-15, 1 - 1e-15)
        # Re-normalize
        val_probs_agg = val_probs_agg / val_probs_agg.sum(axis=1, keepdims=True)

        score = log_loss(y_val_agg, val_probs_agg, labels=range(len(classes)))
        fold_log_losses.append(score)

        logger.info(f"Fold {fold} Log Loss: {score}")

        # Save Model
        joblib.dump(pipeline, os.path.join(models_dir, f"pipeline_fold_{fold}.pkl"))

    avg_log_loss = np.mean(fold_log_losses)
    logger.info(f"Average Log Loss over {N_FOLDS} folds: {avg_log_loss}")

    return avg_log_loss


def predict_ensemble(debug_sample_size=None, load_cached_data=True):
    """
    Generates predictions using the trained ensemble.

    Args:
        debug_sample_size (int, optional): Limit dataset size for debugging.
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    seed_everything(SEED)

    # 1. Load Data
    logger.info("Loading dataset for inference...")
    data = load_dataset(
        load_cached_data=load_cached_data, debug_sample_size=debug_sample_size
    )

    X_test = data["test"]["X"]
    ids_test = data["test"]["ids"]

    # 2. Load Metadata
    models_dir = os.path.join(CACHE_DIR, "models")
    classes_path = os.path.join(models_dir, "classes.pkl")

    if not os.path.exists(classes_path):
        raise FileNotFoundError("Classes file not found. Run training first.")

    classes = joblib.load(classes_path)

    # 3. Inference Loop
    # Accumulate probabilities from all folds
    # We need to know the number of unique test images to initialize the accumulator
    unique_test_ids = np.unique(ids_test)
    test_probs_sum = np.zeros((len(unique_test_ids), len(classes)))

    # We need to ensure the order of unique_test_ids matches the output of aggregate_predictions
    # aggregate_predictions sorts by ID. np.unique also sorts.
    # So they should align, but we will rely on aggregate_predictions return values.

    logger.info("Starting Ensemble Inference...")

    for fold in range(N_FOLDS):
        model_path = os.path.join(models_dir, f"pipeline_fold_{fold}.pkl")
        if not os.path.exists(model_path):
            logger.warning(
                f"Model for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        pipeline = joblib.load(model_path)

        # Predict on Test (Densified)
        test_probs_dense = pipeline.predict_proba(X_test)

        # Aggregate (Centroids -> Image)
        test_ids_agg, test_probs_agg = aggregate_predictions(test_probs_dense, ids_test)

        # Accumulate
        test_probs_sum += test_probs_agg

    # 4. Average and Format
    test_probs_avg = test_probs_sum / N_FOLDS

    # Clip and Normalize
    test_probs_avg = np.clip(test_probs_avg, 1e-15, 1 - 1e-15)
    test_probs_avg = test_probs_avg / test_probs_avg.sum(axis=1, keepdims=True)

    # 5. Save Submission
    submission_df = pd.DataFrame(test_probs_avg, columns=classes)
    submission_df.insert(0, "id", test_ids_agg)

    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    logger.info(f"Submission saved to {submission_path}")
    logger.info("Submission Head:")
    logger.info("\n" + submission_df.head().to_string())
