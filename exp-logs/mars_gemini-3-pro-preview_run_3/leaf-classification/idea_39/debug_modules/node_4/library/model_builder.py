import os
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library.config import (
    CACHE_DIR,
    SUBMISSION_DIR,
    PCA_VARIANCE_THRESHOLD,
    LDA_SOLVER,
    LDA_SHRINKAGE,
    N_FOLDS,
    SEED,
    TABULAR_PREFIXES,
)
from library.utils import setup_logger, seed_everything
from library.data_processor import load_dataset

# Initialize logger
logger = setup_logger("model_builder.log")


def create_pipeline(feature_indices):
    """
    Constructs the Stratified Selective-Topology pipeline.

    Args:
        feature_indices (np.ndarray): Array containing start/end indices for
                                      [DINO, ConvNeXt, Tabular] feature blocks.
                                      Expected format: [start, dino_end, conv_end, tab_end]

    Returns:
        sklearn.pipeline.Pipeline: The constructed modeling pipeline.
    """
    # Unpack indices
    # indices are like [0, 1024, 2560, 2752]
    dino_start, dino_end = feature_indices[0], feature_indices[1]
    conv_start, conv_end = feature_indices[1], feature_indices[2]
    tab_start, tab_end = feature_indices[2], feature_indices[3]

    # Define slices for ColumnTransformer
    # Note: slice(start, end) works with numpy arrays in ColumnTransformer
    dino_slice = slice(dino_start, dino_end)
    conv_slice = slice(conv_start, conv_end)
    tab_slice = slice(tab_start, tab_end)

    # 1. Independent Subspace Reduction & Feature Transformation
    # - Visual streams: PCA only (Linear Topology Preservation)
    # - Tabular stream: QuantileTransformer (Gaussianization)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "dino_pca",
                PCA(
                    n_components=PCA_VARIANCE_THRESHOLD,
                    svd_solver="full",
                    random_state=SEED,
                ),
                dino_slice,
            ),
            (
                "conv_pca",
                PCA(
                    n_components=PCA_VARIANCE_THRESHOLD,
                    svd_solver="full",
                    random_state=SEED,
                ),
                conv_slice,
            ),
            (
                "tab_qt",
                QuantileTransformer(output_distribution="normal", random_state=SEED),
                tab_slice,
            ),
        ],
        n_jobs=1,
    )

    # 2. Pipeline Construction
    # - Preprocessor: Feature specific transforms
    # - Scaler: Global Variance Alignment (Crucial for Ledoit-Wolf shrinkage)
    # - Classifier: LDA with shrinkage
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LinearDiscriminantAnalysis(solver=LDA_SOLVER, shrinkage=LDA_SHRINKAGE),
            ),
        ]
    )

    return pipeline


def aggregate_predictions(probs, ids):
    """
    Aggregates predictions by averaging probabilities across the 3 orthogonal centroids per image.

    Args:
        probs (np.ndarray): Predictions for densified samples (N*3, n_classes).
        ids (np.ndarray): IDs corresponding to the densified samples (N*3,).

    Returns:
        tuple: (unique_ids, aggregated_probs)
    """
    df = pd.DataFrame(probs)
    df["id"] = ids

    # Group by ID and take the mean
    grouped = df.groupby("id").mean()

    # Sort by index (id) to ensure alignment
    grouped = grouped.sort_index()

    return grouped.index.values, grouped.values


def run_training_and_submission(debug_sample_size=None, load_cached_data=True):
    """
    Executes the full training loop (K-Fold), evaluation, and submission generation.

    Args:
        debug_sample_size (int, optional): Limit dataset size for debugging.
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    seed_everything(SEED)

    # 1. Load Data
    logger.info("Loading dataset...")
    data = load_dataset(
        load_cached_data=load_cached_data, debug_sample_size=debug_sample_size
    )

    X_train_full = data["train"]["X"]
    y_train_full = data["train"]["y"]
    ids_train_full = data["train"]["ids"]

    # Validation set provided in metadata is technically a hold-out set.
    # However, for the K-Fold strategy described in the idea, we usually combine Train+Val
    # and do K-Fold on the union to maximize data usage, OR we just K-Fold the 'train' set.
    # Given the small data size, we will merge the provided 'train' and 'val' sets
    # to form the full development set for Cross-Validation.
    X_val_provided = data["val"]["X"]
    y_val_provided = data["val"]["y"]
    ids_val_provided = data["val"]["ids"]

    X_dev = np.concatenate([X_train_full, X_val_provided], axis=0)
    y_dev = np.concatenate([y_train_full, y_val_provided], axis=0)
    ids_dev = np.concatenate([ids_train_full, ids_val_provided], axis=0)

    X_test = data["test"]["X"]
    ids_test = data["test"]["ids"]

    classes = data["classes"]
    feature_indices = data["feature_indices"]

    logger.info(f"Development Set Shape: {X_dev.shape}")
    logger.info(f"Test Set Shape: {X_test.shape}")
    logger.info(f"Feature Indices: {feature_indices}")

    # 2. Prepare K-Fold
    # We must split based on unique IDs to avoid leakage of centroids
    unique_ids, unique_indices = np.unique(ids_dev, return_index=True)
    unique_labels = y_dev[unique_indices]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Storage for OOF and Test predictions
    # OOF metrics will be calculated per fold and averaged
    fold_log_losses = []

    # Test predictions accumulator (sum of probabilities)
    test_probs_sum = np.zeros((len(np.unique(ids_test)), len(classes)))

    # Directory for models
    models_dir = os.path.join(CACHE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Save classes for inference consistency
    joblib.dump(classes, os.path.join(models_dir, "classes.pkl"))

    logger.info(f"Starting {N_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx_unique, val_idx_unique) in enumerate(
        skf.split(unique_ids, unique_labels)
    ):
        logger.info(f"--- Fold {fold} ---")

        # Get IDs for this fold
        fold_train_ids = unique_ids[train_idx_unique]
        fold_val_ids = unique_ids[val_idx_unique]

        # Mask the densified dataset
        # np.isin returns boolean mask
        train_mask = np.isin(ids_dev, fold_train_ids)
        val_mask = np.isin(ids_dev, fold_val_ids)

        X_fold_train = X_dev[train_mask]
        y_fold_train = y_dev[train_mask]

        X_fold_val = X_dev[val_mask]
        y_fold_val = y_dev[val_mask]
        ids_fold_val = ids_dev[val_mask]

        # 3. Build and Train Pipeline
        pipeline = create_pipeline(feature_indices)
        pipeline.fit(X_fold_train, y_fold_train)

        # 4. Evaluate on Validation (Aggregated)
        # Predict on all centroids
        val_probs_dense = pipeline.predict_proba(X_fold_val)

        # Aggregate to image level
        val_ids_agg, val_probs_agg = aggregate_predictions(
            val_probs_dense, ids_fold_val
        )

        # Get true labels for aggregated IDs
        # We can create a map from ID to Label (labels are constant per ID)
        id_to_label = dict(zip(ids_fold_val, y_fold_val))
        y_val_agg = np.array([id_to_label[i] for i in val_ids_agg])

        # Calculate Metric
        # Clip probabilities to avoid log(0)
        val_probs_agg = np.clip(val_probs_agg, 1e-15, 1 - 1e-15)
        # Re-normalize
        val_probs_agg = val_probs_agg / val_probs_agg.sum(axis=1, keepdims=True)

        score = log_loss(y_val_agg, val_probs_agg, labels=range(len(classes)))
        fold_log_losses.append(score)

        logger.info(f"Fold {fold} Log Loss: {score:.10f}")

        # 5. Predict on Test
        test_probs_dense = pipeline.predict_proba(X_test)
        test_ids_agg, test_probs_agg = aggregate_predictions(test_probs_dense, ids_test)

        # Accumulate
        test_probs_sum += test_probs_agg

        # Save Model
        joblib.dump(pipeline, os.path.join(models_dir, f"pipeline_fold_{fold}.pkl"))

    # 6. Finalize
    avg_log_loss = np.mean(fold_log_losses)
    logger.info(f"Average Log Loss over {N_FOLDS} folds: {avg_log_loss:.10f}")

    # Average test probabilities
    test_probs_avg = test_probs_sum / N_FOLDS

    # 7. Generate Submission
    logger.info("Generating submission file...")

    # Clip and Normalize
    test_probs_avg = np.clip(test_probs_avg, 1e-15, 1 - 1e-15)
    test_probs_avg = test_probs_avg / test_probs_avg.sum(axis=1, keepdims=True)

    # Create DataFrame
    # test_ids_agg are sorted by ID from the aggregation function
    submission_df = pd.DataFrame(test_probs_avg, columns=classes)
    submission_df.insert(0, "id", test_ids_agg)

    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    logger.info(f"Submission saved to {submission_path}")

    # Validation check on submission format
    logger.info("Submission Head:")
    logger.info(submission_df.head().to_string())
