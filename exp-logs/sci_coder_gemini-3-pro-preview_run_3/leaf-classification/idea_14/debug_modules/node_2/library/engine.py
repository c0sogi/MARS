import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library.configuration import Config
from library.utilities import setup_logger, seed_everything
from library.topology_manager import TopologyManager
from library.custom_pipeline import LeafSpeciesPipeline


def run_cross_validation(debug: bool = False):
    """
    Executes the 10-Fold Stratified Cross-Validation training loop.
    Trains the Manifold-Densified LDA Ensemble and saves models to disk.
    """
    logger = setup_logger()
    seed_everything(Config.SEED)

    logger.info("Starting Cross-Validation Workflow...")

    # 1. Load Data via Topology Manager
    # We load the densified training data which contains 3 centroids per image.
    # Structure: [Centroids A (0..N-1), Centroids B (N..2N-1), Centroids C (2N..3N-1)]
    tm = TopologyManager()
    limit = Config.DEBUG_SAMPLES if debug else None

    X_img_all, X_tab_all, y_all, ids_all = tm.get_densified_train_data(
        load_cached_data=True, limit=limit
    )

    # 2. Identify Unique Samples for Splitting
    # The total length is 3 * N_images. We need N_images for splitting.
    n_samples = len(ids_all) // 3

    # Indices for the Canonical View (Centroid A) are the first n_samples
    indices_canonical = np.arange(n_samples)
    y_canonical = y_all[:n_samples]
    ids_canonical = ids_all[:n_samples]

    logger.info(f"Total densified samples: {len(ids_all)}")
    logger.info(f"Unique images for splitting: {n_samples}")

    # 3. Initialize Cross-Validation
    # Use fewer folds if debugging to save time
    n_folds = 2 if debug else Config.N_FOLDS
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)

    fold_scores = []

    # 4. Training Loop
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(n_samples), y_canonical)
    ):
        logger.info(f"\n--- Starting Fold {fold} ---")

        # --- Construct Densified Training Set ---
        # We want Centroids A, B, and C for all training indices.
        # Indices in X_all corresponding to train_idx:
        # A: train_idx
        # B: train_idx + n_samples
        # C: train_idx + 2 * n_samples
        train_indices_densified = np.concatenate(
            [train_idx, train_idx + n_samples, train_idx + 2 * n_samples]
        )

        X_img_train = X_img_all[train_indices_densified]
        X_tab_train = X_tab_all[train_indices_densified]
        y_train = y_all[train_indices_densified]

        # --- Construct Canonical Validation Set ---
        # We want only Centroid A for validation indices.
        # This matches the inference topology.
        X_img_val = X_img_all[val_idx]
        X_tab_val = X_tab_all[val_idx]
        y_val = y_all[val_idx]

        logger.info(f"Train Set Size: {len(y_train)} (Densified 3x)")
        logger.info(f"Val Set Size:   {len(y_val)} (Canonical 1x)")

        # --- Train Pipeline ---
        pipeline = LeafSpeciesPipeline(
            dino_dim=1024,  # ViT-Large embedding dim
            pca_variance=Config.PCA_VARIANCE,
            tabular_dist=Config.TABULAR_OUTPUT_DIST,
            lda_solver=Config.LDA_SOLVER,
            lda_shrinkage=Config.LDA_SHRINKAGE,
            random_state=Config.SEED,
        )

        pipeline.fit(X_img_train, X_tab_train, y_train)

        # --- Evaluate ---
        y_pred_proba = pipeline.predict_proba(X_img_val, X_tab_val)

        # Calculate Log Loss
        score = log_loss(y_val, y_pred_proba, labels=pipeline.lda.classes_)
        fold_scores.append(score)
        logger.info(f"Fold {fold} Log Loss: {score:.15f}")

        # --- Save Model ---
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.joblib")
        joblib.dump(pipeline, model_path)

    # 5. Summary
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    logger.info("\n=== Cross-Validation Complete ===")
    logger.info(f"Mean Log Loss: {mean_score:.15f} (+/- {std_score:.15f})")


def generate_submission(debug: bool = False):
    """
    Generates the submission file using the trained ensemble.
    """
    logger = setup_logger()
    seed_everything(Config.SEED)

    logger.info("Starting Submission Generation...")

    # 1. Load Test Data
    tm = TopologyManager()
    limit = Config.DEBUG_SAMPLES if debug else None

    # Get canonical inference data (1 centroid per image)
    X_img_test, X_tab_test, _, ids_test = tm.get_canonical_inference_data(
        subset="test", load_cached_data=True, limit=limit
    )

    logger.info(f"Test samples loaded: {len(ids_test)}")

    # 2. Load Sample Submission for Column Mapping
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    class_columns = [c for c in sample_sub.columns if c != "id"]

    # Initialize probability accumulator
    # Shape: (n_test_samples, n_classes)
    # We will use the order of columns in sample_submission to ensure correctness
    n_classes = len(class_columns)
    accumulated_probs = np.zeros((len(ids_test), n_classes))

    # 3. Ensemble Prediction
    n_folds = 2 if debug else Config.N_FOLDS
    models_found = 0

    for fold in range(n_folds):
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.joblib")

        if not os.path.exists(model_path):
            logger.warning(
                f"Model for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        logger.info(f"Predicting with model fold {fold}...")
        pipeline = joblib.load(model_path)

        # Predict probabilities
        # Shape: (n_samples, n_classes_in_training)
        probs = pipeline.predict_proba(X_img_test, X_tab_test)

        # Map probabilities to submission columns
        # pipeline.lda.classes_ contains the class names in the order of 'probs' columns
        model_classes = pipeline.lda.classes_

        # Create a temporary dataframe to handle column alignment easily
        df_probs = pd.DataFrame(probs, columns=model_classes)

        # Reorder columns to match sample_submission and fill missing with 0 (should not happen in CV)
        df_probs_aligned = df_probs.reindex(columns=class_columns, fill_value=0.0)

        accumulated_probs += df_probs_aligned.values
        models_found += 1

    if models_found == 0:
        raise RuntimeError("No models found to generate submission!")

    # 4. Average Probabilities
    avg_probs = accumulated_probs / models_found

    # 5. Apply Clipping
    # Range: [1e-15, 1 - 1e-15]
    epsilon = 1e-15
    avg_probs = np.clip(avg_probs, epsilon, 1 - epsilon)

    # 6. Format Submission
    logger.info("Formatting submission...")
    df_sub = pd.DataFrame(avg_probs, columns=class_columns)
    df_sub.insert(0, "id", ids_test)

    # 7. Save
    output_path = Config.SUBMISSION_PATH
    df_sub.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")
    logger.info(f"Submission shape: {df_sub.shape}")
