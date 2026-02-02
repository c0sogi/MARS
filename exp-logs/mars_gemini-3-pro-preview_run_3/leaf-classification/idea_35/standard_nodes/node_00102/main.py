import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import (
    seed_everything,
    get_logger,
    calculate_log_loss,
    save_submission,
)
from library.data_processor import LeafDataManager
from library.model_factory import create_classifier


def main():
    # 1. Setup & Configuration
    seed_everything(Config.SEED)
    logger = get_logger(name="runfile")
    logger.info("Starting execution of runfile.py")

    # 2. Data Loading (Train/Val Combined for CV)
    dm = LeafDataManager()
    logger.info("Loading training data...")
    # This triggers feature extraction (if not cached) and densification
    data_train = dm.get_dataset(stage="train", load_cached_data=True)

    X_all = data_train["X"]
    y_all = data_train["y"]
    ids_all = data_train["ids"]

    # Verify densification structure (3 centroids per image)
    assert (
        len(ids_all) % 3 == 0
    ), "Data length must be divisible by 3 (3 centroids per image)."

    # Extract unique IDs and Labels for OOF storage
    # The data is structured as [ID1_A, ID1_B, ID1_C, ID2_A, ...]
    unique_ids = ids_all[::3]
    unique_y = y_all[::3]

    # Identify classes
    classes = np.unique(unique_y)
    n_classes = len(classes)
    logger.info(f"Number of classes: {n_classes}")
    logger.info(f"Total unique training samples: {len(unique_ids)}")

    # Initialize OOF (Out-Of-Fold) prediction array
    # Shape: (N_unique, N_classes)
    oof_preds = np.zeros((len(unique_ids), n_classes))

    # Map ID to index for OOF filling
    id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}

    # Store trained models for test inference
    models = []

    # 3. Stratified K-Fold Cross-Validation
    logger.info(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(
        dm.get_stratified_folds(X_all, y_all, ids_all)
    ):
        logger.info(f"--- Fold {fold} ---")

        # Split data
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]
        ids_val = ids_all[val_idx]  # Densified IDs for validation

        # Create and Train Pipeline
        clf = create_classifier()
        clf.fit(X_train, y_train)
        models.append(clf)

        # Predict on Validation set (Densified: 3 predictions per image)
        # We use predict_proba to get probabilities
        probs_val_densified = clf.predict_proba(X_val)

        # Ensure class order consistency
        if not np.array_equal(clf.classes_, classes):
            # This should rarely happen with sklearn's consistent sorting, but good to be aware.
            # We assume sklearn sorts classes alphabetically.
            pass

        # Aggregate Predictions: Average across the 3 centroids for each image
        # val_idx is sorted, so IDs are grouped: [ID_x, ID_x, ID_x, ID_y, ID_y, ID_y, ...]
        n_val_unique = len(ids_val) // 3

        # Reshape to (N_unique_val, 3, N_classes) and take mean across axis 1
        probs_val_agg = probs_val_densified.reshape(n_val_unique, 3, n_classes).mean(
            axis=1
        )

        # Get the unique IDs corresponding to these aggregated predictions
        unique_ids_val = ids_val[::3]

        # Fill OOF array
        for i, uid in enumerate(unique_ids_val):
            global_idx = id_to_idx[uid]
            oof_preds[global_idx] = probs_val_agg[i]

    # 4. Validation Metric Calculation
    logger.info("Calculating Final Validation Metric...")
    # Use the provided utility which handles clipping/normalization
    # We pass labels=classes to ensure sklearn interprets the string labels correctly vs column order
    val_loss = calculate_log_loss(unique_y, oof_preds, labels=classes)
    print(f"Final Validation Metric: {val_loss}")

    # 5. Failure Analysis
    logger.info("Performing Failure Analysis...")

    # Map string labels to integer indices matching the columns of oof_preds
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_true_indices = np.array([class_to_idx[y] for y in unique_y])

    # Extract predicted probability for the true class
    # oof_preds: (N, C), y_true_indices: (N,)
    true_probs = oof_preds[np.arange(len(unique_y)), y_true_indices]

    # Calculate Error Magnitude (1 - p_true)
    # High error means the model assigned low probability to the correct class
    errors = 1.0 - true_probs

    # Correlate errors with Tabular Features
    # We need to extract the tabular part of X_all.
    # Structure: [DINO (1024) | Conv (1536) | Tabular (192)]
    # Tabular starts at index 2560.
    tabular_start_idx = 1024 + 1536

    # Get tabular features for unique images (take every 3rd row)
    X_tabular_unique = X_all[::3, tabular_start_idx:]

    # Calculate correlation for each feature
    correlations = []
    n_tab_features = X_tabular_unique.shape[1]

    for i in range(n_tab_features):
        # Handle potential constant features (std=0) which give NaN correlation
        feat_col = X_tabular_unique[:, i]
        if np.std(feat_col) == 0:
            corr = 0
        else:
            corr = np.corrcoef(feat_col, errors)[0, 1]

        if np.isnan(corr):
            corr = 0
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Correlation between Error Magnitude and Input Features (Top 5):")
    # We can map index back to name if we had the columns, but index is sufficient for now.
    # Indices 0-63: Margin, 64-127: Shape, 128-191: Texture
    feature_types = ["Margin"] * 64 + ["Shape"] * 64 + ["Texture"] * 64

    for idx, corr in correlations[:5]:
        ftype = feature_types[idx] if idx < len(feature_types) else "Unknown"
        print(f"  Feature {idx} ({ftype}): Correlation = {corr:.4f}")

    # 6. Test Inference & Submission
    logger.info("Generating predictions for Test set...")

    # Load Test Data
    data_test = dm.get_dataset(stage="test", load_cached_data=True)
    X_test = data_test["X"]
    ids_test = data_test["ids"]

    # Initialize accumulator for ensemble averaging
    test_probs_sum = np.zeros((len(ids_test), n_classes))

    # Inference with all fold models
    for model in models:
        # Predict on all test centroids
        probs = model.predict_proba(X_test)
        test_probs_sum += probs

    # Average across models
    test_probs_avg = test_probs_sum / len(models)

    # Aggregate Centroids (N*3 -> N)
    n_test_unique = len(ids_test) // 3
    test_probs_final = test_probs_avg.reshape(n_test_unique, 3, n_classes).mean(axis=1)
    test_ids_final = ids_test[::3]

    # Save Submission
    save_submission(test_ids_final, list(classes), test_probs_final)
    logger.info("Runfile execution complete.")


if __name__ == "__main__":
    main()
