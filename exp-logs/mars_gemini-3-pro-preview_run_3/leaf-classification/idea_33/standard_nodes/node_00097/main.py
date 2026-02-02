import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from library.config import Config
from library.utils import seed_everything, clip_probabilities
from library.feature_extractor import FeatureExtractor
from library.manifold_processor import ManifoldDensifier
from library.topology_pipeline import build_model_pipeline


def main():
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)
    print("Starting pipeline execution...")

    # 2. Load Class Definitions and Target Schema
    # We use sample_submission.csv to define the exact class order required for submission
    sample_sub_path = os.path.join(Config.INPUT_DIR, "sample_submission.csv")
    if not os.path.exists(sample_sub_path):
        raise FileNotFoundError(
            f"Sample submission file not found at {sample_sub_path}"
        )

    sample_sub = pd.read_csv(sample_sub_path)
    # Classes are all columns except 'id'
    class_names = [c for c in sample_sub.columns if c != "id"]

    # Initialize LabelEncoder
    # We fit it on the class names extracted from the sample submission to ensure consistency.
    # LabelEncoder usually sorts classes alphabetically. We will check alignment later.
    le = LabelEncoder()
    le.fit(class_names)

    print(f"Number of classes defined: {len(le.classes_)}")

    # 3. Feature Extraction (Visual Streams)
    # Extracts DINOv2 and ConvNeXt features for 12 rotations per image.
    # Uses caching to avoid re-computation.
    extractor = FeatureExtractor()
    train_raw, val_raw, test_raw = extractor.extract_all(load_cached_data=True)

    # Unpack raw features: (dino_feats, conv_feats, ids)
    train_dino, train_conv, train_ids_raw = train_raw
    val_dino, val_conv, val_ids_raw = val_raw
    test_dino, test_conv, test_ids_raw = test_raw

    # 4. Manifold Densification
    # Aggregates 12 views into 3 orthogonal centroids and densifies the dataset.
    densifier = ManifoldDensifier()

    # Process Train: returns (dino_dense, conv_dense, tab_dense, ids_dense, y_dense)
    train_data = densifier.process_split(
        "train", train_dino, train_conv, train_ids_raw, load_cached_data=True
    )
    X_dino_train, X_conv_train, X_tab_train, ids_train, y_train_raw = train_data

    # Process Val
    val_data = densifier.process_split(
        "val", val_dino, val_conv, val_ids_raw, load_cached_data=True
    )
    X_dino_val, X_conv_val, X_tab_val, ids_val, y_val_raw = val_data

    # Process Test
    test_data = densifier.process_split(
        "test", test_dino, test_conv, test_ids_raw, load_cached_data=True
    )
    X_dino_test, X_conv_test, X_tab_test, ids_test, _ = test_data

    # 5. Data Preparation
    # Concatenate features: [DINO, ConvNeXt, Tabular]
    # The topology_pipeline expects this exact order to apply specific transformers (PCA/QT) to specific columns.
    def concat_features(dino, conv, tab):
        return np.hstack([dino, conv, tab])

    X_train = concat_features(X_dino_train, X_conv_train, X_tab_train)
    X_val = concat_features(X_dino_val, X_conv_val, X_tab_val)
    X_test = concat_features(X_dino_test, X_conv_test, X_tab_test)

    # Encode Labels
    # We transform the string labels from the dataset to integers matching the submission columns
    y_train = le.transform(y_train_raw)
    y_val = le.transform(y_val_raw)

    # Calculate dimensions for the pipeline configuration
    dino_dim = X_dino_train.shape[1]
    conv_dim = X_conv_train.shape[1]
    tab_dim = X_tab_train.shape[1]

    print(f"Feature Dimensions - DINO: {dino_dim}, Conv: {conv_dim}, Tab: {tab_dim}")
    print(f"Total Input Dimension: {X_train.shape[1]}")

    # 6. Training (Stratified K-Fold Ensemble)
    # We must split based on Unique IDs to avoid data leakage (since we have 3 centroids per ID).
    # We reconstruct unique ID information from the densified arrays (every 3rd sample).
    unique_train_indices = np.arange(0, len(ids_train), 3)
    unique_train_ids = ids_train[unique_train_indices]
    unique_train_y = y_train[unique_train_indices]

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    models = []
    print(f"Training {Config.N_FOLDS}-Fold Ensemble...")

    for fold, (train_idx_unique, _) in enumerate(
        skf.split(unique_train_ids, unique_train_y)
    ):
        # Map unique ID indices to densified sample indices.
        # Each unique index i corresponds to densified indices [3*i, 3*i+1, 3*i+2]
        train_idx_dense = []
        for idx in train_idx_unique:
            train_idx_dense.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])

        train_idx_dense = np.array(train_idx_dense)

        # Select fold data
        X_fold_train = X_train[train_idx_dense]
        y_fold_train = y_train[train_idx_dense]

        # Build and Train Pipeline
        model = build_model_pipeline(dino_dim, conv_dim, tab_dim)
        model.fit(X_fold_train, y_fold_train)
        models.append(model)

    print("Training complete.")

    # 7. Validation Inference & Aggregation
    print("Performing Validation...")

    # Predict on the full densified validation set using all models (Ensemble Soft Voting)
    val_probs_sum = np.zeros((X_val.shape[0], len(le.classes_)))

    for model in models:
        # Predict probabilities
        probs = model.predict_proba(X_val)
        val_probs_sum += probs

    # Average over K models
    val_probs_avg_models = val_probs_sum / len(models)

    # Average over Centroids (Test-Time Augmentation)
    # Reshape to (N_val_unique, 3, n_classes) and mean over axis 1 (the 3 centroids)
    n_val_unique = len(val_ids_raw)

    if val_probs_avg_models.shape[0] != n_val_unique * 3:
        raise ValueError(
            f"Validation shape mismatch. Expected {n_val_unique*3}, got {val_probs_avg_models.shape[0]}"
        )

    val_probs_final = val_probs_avg_models.reshape(n_val_unique, 3, -1).mean(axis=1)

    # Get true labels for unique validation images (every 3rd label from densified set)
    y_val_unique = y_val[::3]

    # Clip probabilities to avoid log loss extremes
    val_probs_clipped = clip_probabilities(val_probs_final)

    # Compute Log Loss
    final_metric = log_loss(
        y_val_unique, val_probs_clipped, labels=range(len(le.classes_))
    )

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {final_metric}")

    # 8. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate log loss per sample
    # Get probability assigned to the true class
    true_class_probs = val_probs_clipped[np.arange(len(y_val_unique)), y_val_unique]
    sample_losses = -np.log(true_class_probs)

    # Correlate Error with Tabular Features
    # We use the tabular features of the unique validation samples (every 3rd row)
    X_tab_val_unique = X_tab_val[::3]

    correlations = []
    # Reconstruct feature names for reporting
    feat_names = []
    for prefix in Config.TABULAR_COLS_PREFIXES:
        for i in range(1, 65):
            feat_names.append(f"{prefix}_{i}")

    for i in range(tab_dim):
        feat_vec = X_tab_val_unique[:, i]
        # Check for constant features to avoid NaN correlation
        if np.std(feat_vec) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_vec, sample_losses)[0, 1]
            if np.isnan(corr):
                corr = 0.0
        correlations.append((feat_names[i], corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features Correlated with Error:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 9. Submission Generation
    print("\nGenerating Submission...")

    # Predict on Test Set
    test_probs_sum = np.zeros((X_test.shape[0], len(le.classes_)))
    for model in models:
        probs = model.predict_proba(X_test)
        test_probs_sum += probs

    test_probs_avg_models = test_probs_sum / len(models)

    n_test_unique = len(test_ids_raw)
    test_probs_final = test_probs_avg_models.reshape(n_test_unique, 3, -1).mean(axis=1)

    # Clip probabilities
    test_probs_clipped = clip_probabilities(test_probs_final)

    # Create Submission DataFrame
    # Get unique test IDs (every 3rd from densified)
    test_ids_unique = ids_test[::3]

    submission_df = pd.DataFrame(test_probs_clipped, columns=le.classes_)
    submission_df.insert(0, "id", test_ids_unique)

    # Reorder columns to match sample_submission exactly
    # This ensures that even if LabelEncoder sorted differently, we align with the requirement
    submission_df = submission_df[sample_sub.columns]

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
