import os
import sys
import numpy as np
import pandas as pd

# Import from the provided library
from library.config import TRAIN_META_PATH, WORKING_DIR, get_full_image_path
from library.utils import (
    set_seed,
    normalize_probabilities,
    calculate_log_loss,
    format_submission,
)
from library.feature_extraction import extract_geometric_features_from_image
from library.data_loader import LeafDataLoader
from library.preprocessing import RobustPreprocessor
from library.model import OASLinearDiscriminant


def run_demonstration():
    print("=== Starting Library Demonstration ===\n")

    # 1. Setup and Utils Verification
    print("--- 1. Configuration & Utilities ---")
    set_seed(42)

    # Verify Probability Normalization
    raw_probs = np.array([[10.0, 10.0], [1.0, 3.0]])
    norm_probs = normalize_probabilities(raw_probs)

    # Check row sums
    row_sums = norm_probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"Normalization failed, sums: {row_sums}"
    print("Probability normalization verified.")
    print(f"Sample normalized prob: {norm_probs[1]}")

    # 2. Feature Extraction
    print("\n--- 2. Feature Extraction ---")
    # Load metadata to get a valid image ID
    if not os.path.exists(TRAIN_META_PATH):
        raise FileNotFoundError(
            "Training metadata not found. Ensure metadata generation was successful."
        )

    df_train_meta = pd.read_csv(TRAIN_META_PATH)
    sample_id = df_train_meta.iloc[0]["id"]
    sample_img_path = get_full_image_path(sample_id)

    print(f"Extracting geometric features for Image ID: {sample_id}")
    geo_feats = extract_geometric_features_from_image(sample_img_path)

    print(f"Extracted Feature Vector: {geo_feats}")
    print(f"Shape: {geo_feats.shape}, Dtype: {geo_feats.dtype}")

    # Expecting 6 geometric features: Area, MajorAxis, Eccentricity, Solidity, Extent, AspectRatio
    assert geo_feats.shape == (
        6,
    ), "Geometric feature extraction returned incorrect shape."
    assert geo_feats.dtype == np.float64, "Geometric features must be float64."

    # 3. Data Loading
    print("\n--- 3. Data Loading (Debug Mode) ---")
    # Initialize loader with a small sample size for speed
    debug_size = 100
    loader = LeafDataLoader(debug_sample_size=debug_size)

    print(f"Loading top {debug_size} samples for Train, Val, and Test...")
    # load_cached_data=True is default, will compute if cache missing
    X_train, y_train, ids_train = loader.get_train_data()
    X_val, y_val, ids_val = loader.get_val_data()
    X_test, ids_test = loader.get_test_data()

    print(f"Train Set: X={X_train.shape}, y={y_train.shape}")
    print(f"Val Set:   X={X_val.shape}, y={y_val.shape}")
    print(f"Test Set:  X={X_test.shape}, ids={ids_test.shape}")

    # Verify feature count (192 Tabular + 6 Geometric = 198)
    assert X_train.shape[1] == 198, f"Expected 198 features, got {X_train.shape[1]}"

    # 4. Preprocessing
    print("\n--- 4. Robust Preprocessing ---")
    preprocessor = RobustPreprocessor()

    print("Fitting preprocessor on training data...")
    preprocessor.fit(X_train)

    print("Transforming datasets...")
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # Check if dimensions changed (VarianceThreshold might drop columns)
    n_features_orig = X_train.shape[1]
    n_features_proc = X_train_proc.shape[1]
    print(
        f"Original Features: {n_features_orig} -> Processed Features: {n_features_proc}"
    )

    # Verify statistics (StandardScaler should make mean ~0, std ~1)
    # We check the first feature that wasn't dropped
    feat_mean = np.mean(X_train_proc[:, 0])
    feat_std = np.std(X_train_proc[:, 0])
    print(f"Feature 0 Stats -> Mean: {feat_mean:.4f}, Std: {feat_std:.4f}")
    assert np.abs(feat_mean) < 1e-6, "Preprocessing mean is not centered."

    # 5. Model Training & Evaluation
    print("\n--- 5. OAS Linear Discriminant Model ---")
    model = OASLinearDiscriminant()

    print("Fitting model...")
    model.fit(X_train_proc, y_train)

    # Identify classes learned by the model (might be a subset if debug_size is small)
    learned_classes = model.classes_
    print(f"Model learned {len(learned_classes)} classes.")

    # Filter validation set to strictly those classes present in the training subset
    # This prevents errors when calculating log loss for classes the model has never seen
    mask_val_classes = np.isin(y_val, learned_classes)
    X_val_subset = X_val_proc[mask_val_classes]
    y_val_subset = y_val[mask_val_classes]

    if len(y_val_subset) > 0:
        print(f"Predicting on {len(y_val_subset)} validation samples...")
        probs_val = model.predict_proba(X_val_subset)

        # Calculate Log Loss
        # We must pass the model's classes as labels to ensure correct column mapping
        loss = calculate_log_loss(y_val_subset, probs_val, labels=learned_classes)
        print(f"Validation Log Loss: {loss:.5f}")

        # Verify probability constraints
        assert np.allclose(probs_val.sum(axis=1), 1.0), "Probabilities do not sum to 1."
    else:
        print("Skipping validation scoring: No overlapping classes in debug subset.")

    # 6. Submission Generation
    print("\n--- 6. Submission Generation ---")
    print("Predicting on Test Set...")
    probs_test = model.predict_proba(X_test_proc)

    submission_file = os.path.join(WORKING_DIR, "demo_submission.csv")

    # Use the utility to format and save
    format_submission(ids_test, learned_classes, probs_test, submission_file)

    # Verify file existence and structure
    assert os.path.exists(submission_file), "Submission file was not created."
    df_sub = pd.read_csv(submission_file)

    print(f"Submission file saved to: {submission_file}")
    print(f"Submission Dimensions: {df_sub.shape}")
    print("First 2 rows of submission:")
    print(df_sub.head(2))

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
