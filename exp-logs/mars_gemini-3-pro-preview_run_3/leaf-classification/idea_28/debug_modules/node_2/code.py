import os
import shutil
import numpy as np
import pandas as pd
import torch
from functools import partial

# 1. Disable tqdm progress bars globally before importing library modules
import tqdm

tqdm.tqdm = partial(tqdm.tqdm, disable=True)

# 2. Import Library Modules
from library.config import Config, setup_system
from library.utils import load_data, seed_everything
from library.feature_extraction import DeepFeatureExtractor
from library.densification import ManifoldDensifier
from library.modeling import create_selective_pipeline, run_cross_validation


def main():
    print("Starting Self-Contained Demo Script...")

    # 3. Configure for Demo (Speed & Isolation)
    # We use a temporary directory for caching to avoid affecting real runs
    # and use DEBUG mode to process only a handful of images.
    Config.DEBUG = True
    Config.MAX_DEBUG_SAMPLES = 20  # Process only 20 images for speed
    Config.CACHE_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.CACHE_DIR, "submission.csv")
    Config.N_FOLDS = 2  # Minimum folds for cross-validation

    # Clean up any previous demo artifacts
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)

    # Initialize system (directories, seeds)
    setup_system(seed=42)

    # 4. Demonstrate Data Loading
    print("\n[Demo] Loading Data...")
    # Load a debug subset of the training metadata
    df_train = load_data("train", load_cached_data=False, debug=True)
    print(f"Loaded {len(df_train)} training samples (Debug Mode).")

    # Validation: Ensure we have the expected number of samples
    if len(df_train) != Config.MAX_DEBUG_SAMPLES:
        raise AssertionError(
            f"Expected {Config.MAX_DEBUG_SAMPLES} samples, got {len(df_train)}"
        )

    # 5. Demonstrate Deep Feature Extraction
    print("\n[Demo] Extracting Features (DINOv2 + ConvNeXt)...")
    extractor = DeepFeatureExtractor()

    # Extract features for the training subset
    # This handles image loading, rotation (12 views), and model inference
    train_raw = extractor.extract_features("train", load_cached_data=False)

    # Extract features for the test subset (needed for the full workflow)
    test_raw = extractor.extract_features("test", load_cached_data=False)

    # Validation: Check feature shapes
    # Expected: (N, 12, D) for visual features, (N, 192) for tabular
    n_train = len(train_raw["ids"])
    assert train_raw["dino_features"].shape == (
        n_train,
        12,
        1024,
    ), "Incorrect DINOv2 feature shape"
    assert train_raw["conv_features"].shape == (
        n_train,
        12,
        1536,
    ), "Incorrect ConvNeXt feature shape"
    assert train_raw["tabular_features"].shape == (
        n_train,
        192,
    ), "Incorrect tabular feature shape"
    print("Feature extraction successful. Shapes verified.")

    # 6. Demonstrate Manifold Densification
    print("\n[Demo] Densifying Manifolds (Orthogonal View-Set Averaging)...")
    densifier = ManifoldDensifier()

    # Transform N samples (12 views) -> 3N samples (3 orthogonal centroids)
    densified_train = densifier.prepare_densified_dataset(
        train_raw, "train", load_cached_data=False
    )
    densified_test = densifier.prepare_densified_dataset(
        test_raw, "test", load_cached_data=False
    )

    # Validation: Check expansion factor
    n_dense_train = len(densified_train["ids"])
    assert (
        n_dense_train == n_train * 3
    ), f"Densification failed: Expected {n_train*3}, got {n_dense_train}"

    # Validation: Check flattened shapes
    assert (
        densified_train["X_dino"].ndim == 2
    ), "Densified features should be 2D (flattened)"
    assert densified_train["X_dino"].shape[0] == n_dense_train
    print(
        f"Densification successful. Training set expanded to {n_dense_train} samples."
    )

    # 7. Prepare for Cross-Validation (Label Mocking)
    print("\n[Demo] Preparing for Cross-Validation...")

    # Since we are using a tiny random subset (20 samples), StratifiedKFold might fail
    # if classes have < 2 samples. We mock the labels to ensure the demo runs smoothly.
    # We assign 2 synthetic classes evenly across the unique IDs.
    unique_ids = np.unique(densified_train["ids"])
    n_unique = len(unique_ids)

    # Create balanced synthetic labels for unique IDs
    mock_labels_unique = np.array([0, 1] * (n_unique // 2))
    if len(mock_labels_unique) < n_unique:
        mock_labels_unique = np.append(mock_labels_unique, 0)

    # Map these labels to the densified dataset (3 centroids per ID)
    # We create a dictionary mapping ID -> Label
    id_to_label = dict(zip(unique_ids, mock_labels_unique))

    # Apply to densified data
    new_labels = np.array([id_to_label[uid] for uid in densified_train["ids"]])
    densified_train["y"] = new_labels

    print("Labels mocked for robust StratifiedKFold in debug mode.")

    # 8. Demonstrate Pipeline & Cross-Validation
    print("\n[Demo] Running Cross-Validation & Inference...")

    # Verify Pipeline Creation (Unit Test)
    dino_dim = densified_train["X_dino"].shape[1]
    conv_dim = densified_train["X_conv"].shape[1]
    tab_dim = densified_train["X_tab"].shape[1]

    pipeline = create_selective_pipeline(dino_dim, conv_dim, tab_dim)
    assert pipeline is not None, "Pipeline creation failed"

    # Run the full CV loop
    # This fits the pipeline on folds, predicts on validation, and aggregates test predictions
    run_cross_validation(densified_train, densified_test)

    # 9. Final Verification
    print("\n[Demo] Verifying Submission...")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not generated.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated at {Config.SUBMISSION_PATH}")
    print(df_sub.head())

    # Check structure
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert len(df_sub) == len(
        np.unique(densified_test["ids"])
    ), "Submission row count mismatch"

    # Check probability validity
    prob_cols = [c for c in df_sub.columns if c != "id"]
    probs = df_sub[prob_cols].values
    assert np.all(probs >= 0) and np.all(
        probs <= 1
    ), "Probabilities out of range [0, 1]"

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
