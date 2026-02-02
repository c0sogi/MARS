import os
import sys
import numpy as np
import pandas as pd
import cv2
import shutil

# Set random seeds for reproducibility
np.random.seed(42)

# Import from the provided library
from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    WORKING_DIR,
    SUBMISSION_FILE,
    MORPHOMETRIC_FEATURES,
    FLOAT_PRECISION,
)
from library.features import (
    extract_single_image_features,
    extract_morphometric_features,
)
from library.data import get_datasets
from library.transforms import (
    MarginalTopology,
    RotationalTopology,
    apply_topology,
)
from library.models import (
    get_lda_expert,
    postprocess_probabilities,
)
from library.ensemble import (
    GreedySelector,
    run_ensemble_pipeline,
)


def clean_working_dir():
    """Helper to clean working directory for fresh tests if needed."""
    # We won't delete the whole dir to avoid permission issues,
    # but we can rely on overwrite behavior of the library functions.
    pass


def test_features_module():
    print("\n=== Testing library.features ===")

    # 1. Test Single Image Extraction
    # Get a valid image path from metadata
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    df_train = pd.read_csv(train_meta_path)

    # Pick the first image
    sample_row = df_train.iloc[0]
    image_rel_path = sample_row["image_path"]
    full_image_path = os.path.join(INPUT_DIR, image_rel_path)

    print(f"Extracting features from: {full_image_path}")
    features = extract_single_image_features(full_image_path)

    # Verify all expected keys are present
    missing_keys = [k for k in MORPHOMETRIC_FEATURES if k not in features]
    if missing_keys:
        raise AssertionError(f"Missing feature keys: {missing_keys}")

    print("Single image feature extraction successful. Keys verified.")

    # 2. Test Batch Extraction (using a small subset of metadata for speed)
    print("Testing batch extraction on subset...")
    df_subset = df_train.head(10).copy()

    # We use a custom dataset name to avoid overwriting the main cache used by the pipeline later
    df_feats = extract_morphometric_features(
        df_subset, dataset_name="demo_subset", load_cached_data=False
    )

    expected_cols = ["id"] + MORPHOMETRIC_FEATURES
    if not all(col in df_feats.columns for col in expected_cols):
        raise AssertionError("Batch extraction DataFrame missing columns.")

    if len(df_feats) != 10:
        raise AssertionError(f"Expected 10 rows, got {len(df_feats)}")

    print("Batch feature extraction successful.")


def test_data_module():
    print("\n=== Testing library.data ===")

    # Test 'global' view loading
    print("Loading 'global' view datasets...")
    (train_data, val_data, test_data, train_full, classes) = get_datasets(
        view="global", load_cached_data=False
    )

    X_train, y_train, ids_train = train_data
    X_val, y_val, ids_val = val_data

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    print(f"Number of classes: {len(classes)}")

    # Verify shapes
    # Global features should have 192 columns (64 margin + 64 shape + 64 texture)
    if X_train.shape[1] != 192:
        raise AssertionError(
            f"Expected 192 features for global view, got {X_train.shape[1]}"
        )

    # Test 'combined' view loading
    print("Loading 'combined' view datasets...")
    (train_c, val_c, _, _, _) = get_datasets(view="combined", load_cached_data=False)
    X_train_c, _, _ = train_c

    # Combined = 192 + len(MORPHOMETRIC_FEATURES)
    expected_dim = 192 + len(MORPHOMETRIC_FEATURES)
    if X_train_c.shape[1] != expected_dim:
        raise AssertionError(
            f"Expected {expected_dim} features for combined view, got {X_train_c.shape[1]}"
        )

    print("Data loading successful.")
    return X_train, y_train  # Return for next tests


def test_transforms_module(X_train):
    print("\n=== Testing library.transforms ===")

    # Create a small subset for quick testing
    X_sub = X_train[:50].copy()

    # 1. Test MarginalTopology
    print("Testing MarginalTopology...")
    mt = MarginalTopology()
    mt.fit(X_sub)
    X_mt = mt.transform(X_sub)

    if X_mt.shape != X_sub.shape:
        raise AssertionError("MarginalTopology changed input shape.")
    if X_mt.dtype != np.float64:  # library.config.FLOAT_PRECISION is float64
        raise AssertionError("MarginalTopology did not return float64.")

    # 2. Test RotationalTopology
    print("Testing RotationalTopology...")
    rt = RotationalTopology(pca_components=10)  # Reduce dims to test PCA
    rt.fit(X_sub)
    X_rt = rt.transform(X_sub)

    if X_rt.shape != (50, 10):
        raise AssertionError(
            f"RotationalTopology output shape mismatch. Expected (50, 10), got {X_rt.shape}"
        )

    print("Transforms successful.")


def test_models_module(X_train, y_train):
    print("\n=== Testing library.models ===")

    # 1. Test LDA Expert
    print("Testing LDA Expert...")
    # Use a small subset
    X_sub = X_train[:100]
    y_sub = y_train[:100]

    # Ensure at least 2 classes in subset for LDA
    if len(np.unique(y_sub)) < 2:
        print("Warning: Subset has < 2 classes, skipping LDA fit test.")
    else:
        clf = get_lda_expert(shrinkage_param="auto")
        clf.fit(X_sub, y_sub)
        probas = clf.predict_proba(X_sub)

        if probas.shape != (100, len(clf.classes_)):
            raise AssertionError("LDA predict_proba shape mismatch.")

    # 2. Test Post-processing
    print("Testing Post-processing...")
    # Create a matrix with a zero row and a normal row
    dummy_probas = np.array(
        [
            [0.0, 0.0, 0.0],  # Sum is 0
            [0.1, 0.9, 0.0],  # Sum is 1
            [10.0, 10.0, 0.0],  # Sum is 20
        ]
    )

    processed = postprocess_probabilities(dummy_probas)

    # Check row sums
    row_sums = processed.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        raise AssertionError(f"Post-processed rows do not sum to 1: {row_sums}")

    # Check clipping (no exact 0.0 or 1.0)
    if np.min(processed) < 1e-15:
        raise AssertionError("Probabilities not clipped correctly at lower bound.")
    if np.max(processed) > (1.0 - 1e-15):
        raise AssertionError("Probabilities not clipped correctly at upper bound.")

    print("Models module successful.")


def test_ensemble_module():
    print("\n=== Testing library.ensemble (GreedySelector) ===")

    # Create synthetic data
    n_samples = 20
    n_classes = 3
    y_true = np.random.randint(0, n_classes, size=n_samples)

    # Create 3 "experts"
    # Expert 1: Random
    pred_1 = np.random.rand(n_samples, n_classes)
    pred_1 /= pred_1.sum(axis=1, keepdims=True)

    # Expert 2: Perfect prediction (one-hot)
    pred_2 = np.zeros((n_samples, n_classes))
    for i, label in enumerate(y_true):
        pred_2[i, label] = 1.0
    pred_2 = postprocess_probabilities(pred_2)

    # Expert 3: Random
    pred_3 = np.random.rand(n_samples, n_classes)
    pred_3 /= pred_3.sum(axis=1, keepdims=True)

    predictions_dict = {
        "expert_random_1": pred_1,
        "expert_perfect": pred_2,
        "expert_random_2": pred_3,
    }

    # Initialize Selector with small iterations
    selector = GreedySelector(max_iterations=5, patience=2)
    selected_keys = selector.fit(predictions_dict, y_true)

    print(f"Selected experts: {selected_keys}")

    # Verify that the perfect expert was selected at least once
    if "expert_perfect" not in selected_keys:
        raise AssertionError("GreedySelector failed to select the best expert.")

    print("Ensemble selection logic successful.")


def run_full_pipeline_demo():
    print("\n=== Running Full Pipeline (library.ensemble.run_ensemble_pipeline) ===")
    # This runs the actual assignment task logic
    # We use load_cached_data=True to speed it up if we ran parts before,
    # but since this is a demo, we let it run naturally.

    run_ensemble_pipeline(load_cached_data=True)

    if os.path.exists(SUBMISSION_FILE):
        print(f"Submission file created at {SUBMISSION_FILE}")
        df = pd.read_csv(SUBMISSION_FILE)
        print(f"Submission shape: {df.shape}")

        # Basic check
        if df.shape[1] != 100:  # id + 99 classes
            raise AssertionError(
                f"Submission has incorrect number of columns: {df.shape[1]}"
            )
    else:
        raise AssertionError("Submission file was not created.")


if __name__ == "__main__":
    print("Starting Demonstration Script...")

    # 1. Test Feature Extraction
    test_features_module()

    # 2. Test Data Loading
    # We capture X_train, y_train to use in subsequent tests
    X_train, y_train = test_data_module()

    # 3. Test Transforms
    test_transforms_module(X_train)

    # 4. Test Models
    test_models_module(X_train, y_train)

    # 5. Test Ensemble Logic (Synthetic)
    test_ensemble_module()

    # 6. Run Full Pipeline
    run_full_pipeline_demo()

    print("\nAll demonstrations completed successfully.")
