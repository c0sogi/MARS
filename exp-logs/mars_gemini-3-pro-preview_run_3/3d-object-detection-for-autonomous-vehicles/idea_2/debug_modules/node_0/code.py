import os
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import box_to_corners, calc_iou_3d, load_lidar
from library.cluster_proposal import GeometricProposalGenerator
from library.feature_engineering import extract_features_single, compute_residuals
from library.dataset_factory import build_tabular_dataset
from library.lgbm_model import ObjectDetector

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("Initializing Demonstration...")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Override Config for speed in this demo
    Config.set_seed(42)
    Config.NUM_BOOST_ROUND = 10
    Config.EARLY_STOPPING_ROUNDS = 5
    Config.RANSAC_ITERATIONS = 20  # Reduced for speed
    Config.NUM_WORKERS = 2  # Reduced overhead for small data

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # --------------------------------------------------------------------------
    print("\n--- Verifying Utility Functions ---")

    # Test Box: [x, y, z, w, l, h, yaw]
    box_a = np.array([10.0, 10.0, 0.0, 2.0, 4.0, 2.0, 0.0])
    box_b = np.array([10.0, 10.0, 0.0, 2.0, 4.0, 2.0, 0.0])  # Identical
    box_c = np.array([20.0, 20.0, 0.0, 2.0, 4.0, 2.0, 0.0])  # No overlap

    # Test IoU
    iou_perfect = calc_iou_3d(box_a, box_b)
    iou_none = calc_iou_3d(box_a, box_c)

    assert np.isclose(iou_perfect, 1.0), f"Expected IoU 1.0, got {iou_perfect}"
    assert np.isclose(iou_none, 0.0), f"Expected IoU 0.0, got {iou_none}"
    print("IoU Calculation: Verified.")

    # Test Corner Generation
    corners = box_to_corners(box_a)
    assert corners.shape == (8, 3), f"Expected (8, 3) corners, got {corners.shape}"
    print("Corner Generation: Verified.")

    # --------------------------------------------------------------------------
    # 3. Verify Proposal Generation (Single File)
    # --------------------------------------------------------------------------
    print("\n--- Verifying Geometric Proposal Generation ---")

    # Load metadata to find a valid sample
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    sample_row = train_meta.iloc[0]
    lidar_path = sample_row["lidar_path"]
    sample_token = sample_row["sample_token"]

    print(f"Processing sample: {sample_token}")

    # Instantiate Generator
    generator = GeometricProposalGenerator()

    # Run pipeline on single file
    # Note: process_lidar_file calls load_lidar internally
    proposals = generator.process_lidar_file(lidar_path)

    # Assertions
    # Note: It's possible for a scene to have no clusters if parameters are strict,
    # but with default params on standard data, we expect some.
    if len(proposals) > 0:
        prop = proposals[0]
        assert "box" in prop, "Proposal missing 'box' key"
        assert "points" in prop, "Proposal missing 'points' key"
        assert prop["box"].shape == (7,), "Invalid box shape"
        print(f"Generated {len(proposals)} proposals from sample.")
    else:
        print(
            "Warning: No proposals generated for this sample (check RANSAC/DBSCAN params)."
        )

    # --------------------------------------------------------------------------
    # 4. Verify Feature Engineering
    # --------------------------------------------------------------------------
    print("\n--- Verifying Feature Engineering ---")

    if len(proposals) > 0:
        # Extract features for the first proposal
        feats = extract_features_single(proposals[0], sample_token)

        # Check for key features
        expected_keys = [
            "eigen_1",
            "linearity",
            "bbox_volume",
            "intensity_mean",
            "sample_token",
        ]
        for k in expected_keys:
            assert k in feats, f"Feature dictionary missing {k}"

        print("Feature Extraction: Verified.")
    else:
        print("Skipping Feature Engineering verification (no proposals).")

    # --------------------------------------------------------------------------
    # 5. Verify Dataset Factory (End-to-End Data Creation)
    # --------------------------------------------------------------------------
    print("\n--- Verifying Dataset Factory ---")

    # Build a tiny dataset (max_samples=20) to ensure speed
    # We force regeneration by setting load_cached_data=False (or ensuring unique name via max_samples)
    print("Building training subset...")
    df_train = build_tabular_dataset(
        mode="train", load_cached_data=False, max_samples=20
    )

    print("Building validation subset...")
    df_val = build_tabular_dataset(mode="val", load_cached_data=False, max_samples=10)

    assert not df_train.empty, "Training dataset is empty!"
    assert not df_val.empty, "Validation dataset is empty!"

    print(f"Train Dataset Shape: {df_train.shape}")
    print(f"Val Dataset Shape: {df_val.shape}")

    # Verify columns
    assert "target_class" in df_train.columns, "Dataset missing target_class"
    assert "dx" in df_train.columns, "Dataset missing regression targets"

    # --------------------------------------------------------------------------
    # 6. Verify Model Training & Inference
    # --------------------------------------------------------------------------
    print("\n--- Verifying Model Training & Inference ---")

    detector = ObjectDetector()

    # Train
    # LightGBM output is suppressed via config params, but we might see some logs
    detector.train(df_train, df_val)

    # Verify models exist
    assert detector.classifier is not None, "Classifier not trained"
    if len(detector.regressors) == 0:
        print(
            "Note: No regressors trained (likely no positive samples in tiny subset)."
        )
    else:
        print(f"Trained {len(detector.regressors)} regressors.")

    # Predict on validation set (simulating test)
    # We drop targets to simulate inference time
    X_test = df_val.drop(
        columns=["target_class"] + Config.REGRESSION_TARGETS, errors="ignore"
    )

    # Run prediction
    preds = detector.predict(X_test)

    if not preds.empty:
        print(f"Generated {len(preds)} predictions.")
        assert "confidence" in preds.columns
        assert "final_x" in preds.columns
        assert "class_name" in preds.columns
    else:
        print(
            "No predictions passed confidence threshold (expected for tiny/random subset)."
        )

    # --------------------------------------------------------------------------
    # 7. Verify Submission Generation
    # --------------------------------------------------------------------------
    print("\n--- Verifying Submission Generation ---")

    # We need to run this on the test set structure usually, but we can test the function
    # using our validation subset acting as test data (it has sample_token).
    # Ideally, we should use the actual test metadata to ensure all IDs are present.

    # Load a tiny subset of test metadata for speed
    test_subset_path = os.path.join(Config.WORKING_DIR, "test_metadata_subset.csv")
    test_meta_orig = pd.read_csv(Config.TEST_METADATA_PATH).head(10)
    test_meta_orig.to_csv(test_subset_path, index=False)

    # Temporarily point Config to this subset
    original_test_path = Config.TEST_METADATA_PATH
    Config.TEST_METADATA_PATH = test_subset_path

    try:
        # Create features for these test samples
        df_test_features = build_tabular_dataset(
            mode="test", load_cached_data=False, max_samples=10
        )

        # Generate submission
        detector.generate_submission(df_test_features)

        assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

        # Verify submission format
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        assert "Id" in sub_df.columns
        assert "PredictionString" in sub_df.columns
        print(f"Submission file verified. Rows: {len(sub_df)}")

    finally:
        # Restore config
        Config.TEST_METADATA_PATH = original_test_path

    print("\n===========================================")
    print("       DEMONSTRATION COMPLETED SUCCESS     ")
    print("===========================================")


if __name__ == "__main__":
    run_demo()
