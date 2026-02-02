import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings
import shutil

# Import provided library modules
import library.config as cfg
import library.utils as utils
from library.feature_extractor import DualStreamExtractor
from library.data_processor import CentroidGenerator
from library.model_factory import SelectiveFeaturePipeline
from library.train_eval import CrossValidationRunner


def main():
    # ==========================================
    # 1. Setup & Configuration Overrides
    # ==========================================
    print("[1/7] Configuring environment for fast demonstration...")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set seed for reproducibility
    utils.seed_everything(42)

    # Define a demo working directory to avoid conflicts
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Global Config for Speed
    cfg.WORKING_DIR = demo_dir
    cfg.N_FOLDS = 2  # Use 2 folds instead of 10 for speed
    cfg.BATCH_SIZE = 16  # Smaller batch size

    # Redirect metadata paths to demo files (created in step 2)
    cfg.TRAIN_METADATA_PATH = os.path.join(demo_dir, "train.csv")
    cfg.VAL_METADATA_PATH = os.path.join(demo_dir, "val.csv")
    cfg.TEST_METADATA_PATH = os.path.join(demo_dir, "test.csv")
    cfg.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Redirect cache paths to demo directory
    cfg.CACHE_TRAIN_IMG_FEATURES = os.path.join(demo_dir, "train_img.npy")
    cfg.CACHE_TRAIN_IDS = os.path.join(demo_dir, "train_ids.npy")
    cfg.CACHE_TRAIN_LABELS = os.path.join(demo_dir, "train_labels.npy")
    cfg.CACHE_TRAIN_TAB_FEATURES = os.path.join(demo_dir, "train_tab.npy")
    cfg.CACHE_TEST_IMG_FEATURES = os.path.join(demo_dir, "test_img.npy")
    cfg.CACHE_TEST_IDS = os.path.join(demo_dir, "test_ids.npy")
    cfg.CACHE_TEST_TAB_FEATURES = os.path.join(demo_dir, "test_tab.npy")
    cfg.CACHE_CLASSES = os.path.join(demo_dir, "classes.npy")

    # ==========================================
    # 2. Data Subsetting
    # ==========================================
    print("[2/7] Creating data subset for demonstration...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Select top 3 classes to ensure we have enough samples for Stratified K-Fold
    top_classes = orig_train["species"].value_counts().index[:3]

    # Sample a small number of rows per class
    demo_train = (
        orig_train[orig_train["species"].isin(top_classes)].groupby("species").head(4)
    )
    demo_val = (
        orig_val[orig_val["species"].isin(top_classes)].groupby("species").head(2)
    )

    # Fallback if validation set is too small for the selected classes
    if len(demo_val) < 2:
        demo_val = demo_train.iloc[:2].copy()

    demo_test = orig_test.head(6)

    # Save subset metadata
    demo_train.to_csv(cfg.TRAIN_METADATA_PATH, index=False)
    demo_val.to_csv(cfg.VAL_METADATA_PATH, index=False)
    demo_test.to_csv(cfg.TEST_METADATA_PATH, index=False)

    print(f"   Train subset: {len(demo_train)} samples")
    print(f"   Val subset:   {len(demo_val)} samples")
    print(f"   Test subset:  {len(demo_test)} samples")

    # ==========================================
    # 3. Verify Utility Functions
    # ==========================================
    print("[3/7] Verifying Utility functions...")

    # Test clip_and_normalize
    raw_probs = np.array([[10.0, 10.0], [0.0, 0.0]])
    processed_probs = utils.clip_and_normalize(raw_probs)

    # Assertions
    assert np.allclose(
        processed_probs.sum(axis=1), 1.0
    ), "Probabilities must sum to 1 per row"
    assert processed_probs.min() >= 1e-15, "Clipping lower bound failed"
    assert processed_probs.max() <= 1 - 1e-15, "Clipping upper bound failed"
    print("   Utils verification passed.")

    # ==========================================
    # 4. Verify Centroid Generator
    # ==========================================
    print("[4/7] Verifying CentroidGenerator logic...")

    # Mock features: 1 sample, 36 views, 10 dimensions
    # Config defaults: N_EXPERTS=9, VIEWS_PER_CENTROID=4
    # Expert 0 uses indices [0, 9, 18, 27]
    mock_features = np.zeros((1, 36, 10))
    mock_features[0, 0, :] = 1.0
    mock_features[0, 9, :] = 2.0
    mock_features[0, 18, :] = 3.0
    mock_features[0, 27, :] = 4.0

    processor = CentroidGenerator()
    centroids = processor.compute_orthogonal_centroids(mock_features)

    # Expected shape: (1, 9, 10)
    assert centroids.shape == (1, 9, 10), f"Incorrect centroid shape: {centroids.shape}"

    # Expected value for Expert 0: Mean(1, 2, 3, 4) = 2.5
    assert np.allclose(centroids[0, 0, :], 2.5), "Centroid calculation logic failed"
    print("   CentroidGenerator verification passed.")

    # ==========================================
    # 5. Verify Model Factory
    # ==========================================
    print("[5/7] Verifying SelectiveFeaturePipeline...")

    # Temporarily set PCA variance to integer for stable unit testing on random noise
    original_variance = cfg.PCA_VARIANCE
    cfg.PCA_VARIANCE = 2

    # Define small dimensions for test
    d_dino, d_conv, d_tab = 10, 10, 5
    total_dim = d_dino + d_conv + d_tab

    pipeline_factory = SelectiveFeaturePipeline(
        dino_dim=d_dino, conv_dim=d_conv, tab_dim=d_tab
    )
    pipeline = pipeline_factory.create_expert_pipeline()

    # Create random mock data (20 samples, 3 classes)
    X_mock = np.random.rand(20, total_dim)
    y_mock = np.random.randint(0, 3, 20)

    # Ensure all classes are present for LDA
    y_mock[0], y_mock[1], y_mock[2] = 0, 1, 2

    # Fit and Predict
    pipeline.fit(X_mock, y_mock)
    preds = pipeline.predict_proba(X_mock)

    assert preds.shape == (20, 3), "Prediction shape mismatch"

    # Restore config
    cfg.PCA_VARIANCE = original_variance
    print("   Model Factory verification passed.")

    # ==========================================
    # 6. Run Full Pipeline
    # ==========================================
    print("[6/7] Running Full Cross-Validation Pipeline on subset...")

    runner = CrossValidationRunner()

    # Run with load_cached_data=False to force feature extraction on our new subset
    # This uses the DualStreamExtractor (DINOv2 + ConvNeXt)
    runner.run(load_cached_data=False)

    # ==========================================
    # 7. Verify Submission Output
    # ==========================================
    print("[7/7] Verifying Submission output...")

    if not os.path.exists(cfg.SUBMISSION_PATH):
        raise FileNotFoundError(f"Submission file not found at {cfg.SUBMISSION_PATH}")

    sub_df = pd.read_csv(cfg.SUBMISSION_PATH)

    # Check dimensions
    expected_rows = len(demo_test)
    assert (
        len(sub_df) == expected_rows
    ), f"Submission has {len(sub_df)} rows, expected {expected_rows}"

    # Check ID column
    assert "id" in sub_df.columns, "Submission missing 'id' column"

    # Check probability validity
    prob_cols = [c for c in sub_df.columns if c != "id"]
    probs = sub_df[prob_cols].values
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities must be in [0, 1]"

    print("   Submission verification passed.")
    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
