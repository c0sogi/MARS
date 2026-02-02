import os
import shutil
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, ensure_directory
from library.image_processing import load_image, generate_rotated_views
from library.feature_extraction import FeatureExtractor
from library.manifold_densification import (
    get_densified_train_data,
    get_densified_test_data,
    compute_orthogonal_centroids,
)
from library.modeling import train_and_evaluate, generate_submission


def main():
    # 1. Setup
    print("=== Starting Demonstration Script ===")
    seed_everything(42)

    # Define paths for the demo
    DEMO_DIR = "./working/demo_data"
    DEMO_OUTPUT_DIR = "./working/demo_run"

    # Clean up previous runs if they exist
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    if os.path.exists(DEMO_OUTPUT_DIR):
        shutil.rmtree(DEMO_OUTPUT_DIR)

    ensure_directory(DEMO_DIR + "/")
    ensure_directory(DEMO_OUTPUT_DIR + "/")

    # 2. Create Data Subsets (Optimization for Speed)
    print("\n[1/6] Creating dataset subsets...")

    # Load original metadata
    df_train_full = pd.read_csv("./metadata/train.csv")
    df_val_full = pd.read_csv("./metadata/val.csv")
    df_test_full = pd.read_csv("./metadata/test.csv")

    # Select 3 classes that exist in both train and val to ensure valid stratification
    # We need enough samples for 2-fold CV (at least 2 per class)
    # The original dataset has ~8 samples per class in train
    common_classes = np.intersect1d(
        df_train_full["species"].unique(), df_val_full["species"].unique()
    )
    selected_classes = common_classes[:3]
    print(f"Selected classes for demo: {selected_classes}")

    # Filter data
    demo_train = df_train_full[df_train_full["species"].isin(selected_classes)].copy()
    demo_val = df_val_full[df_val_full["species"].isin(selected_classes)].copy()
    demo_test = df_test_full.iloc[:5].copy()  # Arbitrary 5 test images

    # Save demo metadata
    demo_train_path = os.path.join(DEMO_DIR, "train.csv")
    demo_val_path = os.path.join(DEMO_DIR, "val.csv")
    demo_test_path = os.path.join(DEMO_DIR, "test.csv")

    demo_train.to_csv(demo_train_path, index=False)
    demo_val.to_csv(demo_val_path, index=False)
    demo_test.to_csv(demo_test_path, index=False)

    print(f"Demo Train Samples: {len(demo_train)}")
    print(f"Demo Val Samples:   {len(demo_val)}")
    print(f"Demo Test Samples:  {len(demo_test)}")

    # 3. Patch Configuration
    print("\n[2/6] Patching Config for Demo Environment...")
    # We override the Config attributes to point to our demo environment
    Config.WORKING_DIR = DEMO_OUTPUT_DIR
    Config.TRAIN_METADATA_PATH = demo_train_path
    Config.VAL_METADATA_PATH = demo_val_path
    Config.TEST_METADATA_PATH = demo_test_path
    Config.SUBMISSION_PATH = os.path.join(DEMO_OUTPUT_DIR, "submission.csv")
    Config.N_FOLDS = 2  # Reduce folds to 2 for speed

    # Update cache paths to reside in the demo output directory
    Config.CACHE_CLASSES = os.path.join(DEMO_OUTPUT_DIR, "classes.npy")
    Config.CACHE_TRAIN_IMG_FEATURES = os.path.join(
        DEMO_OUTPUT_DIR, "train_img_features.npy"
    )
    Config.CACHE_TRAIN_TAB_FEATURES = os.path.join(
        DEMO_OUTPUT_DIR, "train_tab_features.npy"
    )
    Config.CACHE_TRAIN_IDS = os.path.join(DEMO_OUTPUT_DIR, "train_ids.npy")
    Config.CACHE_TRAIN_LABELS = os.path.join(DEMO_OUTPUT_DIR, "train_labels.npy")
    Config.CACHE_TEST_IMG_FEATURES = os.path.join(
        DEMO_OUTPUT_DIR, "test_img_features.npy"
    )
    Config.CACHE_TEST_TAB_FEATURES = os.path.join(
        DEMO_OUTPUT_DIR, "test_tab_features.npy"
    )
    Config.CACHE_TEST_IDS = os.path.join(DEMO_OUTPUT_DIR, "test_ids.npy")

    # Also update densified cache paths (hardcoded in manifold_densification.py, but uses Config.WORKING_DIR)
    # Since we updated Config.WORKING_DIR, the functions in manifold_densification.py will use it automatically.

    # 4. Verify Image Processing Components
    print("\n[3/6] Verifying Image Processing...")
    sample_rel_path = demo_train.iloc[0]["file_path"]
    sample_full_path = os.path.join(Config.INPUT_DIR, sample_rel_path)

    # Test load_image
    img_tensor = load_image(sample_full_path)
    print(f"Loaded image shape: {img_tensor.shape}")
    assert img_tensor.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Expected (3, {Config.IMAGE_SIZE}, {Config.IMAGE_SIZE}), got {img_tensor.shape}"
    assert (
        img_tensor.min() >= 0.0 and img_tensor.max() <= 1.0
    ), "Image tensor not normalized [0, 1]"

    # Test generate_rotated_views
    views_tensor = generate_rotated_views(img_tensor)
    print(f"Rotated views shape: {views_tensor.shape}")
    assert views_tensor.shape == (
        Config.NUM_ROTATIONS,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Expected ({Config.NUM_ROTATIONS}, 3, ...), got {views_tensor.shape}"

    # 5. Verify Feature Extraction & Densification Logic
    print("\n[4/6] Verifying Feature Extraction & Densification...")

    # Initialize Extractor
    extractor = FeatureExtractor()

    # Extract features for a mini-batch (first 2 views)
    mini_batch = views_tensor[:2]
    features = extractor.extract_batch(mini_batch)
    print(f"Extracted features shape: {features.shape}")

    # Expected dimension: DINO (1024) + ConvNeXt (1536) = 2560
    expected_dim = 1024 + 1536
    assert features.shape == (
        2,
        expected_dim,
    ), f"Expected (2, {expected_dim}), got {features.shape}"

    # Verify Centroid Computation Logic
    # Create dummy features: (1 sample, 36 views, 10 dims)
    dummy_feats = np.random.rand(1, 36, 10).astype(np.float32)
    centroids = compute_orthogonal_centroids(dummy_feats)
    print(f"Centroids shape: {centroids.shape}")
    # Expected: (1, 9, 10) because 36 views / 4 orthogonal views = 9 centroids
    assert centroids.shape == (1, 9, 10), f"Expected (1, 9, 10), got {centroids.shape}"

    # 6. Run Training Pipeline
    print(
        "\n[5/6] Running Training Pipeline (Feature Extraction + Densification + LDA)..."
    )
    # This function orchestrates the entire training process:
    # 1. Extracts features from images in demo_train.csv (and caches them)
    # 2. Densifies the manifold (expands dataset by 9x)
    # 3. Runs Stratified K-Fold with DualStreamLDA
    train_and_evaluate(load_cached_data=False)

    # Verify model artifacts
    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pkl")
        assert os.path.exists(model_path), f"Model for fold {fold} was not saved."

    classes_path = Config.CACHE_CLASSES
    assert os.path.exists(classes_path), "Classes metadata file was not saved."
    saved_classes = np.load(classes_path, allow_pickle=True)
    assert len(saved_classes) == 3, f"Expected 3 classes, found {len(saved_classes)}"

    # 7. Run Inference Pipeline
    print("\n[6/6] Running Inference Pipeline...")
    # This function:
    # 1. Extracts features from demo_test.csv
    # 2. Densifies test data (N -> N*9 structure)
    # 3. Loads models and predicts
    # 4. Aggregates predictions (Test-Time Aggregation)
    generate_submission(load_cached_data=False)

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission Head:")
    print(df_sub.head())

    # Check dimensions
    # Columns should be id + 3 species
    assert df_sub.shape == (
        len(demo_test),
        4,
    ), f"Submission shape mismatch. Expected ({len(demo_test)}, 4), got {df_sub.shape}"

    # Check ID alignment
    assert sorted(df_sub["id"].tolist()) == sorted(
        demo_test["id"].tolist()
    ), "Submission IDs do not match Test IDs."

    print("\n=== Demonstration Completed Successfully ===")
    print(f"Output stored in: {Config.WORKING_DIR}")


if __name__ == "__main__":
    main()
