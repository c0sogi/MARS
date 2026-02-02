import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import cv2
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.image_utils import load_image, generate_rotated_views
from library.feature_extractor import FeatureExtractor
from library.densification import ManifoldDensifier, get_densified_data
from library.custom_transformers import DualStreamPreprocessor
from library.trainer import Trainer, train_ensemble
from library.inference import InferenceManager


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def setup_demo_environment():
    """
    Sets up a temporary environment for the demo.
    Monkeypatches Config to point to demo directories and use small subsets.
    """
    print("Setting up demo environment...")

    # Define demo paths
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    metadata_dir = os.path.join(demo_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    # Create subset metadata
    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_test = pd.read_csv("./metadata/test.csv")
    orig_val = pd.read_csv("./metadata/val.csv")

    # Take top 10 samples for train, 5 for val, 5 for test
    # Ensure we have at least 2 classes for LDA to work in training
    # We pick samples such that we have multiple classes
    demo_train = orig_train.head(12).copy()
    demo_val = orig_val.head(6).copy()
    demo_test = orig_test.head(6).copy()

    # Save demo metadata
    demo_train_path = os.path.join(metadata_dir, "train.csv")
    demo_val_path = os.path.join(metadata_dir, "val.csv")
    demo_test_path = os.path.join(metadata_dir, "test.csv")

    demo_train.to_csv(demo_train_path, index=False)
    demo_val.to_csv(demo_val_path, index=False)
    demo_test.to_csv(demo_test_path, index=False)

    # Monkeypatch Config
    print("Patching Config parameters for speed...")
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.METADATA_DIR = metadata_dir

    Config.TRAIN_METADATA_PATH = demo_train_path
    Config.VAL_METADATA_PATH = demo_val_path
    Config.TEST_METADATA_PATH = demo_test_path

    Config.TRAIN_FEATURES_CACHE = os.path.join(demo_dir, "train_features.parquet")
    Config.VAL_FEATURES_CACHE = os.path.join(demo_dir, "val_features.parquet")
    Config.TEST_FEATURES_CACHE = os.path.join(demo_dir, "test_features.parquet")

    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    Config.N_FOLDS = 2
    Config.BATCH_SIZE = 4

    # Ensure directories exist
    Config.setup()

    return demo_train, demo_test


def test_image_utils(sample_image_path):
    print("\n=== Testing Image Utils ===")

    # Test load_image
    img = load_image(sample_image_path)
    if img is None:
        print(f"Skipping image utils test: {sample_image_path} not found.")
        return

    print(f"Loaded image shape: {img.shape}")
    assert img.shape == (
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
        3,
    ), "Image resizing failed"

    # Test generate_rotated_views
    # Config.ROTATION_ANGLES has 12 angles
    views = generate_rotated_views(img)
    print(f"Generated {len(views)} rotated views.")
    assert len(views) == 12, "Should generate 12 views"
    assert views[0].shape == img.shape, "Rotated view shape mismatch"

    print("Image Utils validated.")


def test_feature_extractor():
    print("\n=== Testing Feature Extractor ===")

    # Initialize extractor
    extractor = FeatureExtractor()

    # Extract features for the demo training set
    # We use load_cached_data=False to force execution of the model
    print("Extracting features for demo training set...")
    df_features = extractor.extract_dataset_features(
        metadata_path=Config.TRAIN_METADATA_PATH,
        cache_path=Config.TRAIN_FEATURES_CACHE,
        load_cached_data=False,
    )

    print(f"Extracted features dataframe shape: {df_features.shape}")

    # Expected rows: 12 images * 12 views = 144 rows
    expected_rows = 12 * 12
    assert (
        len(df_features) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_features)}"

    # Check columns
    expected_cols = ["id", "view_angle", "dino_features", "convnext_features"]
    for col in expected_cols:
        assert col in df_features.columns, f"Missing column: {col}"

    # Check feature dimensions (DINOv2 Large = 1024, ConvNeXt Large = 1536)
    dino_dim = len(df_features.iloc[0]["dino_features"])
    conv_dim = len(df_features.iloc[0]["convnext_features"])

    print(f"DINO dimension: {dino_dim}")
    print(f"ConvNeXt dimension: {conv_dim}")

    assert dino_dim == 1024, "Incorrect DINOv2 feature dimension"
    assert conv_dim == 1536, "Incorrect ConvNeXt feature dimension"

    print("Feature Extractor validated.")


def test_densification():
    print("\n=== Testing Densification ===")

    # We rely on the cache generated in the previous step
    # Test get_densified_data for training split
    ids, X_dino, X_conv, X_tab, y = get_densified_data(
        split="train", load_cached_data=True
    )

    print(f"Densified Data Shapes:")
    print(f"  IDs: {ids.shape}")
    print(f"  X_dino: {X_dino.shape}")
    print(f"  X_conv: {X_conv.shape}")
    print(f"  X_tab: {X_tab.shape}")
    print(f"  y: {y.shape}")

    # For training, we expect 6 samples per image (3 primary + 3 interpolated)
    # 12 images * 6 samples = 72 samples
    expected_samples = 12 * 6
    assert (
        len(ids) == expected_samples
    ), f"Expected {expected_samples} samples, got {len(ids)}"
    assert X_dino.shape[1] == 1024
    assert X_conv.shape[1] == 1536
    assert X_tab.shape[1] == 192
    assert y is not None

    print("Densification validated.")


def test_custom_transformer():
    print("\n=== Testing DualStreamPreprocessor ===")

    # Create synthetic data
    n_samples = 20
    dino_dim = 1024
    conv_dim = 1536
    tab_dim = 192

    X_dino = np.random.rand(n_samples, dino_dim).astype(np.float32)
    X_conv = np.random.rand(n_samples, conv_dim).astype(np.float32)
    X_tab = np.random.rand(n_samples, tab_dim).astype(np.float32)

    X_concat = np.concatenate([X_dino, X_conv, X_tab], axis=1)

    # Initialize transformer
    # Use PCA variance 0.99
    preprocessor = DualStreamPreprocessor(
        pca_variance=0.99, dino_dim=dino_dim, conv_dim=conv_dim, tab_dim=tab_dim
    )

    # Fit
    preprocessor.fit(X_concat)

    # Transform
    X_trans = preprocessor.transform(X_concat)

    print(f"Original shape: {X_concat.shape}")
    print(f"Transformed shape: {X_trans.shape}")

    # Check that dimensions are reduced (PCA should reduce 1024/1536 significantly)
    assert X_trans.shape[1] < X_concat.shape[1], "Dimensionality reduction failed"
    assert not np.isnan(X_trans).any(), "Transformed data contains NaNs"

    print("Custom Transformer validated.")


def test_training_pipeline():
    print("\n=== Testing Training Pipeline ===")

    # This runs the full training loop using the demo data
    # It will use the cached densified data we verified earlier
    try:
        train_ensemble(load_cached_data=True)
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e

    # Verify models were saved
    models_dir = os.path.join(Config.WORKING_DIR, "models")
    expected_files = ["pipeline_fold_0.pkl", "pipeline_fold_1.pkl", "classes.pkl"]

    for f in expected_files:
        path = os.path.join(models_dir, f)
        assert os.path.exists(path), f"Model file missing: {path}"

    print("Training Pipeline validated.")


def test_inference_pipeline():
    print("\n=== Testing Inference Pipeline ===")

    # We need to ensure test features are extracted first
    # The InferenceManager calls get_densified_data(split='test'), which handles extraction
    # But for the demo, let's trigger it explicitly to see output

    manager = InferenceManager()
    manager.generate_submission(load_cached_data=False)

    # Verify submission file
    sub_path = Config.SUBMISSION_PATH
    assert os.path.exists(sub_path), "Submission file not created"

    df_sub = pd.read_csv(sub_path)
    print(f"Submission shape: {df_sub.shape}")
    print(f"Submission columns: {df_sub.columns[:5]}")

    # Check rows (should match demo test set size = 6)
    assert len(df_sub) == 6, f"Expected 6 predictions, got {len(df_sub)}"

    # Check values are probabilities
    numeric_cols = df_sub.columns.drop("id")
    assert (df_sub[numeric_cols].values >= 0).all(), "Probabilities must be >= 0"
    assert (df_sub[numeric_cols].values <= 1).all(), "Probabilities must be <= 1"

    print("Inference Pipeline validated.")


if __name__ == "__main__":
    seed_everything()

    # 1. Setup
    demo_train, _ = setup_demo_environment()

    # 2. Test Image Utils (using the first image from demo train)
    first_img_path = os.path.join(Config.INPUT_DIR, demo_train.iloc[0]["file_path"])
    test_image_utils(first_img_path)

    # 3. Test Feature Extraction
    test_feature_extractor()

    # 4. Test Densification
    test_densification()

    # 5. Test Custom Transformer
    test_custom_transformer()

    # 6. Test Training
    test_training_pipeline()

    # 7. Test Inference
    test_inference_pipeline()

    print("\nAll demonstration steps completed successfully.")
