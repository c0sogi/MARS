import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Import library modules
from library.config import Config
from library.utils import HierarchyMap, LabelEncoder
from library.feature_extractor import extract_features, DualBackbone
from library.dataset import FeatureDataset
from library.model import HierarchicalMLP
from library.trainer import train_ensemble
from library.inference import predict_ensemble


def setup_demo_environment():
    """
    Sets up a temporary environment for the demo by overriding Config paths
    and creating a mini-dataset.
    """
    print(">>> Setting up demo environment...")

    # 1. Define Demo Directory
    DEMO_DIR = os.path.join(Config.WORKING_DIR, "demo_run")
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # 2. Create Mini Metadata (Subset of real data)
    # We read the original metadata files provided in ./metadata
    print(">>> Creating mini metadata files...")

    # Load original metadata
    train_meta_full = pd.read_csv(Config.TRAIN_META)
    val_meta_full = pd.read_csv(Config.VAL_META)
    test_meta_full = pd.read_csv(Config.TEST_META)

    # Sample subsets (ensure we have enough for a batch)
    mini_train = train_meta_full.head(64).copy()
    mini_val = val_meta_full.head(32).copy()
    mini_test = test_meta_full.head(32).copy()

    # Save mini metadata
    mini_train_path = os.path.join(DEMO_DIR, "mini_train.csv")
    mini_val_path = os.path.join(DEMO_DIR, "mini_val.csv")
    mini_test_path = os.path.join(DEMO_DIR, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # 3. Override Config Parameters for Speed and Isolation
    # Note: Since Config attributes are static, we modify them on the class directly.

    # Directories
    Config.CACHE_DIR = DEMO_DIR
    Config.MODEL_DIR = os.path.join(DEMO_DIR, "models")

    # Metadata Paths
    Config.TRAIN_META = mini_train_path
    Config.VAL_META = mini_val_path
    Config.TEST_META = mini_test_path

    # Feature Cache Paths (Must update these as they were derived from CACHE_DIR originally)
    Config.TRAIN_FEATURES = os.path.join(Config.CACHE_DIR, "train_features.npy")
    Config.TRAIN_LABELS = os.path.join(Config.CACHE_DIR, "train_labels.npy")
    Config.VAL_FEATURES = os.path.join(Config.CACHE_DIR, "val_features.npy")
    Config.VAL_LABELS = os.path.join(Config.CACHE_DIR, "val_labels.npy")
    Config.TEST_FEATURES = os.path.join(Config.CACHE_DIR, "test_features.npy")
    Config.TEST_IDS = os.path.join(Config.CACHE_DIR, "test_ids.npy")
    Config.HIERARCHY_MAPPING = os.path.join(
        Config.CACHE_DIR, "hierarchy_mapping.parquet"
    )
    Config.SUBMISSION_PATH = os.path.join(Config.CACHE_DIR, "submission.csv")

    # Hyperparameters for Demo
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.ENSEMBLE_SIZE = 1  # Train only 1 model
    Config.TRAIN_BATCH_SIZE = 16  # Small batch size
    Config.EXTRACT_BATCH_SIZE = 16  # Small extraction batch
    Config.PATIENCE = 1  # Minimal patience

    print(">>> Config updated for demo execution.")
    return mini_train, mini_val, mini_test


def demo_feature_extraction():
    """
    Demonstrates feature extraction using the DualBackbone model.
    """
    print("\n>>> [Step 1] Running Feature Extraction...")

    # Force extraction by setting load_cached_data=False
    # This reads BSON files based on our mini-metadata and saves .npy files
    extract_features(load_cached_data=False)

    # Validation
    assert os.path.exists(Config.TRAIN_FEATURES), "Train features file not created"
    assert os.path.exists(Config.TRAIN_LABELS), "Train labels file not created"

    # Check shape
    feats = np.load(Config.TRAIN_FEATURES, mmap_mode="r")
    print(f"    Generated Train Features Shape: {feats.shape}")
    assert (
        feats.shape[1] == Config.TOTAL_FEAT_DIM
    ), f"Feature dim mismatch. Expected {Config.TOTAL_FEAT_DIM}, got {feats.shape[1]}"
    assert (
        feats.shape[0] == 64
    ), f"Sample count mismatch. Expected 64, got {feats.shape[0]}"
    print("    Feature extraction verified.")


def demo_hierarchy_map():
    """
    Demonstrates HierarchyMap usage.
    """
    print("\n>>> [Step 2] Verifying Hierarchy Map...")

    # Initialize map (will compute from category_names.csv and cache it)
    hmap = HierarchyMap(load_cached_data=False)

    # Validation
    assert hmap.get_l3_count() == Config.NUM_CLASSES_L3, "Level 3 class count mismatch"
    assert hmap.get_l1_count() == Config.NUM_CLASSES_L1, "Level 1 class count mismatch"

    # Check mapping for a known ID (from the first row of category_names.csv provided in description)
    # ID: 1000021794 -> ABONNEMENT / SERVICES
    test_id = 1000021794
    l1, l2, l3 = hmap.get_targets(test_id)

    print(f"    Mapped Category {test_id} -> L1:{l1}, L2:{l2}, L3:{l3}")
    assert l3 != -1, "Failed to map valid category ID"
    print("    Hierarchy mapping verified.")


def demo_dataset_loading():
    """
    Demonstrates FeatureDataset loading.
    """
    print("\n>>> [Step 3] Verifying Dataset Loading...")

    hmap = HierarchyMap(load_cached_data=True)

    ds = FeatureDataset(
        feature_path=Config.TRAIN_FEATURES,
        label_path=Config.TRAIN_LABELS,
        hierarchy_map=hmap,
        mode="train",
    )

    # Fetch one sample
    feat, l1, l2, l3 = ds[0]

    assert isinstance(feat, torch.Tensor), "Feature is not a tensor"
    assert feat.shape[0] == Config.TOTAL_FEAT_DIM, "Feature tensor shape incorrect"
    assert isinstance(l1, torch.Tensor), "Label is not a tensor"
    print(f"    Loaded sample: Feature {feat.shape}, Targets ({l1}, {l2}, {l3})")
    print("    Dataset loading verified.")


def demo_training():
    """
    Demonstrates the training loop.
    """
    print("\n>>> [Step 4] Running Training Loop...")

    # Run the ensemble training function from library
    # This uses the Config parameters we overrode (Epochs=1, Batch=16)
    train_ensemble()

    # Validation
    model_path = os.path.join(Config.MODEL_DIR, "ensemble_model_0.pth")
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    print("    Training complete. Model saved.")


def demo_inference():
    """
    Demonstrates inference and submission generation.
    """
    print("\n>>> [Step 5] Running Inference...")

    # Run inference
    submission_df = predict_ensemble()

    # Validation
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"
    assert (
        len(submission_df) == 32
    ), f"Submission length mismatch. Expected 32, got {len(submission_df)}"
    assert (
        "_id" in submission_df.columns and "category_id" in submission_df.columns
    ), "Submission columns incorrect"

    print("    Head of submission:")
    print(submission_df.head())
    print("    Inference verified.")


def main():
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # 1. Setup
        setup_demo_environment()

        # 2. Feature Extraction
        demo_feature_extraction()

        # 3. Hierarchy Map
        demo_hierarchy_map()

        # 4. Dataset
        demo_dataset_loading()

        # 5. Training
        demo_training()

        # 6. Inference
        demo_inference()

        print("\n>>> DEMO COMPLETED SUCCESSFULLY.")

    except AssertionError as e:
        print(f"\n!!! DEMO FAILED: Assertion Error - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! DEMO FAILED: Exception - {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
