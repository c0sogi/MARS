import os
import shutil
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config, seed_everything
from library.utils import HierarchyMapper, BSONLoader
from library.datasets import RawImageDataset, FeatureDataset
from library.feature_extractor import extract_and_save_features, DualBackbone
from library.model import HierarchicalMLP, generate_submission
from library.trainer import Trainer


def run_demo():
    print("=== Starting Demonstration Pipeline ===\n")

    # ---------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # ---------------------------------------------------------
    print("[1] Configuring environment for fast demonstration...")

    # Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_execution"
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Small subset for speed
    Config.BATCH_SIZE = 10  # Small batch size for the subset
    Config.NUM_EPOCHS = 2  # Minimal epochs
    Config.NUM_WORKERS = 2  # Reduce overhead

    # Update paths in Config based on new WORKING_DIR
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Re-assign output paths to the new working directory
    Config.HIERARCHY_MAPPING_PATH = os.path.join(
        Config.WORKING_DIR, "hierarchy_map.parquet"
    )
    Config.TRAIN_FEATURES = os.path.join(Config.WORKING_DIR, "train_features.npy")
    Config.TRAIN_LABELS = os.path.join(Config.WORKING_DIR, "train_labels.npy")
    Config.VAL_FEATURES = os.path.join(Config.WORKING_DIR, "val_features.npy")
    Config.VAL_LABELS = os.path.join(Config.WORKING_DIR, "val_labels.npy")
    Config.TEST_FEATURES = os.path.join(Config.WORKING_DIR, "test_features.npy")
    Config.TEST_IDS = os.path.join(Config.WORKING_DIR, "test_ids.npy")
    Config.MODEL_CHECKPOINT = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    seed_everything(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Subset Size: {Config.DEBUG_SUBSET_SIZE}")
    print("Configuration complete.\n")

    # ---------------------------------------------------------
    # 2. Validate HierarchyMapper
    # ---------------------------------------------------------
    print("[2] Validating HierarchyMapper...")
    mapper = HierarchyMapper()
    mapper.process(load_cached_data=False, cache_path=Config.HIERARCHY_MAPPING_PATH)

    # Test a known category_id (from category_names.csv description)
    # Example: 1000012764 -> AMENAGEMENT URBAIN - VOIRIE -> AMENAGEMENT URBAIN -> ABRI FUMEUR
    test_cat_id = 1000012764
    l1, l2, l3 = mapper.get_labels(test_cat_id)

    assert (
        l1 is not None and l2 is not None and l3 is not None
    ), "Failed to map valid category_id"
    assert mapper.get_category_id(l3) == test_cat_id, "Inverse mapping failed"

    print(f"Mapped Category {test_cat_id} -> L1:{l1}, L2:{l2}, L3:{l3}")
    print("HierarchyMapper validation passed.\n")

    # ---------------------------------------------------------
    # 3. Validate BSONLoader & RawImageDataset
    # ---------------------------------------------------------
    print("[3] Validating BSONLoader and RawImageDataset...")

    # Check BSONLoader directly
    loader = BSONLoader(Config.TRAIN_BSON)
    # Use the first record from metadata
    train_meta = pd.read_csv(Config.TRAIN_META).iloc[0]
    images = loader.read_images(train_meta["bson_offset"], train_meta["bson_length"])

    assert isinstance(images, list), "read_images should return a list"
    assert len(images) > 0, "Should read at least one image"
    assert isinstance(images[0], np.ndarray), "Image should be numpy array"
    print(
        f"Successfully read {len(images)} image(s) from BSON offset {train_meta['bson_offset']}."
    )

    # Check RawImageDataset
    raw_dataset = RawImageDataset(
        metadata_path=Config.TRAIN_META, bson_path=Config.TRAIN_BSON, subset_size=10
    )
    img_tensor, _id, cat_id = raw_dataset[0]

    assert torch.is_tensor(img_tensor), "Dataset should return a tensor"
    # Shape: (Num_Images, Channels, Height, Width) -> (N, 3, 224, 224) default
    assert img_tensor.dim() == 4, f"Unexpected tensor shape: {img_tensor.shape}"
    assert img_tensor.shape[1] == 3, "Should be RGB"
    print(f"RawImageDataset item shape: {img_tensor.shape}")
    print("BSONLoader and RawImageDataset validation passed.\n")

    # ---------------------------------------------------------
    # 4. Feature Extraction
    # ---------------------------------------------------------
    print("[4] Running Feature Extraction (Subset)...")

    # This function uses DualBackbone to extract features and save .npy files
    # We force load_cached_data=False to ensure the code actually runs
    extract_and_save_features(
        load_cached_data=False, subset_size=Config.DEBUG_SUBSET_SIZE
    )

    assert os.path.exists(Config.TRAIN_FEATURES), "Train features not saved"
    assert os.path.exists(Config.TRAIN_LABELS), "Train labels not saved"
    assert os.path.exists(Config.TEST_FEATURES), "Test features not saved"

    # Verify dimensions of saved features
    feats = np.load(Config.TRAIN_FEATURES)
    assert feats.shape == (
        Config.DEBUG_SUBSET_SIZE,
        Config.INPUT_DIM,
    ), f"Feature shape mismatch. Expected ({Config.DEBUG_SUBSET_SIZE}, {Config.INPUT_DIM}), got {feats.shape}"

    print("Feature extraction completed and validated.\n")

    # ---------------------------------------------------------
    # 5. FeatureDataset & DataLoader
    # ---------------------------------------------------------
    print("[5] Initializing FeatureDatasets...")

    train_dataset = FeatureDataset(
        features_path=Config.TRAIN_FEATURES,
        labels_path=Config.TRAIN_LABELS,
        hierarchy_mapper=mapper,
        mode="train",
    )

    val_dataset = FeatureDataset(
        features_path=Config.VAL_FEATURES,
        labels_path=Config.VAL_LABELS,
        hierarchy_mapper=mapper,
        mode="val",
    )

    test_dataset = FeatureDataset(
        features_path=Config.TEST_FEATURES, ids_path=Config.TEST_IDS, mode="test"
    )

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Verify batch structure
    batch_feats, b_l1, b_l2, b_l3 = next(iter(train_loader))
    assert batch_feats.shape[1] == Config.INPUT_DIM
    assert b_l1.shape[0] == Config.BATCH_SIZE

    print("Datasets and Loaders initialized successfully.\n")

    # ---------------------------------------------------------
    # 6. Model Training (Trainer & HierarchicalMLP)
    # ---------------------------------------------------------
    print("[6] Starting Model Training...")

    trainer = Trainer()

    # Verify model architecture
    assert isinstance(trainer.model, HierarchicalMLP)

    # Run training loop
    trained_model = trainer.fit(train_loader, val_loader)

    assert os.path.exists(Config.MODEL_CHECKPOINT), "Model checkpoint was not created"
    print("Training loop completed successfully.\n")

    # ---------------------------------------------------------
    # 7. Inference & Submission
    # ---------------------------------------------------------
    print("[7] Generating Submission...")

    generate_submission(trained_model, test_loader, mapper)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    # Verify submission format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "_id" in df_sub.columns and "category_id" in df_sub.columns
    ), "Invalid submission columns"
    assert (
        len(df_sub) == Config.DEBUG_SUBSET_SIZE
    ), f"Submission length mismatch: {len(df_sub)}"

    print(f"Submission generated with {len(df_sub)} rows.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
