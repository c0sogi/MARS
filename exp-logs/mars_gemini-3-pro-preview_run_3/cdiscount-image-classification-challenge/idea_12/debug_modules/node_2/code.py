import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Import library modules
import library.config as config
from library.hierarchy_utils import HierarchyMapper
from library.feature_extractor import extract_and_cache_features
from library.dataset import CachedFeatureDataset
from library.model import HierarchicalMLP
from library.trainer import train_ensemble
from library.inference import generate_submission


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # ==========================================
    # 1. CONFIGURATION OVERRIDES
    # ==========================================
    # We modify the config module at runtime to create a safe, fast demo environment.
    print("Configuring demo parameters...")

    # Create a specific directory for this demo execution
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Paths
    config.WORKING_DIR = DEMO_DIR
    config.TRAIN_FEATURES = os.path.join(DEMO_DIR, "train_features.npy")
    config.TRAIN_LABELS_L3 = os.path.join(DEMO_DIR, "train_labels_l3.npy")
    config.TRAIN_IDS = os.path.join(DEMO_DIR, "train_ids.npy")

    config.VAL_FEATURES = os.path.join(DEMO_DIR, "val_features.npy")
    config.VAL_LABELS_L3 = os.path.join(DEMO_DIR, "val_labels_l3.npy")
    config.VAL_IDS = os.path.join(DEMO_DIR, "val_ids.npy")

    config.TEST_FEATURES = os.path.join(DEMO_DIR, "test_features.npy")
    config.TEST_IDS = os.path.join(DEMO_DIR, "test_ids.npy")

    config.HIERARCHY_MAPPING = os.path.join(DEMO_DIR, "hierarchy_map.parquet")
    config.CATEGORY_ENCODER = os.path.join(DEMO_DIR, "category_encoder.pkl")

    # Override Model/Training Params for Speed
    config.MODEL_SAVE_PATH_TEMPLATE = os.path.join(
        DEMO_DIR, "demo_model.pth"
    )  # Single model
    config.SUBMISSION_DIR = DEMO_DIR
    config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    config.DEBUG_SAMPLE_SIZE = 50  # Only process 50 samples per dataset
    config.BATCH_SIZE_EXTRACT = 10
    config.BATCH_SIZE_TRAIN = 10
    config.EPOCHS = 1
    config.ENSEMBLE_SIZE = 1  # Train only 1 model instead of 5
    config.NUM_WORKERS = 2  # Low worker count for small data
    config.EARLY_STOPPING_PATIENCE = 1

    # Set seeds
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # ==========================================
    # 2. HIERARCHY MAPPING
    # ==========================================
    print("\n--- Step 2: Verifying Hierarchy Mapper ---")
    mapper = HierarchyMapper(load_cached_data=False)

    # Validation
    assert os.path.exists(
        config.HIERARCHY_MAPPING
    ), "Hierarchy mapping file was not created."
    assert (
        len(mapper.l3_to_cat) == config.NUM_CLASSES_L3
    ), f"Expected {config.NUM_CLASSES_L3} L3 classes, got {len(mapper.l3_to_cat)}"

    # Check consistency: Get an arbitrary category ID, map to index, map back
    test_idx = 10
    cat_id = mapper.get_category_id(test_idx)
    mapped_idx = mapper.get_l3_index(cat_id)
    assert test_idx == mapped_idx, "Mapping consistency check failed."
    print("Hierarchy Mapper initialized and verified.")

    # ==========================================
    # 3. FEATURE EXTRACTION
    # ==========================================
    print("\n--- Step 3: Running Feature Extraction (Subset) ---")
    # This uses the overridden config paths and DEBUG_SAMPLE_SIZE
    extract_and_cache_features(load_cached_data=False)

    # Verify outputs
    assert os.path.exists(config.TRAIN_FEATURES), "Train features .npy missing."
    assert os.path.exists(config.TRAIN_LABELS_L3), "Train labels .npy missing."
    assert os.path.exists(config.TEST_FEATURES), "Test features .npy missing."

    # Check shapes
    train_feats = np.load(config.TRAIN_FEATURES, mmap_mode="r")
    print(f"Extracted Train Features Shape: {train_feats.shape}")
    assert train_feats.shape == (
        config.DEBUG_SAMPLE_SIZE,
        config.INPUT_DIM,
    ), f"Expected shape ({config.DEBUG_SAMPLE_SIZE}, {config.INPUT_DIM}), got {train_feats.shape}"

    # ==========================================
    # 4. DATASET LOADING
    # ==========================================
    print("\n--- Step 4: Verifying Dataset Loading ---")
    # Instantiate dataset
    train_ds = CachedFeatureDataset(
        features_path=config.TRAIN_FEATURES,
        labels_path=config.TRAIN_LABELS_L3,
        ids_path=config.TRAIN_IDS,
        hierarchy_mapper=mapper,
    )

    # Test __getitem__
    features, targets = train_ds[0]
    l1, l2, l3 = targets

    # Verify types and shapes
    assert isinstance(features, torch.Tensor), "Feature is not a tensor."
    assert features.shape[0] == config.INPUT_DIM, "Feature dimension mismatch."
    assert (
        isinstance(l1, torch.Tensor)
        and isinstance(l2, torch.Tensor)
        and isinstance(l3, torch.Tensor)
    ), "Targets are not tensors."
    print("Dataset loaded and verified successfully.")

    # ==========================================
    # 5. MODEL TRAINING
    # ==========================================
    print("\n--- Step 5: Training Demo Model ---")
    # This runs the training loop defined in library.trainer
    # It will use the CachedFeatureDataset internally
    model_paths = train_ensemble()

    assert len(model_paths) == 1, "Expected 1 model path returned."
    assert os.path.exists(model_paths[0]), f"Model file {model_paths[0]} not found."
    print(f"Model trained and saved to: {model_paths[0]}")

    # ==========================================
    # 6. INFERENCE & SUBMISSION
    # ==========================================
    print("\n--- Step 6: Generating Submission ---")
    # Generate submission using the trained model
    submission_df = generate_submission(
        model_paths=model_paths, batch_size=config.BATCH_SIZE_TRAIN
    )

    # Verify Submission
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not created."
    assert len(submission_df) > 0, "Submission DataFrame is empty."

    # Since we used DEBUG_SAMPLE_SIZE for test extraction as well (via extract_dataset logic in feature_extractor),
    # the submission length should match the number of test samples processed.
    # Note: The provided metadata/test.csv has ~700k rows.
    # The feature extractor respects DEBUG_SAMPLE_SIZE.
    # So we expect DEBUG_SAMPLE_SIZE rows in the submission.
    expected_len = config.DEBUG_SAMPLE_SIZE
    # Note: ShardedBSONDataset logic might result in slightly fewer if batching/sharding drops last incomplete batch
    # depending on implementation details, but here it should be exact.
    # Actually, let's check the test_features shape to be sure.
    test_feats_shape = np.load(config.TEST_FEATURES, mmap_mode="r").shape
    assert (
        len(submission_df) == test_feats_shape[0]
    ), f"Submission length {len(submission_df)} does not match test features count {test_feats_shape[0]}."

    assert (
        "_id" in submission_df.columns and "category_id" in submission_df.columns
    ), "Invalid submission columns."
    print("Submission generated and verified.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
