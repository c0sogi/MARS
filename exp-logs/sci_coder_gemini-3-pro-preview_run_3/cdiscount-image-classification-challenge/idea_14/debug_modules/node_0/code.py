import os
import sys
import numpy as np
import pandas as pd
import torch

# Import provided library modules
from library.configuration import Config
from library.utilities import seed_everything, HierarchyManager
from library.preprocessing import extract_features
from library.dataset import get_training_dataloader, FeatureMemoryDataset
from library.architecture import ConditionalCascadeMLP
from library.engine import Trainer, generate_submission


def run_demo():
    print("=== Starting End-to-End Demo ===")

    # ==========================================
    # 1. CONFIGURE FOR SPEED/DEBUG
    # ==========================================
    print("\n[1] Configuring environment for rapid demonstration...")

    # Patch Config for the demo run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Very small subset for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE_EXTRACT = 10
    Config.BATCH_SIZE_TRAIN = 10
    Config.NUM_WORKERS = 2

    # Define a custom working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update paths in Config to point to this new working directory
    Config.HIERARCHY_MAPPING_PATH = os.path.join(
        Config.WORKING_DIR, "hierarchy_map.parquet"
    )
    Config.TRAIN_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "train_features.npy")
    Config.TRAIN_LABELS_PATH = os.path.join(Config.WORKING_DIR, "train_labels_l3.npy")
    Config.VAL_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "val_features.npy")
    Config.VAL_LABELS_PATH = os.path.join(Config.WORKING_DIR, "val_labels_l3.npy")
    Config.TEST_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "test_features.npy")
    Config.TEST_IDS_PATH = os.path.join(Config.WORKING_DIR, "test_ids.npy")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")

    seed_everything(Config.SEED)
    print("Configuration updated. Debug mode: ON.")

    # ==========================================
    # 2. HIERARCHY MANAGER
    # ==========================================
    print("\n[2] Verifying HierarchyManager...")
    hm = HierarchyManager(load_cached_data=False)

    # Test mapping logic
    sample_cat_id = 1000000055  # Known ID from description
    if sample_cat_id in hm.cat_id_to_l3_idx:
        l3_idx = hm.cat_id_to_l3_idx[sample_cat_id]
        l2_idx = hm.l3_idx_to_l2_idx[l3_idx]
        l1_idx = hm.l3_idx_to_l1_idx[l3_idx]

        print(
            f"Mapped Category {sample_cat_id} -> L3:{l3_idx}, L2:{l2_idx}, L1:{l1_idx}"
        )

        # Test decoding
        decoded_id = hm.decode_predictions([l3_idx])[0]
        if decoded_id != sample_cat_id:
            raise AssertionError(
                f"Decoding failed: Expected {sample_cat_id}, got {decoded_id}"
            )
    else:
        print(
            f"Note: Sample category {sample_cat_id} not in dataset (might be rare). Skipping specific ID check."
        )

    print("HierarchyManager logic verified.")

    # ==========================================
    # 3. FEATURE EXTRACTION
    # ==========================================
    print("\n[3] Running Feature Extraction (DualBackbone)...")

    # Extract Train
    print("-> Extracting Train...")
    train_feats, train_aux = extract_features(
        split="train", load_cached_data=False, debug=True
    )
    if train_feats.shape != (Config.DEBUG_SAMPLE_SIZE, Config.TOTAL_FEATURE_DIM):
        raise AssertionError(f"Train features shape mismatch: {train_feats.shape}")

    # Extract Val
    print("-> Extracting Val...")
    val_feats, val_aux = extract_features(
        split="val", load_cached_data=False, debug=True
    )
    if val_feats.shape != (Config.DEBUG_SAMPLE_SIZE, Config.TOTAL_FEATURE_DIM):
        raise AssertionError(f"Val features shape mismatch: {val_feats.shape}")

    # Extract Test
    print("-> Extracting Test...")
    test_feats, test_aux = extract_features(
        split="test", load_cached_data=False, debug=True
    )
    if test_feats.shape != (Config.DEBUG_SAMPLE_SIZE, Config.TOTAL_FEATURE_DIM):
        raise AssertionError(f"Test features shape mismatch: {test_feats.shape}")

    print("Feature extraction verified.")

    # ==========================================
    # 4. DATASET & DATALOADER
    # ==========================================
    print("\n[4] Verifying FeatureMemoryDataset & DataLoader...")

    train_loader = get_training_dataloader(split="train", debug=True)

    # Fetch one batch
    batch = next(iter(train_loader))
    feats, l1, l2, l3 = batch

    print(
        f"Batch shapes -> Features: {feats.shape}, L1: {l1.shape}, L2: {l2.shape}, L3: {l3.shape}"
    )

    if feats.shape[1] != Config.TOTAL_FEATURE_DIM:
        raise AssertionError("Feature dimension in DataLoader is incorrect.")
    if l1.shape[0] != feats.shape[0]:
        raise AssertionError("Batch size mismatch between features and labels.")

    print("Dataset and DataLoader verified.")

    # ==========================================
    # 5. MODEL TRAINING
    # ==========================================
    print("\n[5] Running Training Loop (ConditionalCascadeMLP)...")

    val_loader = get_training_dataloader(split="val", debug=True)

    trainer = Trainer(model_save_path=MODEL_SAVE_PATH)

    # Run fit
    best_acc = trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError("Model file was not saved after training.")

    print(f"Training complete. Best Val Accuracy: {best_acc:.4f}")

    # ==========================================
    # 6. INFERENCE & SUBMISSION
    # ==========================================
    print("\n[6] Generating Submission...")

    # Create test loader manually to ensure we use the cached test features
    test_dataset = FeatureMemoryDataset(split="test", debug=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE_TRAIN,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    generate_submission([MODEL_SAVE_PATH], test_loader)

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Rows: {len(sub_df)}")
    print(sub_df.head())

    if len(sub_df) != Config.DEBUG_SAMPLE_SIZE:
        raise AssertionError(
            f"Submission row count mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(sub_df)}"
        )

    if list(sub_df.columns) != ["_id", "category_id"]:
        raise AssertionError("Submission columns are incorrect.")

    print("Submission generation verified.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
