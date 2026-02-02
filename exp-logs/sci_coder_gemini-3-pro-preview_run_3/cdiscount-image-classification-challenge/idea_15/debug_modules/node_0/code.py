import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, HierarchyMapper
from library.feature_engine import FeatureEngine
from library.feature_dataset import RamFeatureDataset
from library.model import DeepFeatureCascade
from library.trainer import ModelTrainer
from library.inference import EnsemblePredictor


def run_demo():
    print("=== Starting Demonstration Script ===")

    # ==========================================
    # 1. SETUP & CONFIGURATION OVERRIDES
    # ==========================================
    # We override Config parameters to run a fast, minimal execution
    # without modifying the original config.py file.

    print("Configuring environment for demo run...")

    # Use a specific directory for demo outputs to avoid overwriting real work
    Config.WORKING_DIR = "./working/demo_execution"
    Config.make_dirs()

    # Update output paths to point to the demo directory
    Config.HIERARCHY_MAPPING_PATH = os.path.join(
        Config.WORKING_DIR, "hierarchy_map.parquet"
    )

    # Feature paths
    Config.TRAIN_FEATURES = os.path.join(Config.WORKING_DIR, "train_features.npy")
    Config.TRAIN_LABELS = os.path.join(Config.WORKING_DIR, "train_labels.npy")
    Config.VAL_FEATURES = os.path.join(Config.WORKING_DIR, "val_features.npy")
    Config.VAL_LABELS = os.path.join(Config.WORKING_DIR, "val_labels.npy")
    Config.TEST_FEATURES = os.path.join(Config.WORKING_DIR, "test_features.npy")
    Config.TEST_IDS = os.path.join(Config.WORKING_DIR, "test_ids.npy")

    # Model and Submission paths
    Config.MODEL_CHECKPOINT = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Enable Debug Mode to process only a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 20  # Process only 20 images per split

    # Training Hyperparameters for Demo
    Config.EPOCHS = 1
    Config.EXTRACT_BATCH_SIZE = 4
    Config.TRAIN_BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Disable multiprocessing for tiny data to avoid overhead

    # Ensure reproducibility
    seed_everything(Config.SEED)
    print("Configuration complete.")

    # ==========================================
    # 2. HIERARCHY MAPPING
    # ==========================================
    print("\n[Step 1/6] Testing HierarchyMapper...")
    # Initialize mapper (builds and caches the mapping)
    mapper = HierarchyMapper(load_cached_data=False)

    # Validation
    if not os.path.exists(Config.HIERARCHY_MAPPING_PATH):
        raise FileNotFoundError("Hierarchy mapping file was not created.")

    # Test mapping logic with a known category from the dataset
    # Category ID: 1000000055 -> L3 Index mapping check
    sample_cat_id = 1000000055
    if sample_cat_id in mapper.cat_to_l3:
        l3_idx = mapper.cat_to_l3[sample_cat_id]
        targets = mapper.get_training_targets([sample_cat_id])

        if targets["l3"][0] != l3_idx:
            raise AssertionError("HierarchyMapper target generation failed.")
        print(f"  Mapped Category {sample_cat_id} -> L3 Index {l3_idx}")
    else:
        print(
            "  Sample category not found in mapping (might be rare), skipping specific ID check."
        )

    print("HierarchyMapper validated.")

    # ==========================================
    # 3. FEATURE EXTRACTION
    # ==========================================
    print("\n[Step 2/6] Testing FeatureEngine (Extraction)...")
    print("  Extracting features for Train, Val, and Test splits (20 samples each)...")

    engine = FeatureEngine()
    # Force extraction (ignore cache) to verify logic
    engine.extract_features(load_cached_data=False)

    # Validation
    expected_shape = (Config.DEBUG_SAMPLES, Config.INPUT_DIM)  # (20, 3328)

    if not os.path.exists(Config.TRAIN_FEATURES):
        raise FileNotFoundError("Train features file not created.")

    train_feats = np.load(Config.TRAIN_FEATURES)
    if train_feats.shape != expected_shape:
        raise AssertionError(
            f"Train features shape mismatch. Expected {expected_shape}, got {train_feats.shape}"
        )

    print(f"  Generated Train Features: {train_feats.shape}")
    print("FeatureEngine validated.")

    # ==========================================
    # 4. DATASET LOADING
    # ==========================================
    print("\n[Step 3/6] Testing RamFeatureDataset...")

    # Load the extracted features
    train_ds = RamFeatureDataset(
        Config.TRAIN_FEATURES, Config.TRAIN_LABELS, mode="train"
    )
    val_ds = RamFeatureDataset(Config.VAL_FEATURES, Config.VAL_LABELS, mode="val")

    # Validation
    if len(train_ds) != Config.DEBUG_SAMPLES:
        raise AssertionError(
            f"Dataset length mismatch. Expected {Config.DEBUG_SAMPLES}, got {len(train_ds)}"
        )

    # Check item structure: (feature, l1, l2, l3)
    item = train_ds[0]
    if len(item) != 4:
        raise AssertionError(
            "Dataset __getitem__ did not return 4 elements (feat, l1, l2, l3)."
        )

    feat, l1, l2, l3 = item
    if feat.shape[0] != Config.INPUT_DIM:
        raise AssertionError(
            f"Feature vector dimension incorrect. Expected {Config.INPUT_DIM}, got {feat.shape[0]}"
        )

    print("RamFeatureDataset validated.")

    # ==========================================
    # 5. MODEL INITIALIZATION
    # ==========================================
    print("\n[Step 4/6] Testing DeepFeatureCascade Model...")

    model = DeepFeatureCascade()

    # Test Forward Pass with dummy data
    dummy_input = torch.randn(2, Config.INPUT_DIM)
    l1_logits, l2_logits, l3_logits = model(dummy_input)

    # Validate output shapes
    if l1_logits.shape != (2, Config.NUM_CLASSES_L1):
        raise AssertionError(
            f"L1 logits shape mismatch. Expected (2, {Config.NUM_CLASSES_L1}), got {l1_logits.shape}"
        )
    if l3_logits.shape != (2, Config.NUM_CLASSES_L3):
        raise AssertionError(
            f"L3 logits shape mismatch. Expected (2, {Config.NUM_CLASSES_L3}), got {l3_logits.shape}"
        )

    print("Model architecture and forward pass validated.")

    # ==========================================
    # 6. TRAINING LOOP
    # ==========================================
    print("\n[Step 5/6] Testing ModelTrainer...")

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=Config.TRAIN_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=Config.TRAIN_BATCH_SIZE, shuffle=False)

    # Initialize Trainer
    trainer = ModelTrainer(model, train_loader, val_loader)

    # Run Training (1 Epoch)
    trainer.train()

    # Validate Checkpoint
    if not os.path.exists(Config.MODEL_CHECKPOINT):
        raise FileNotFoundError(
            f"Model checkpoint was not saved to {Config.MODEL_CHECKPOINT}"
        )

    print("ModelTrainer validated.")

    # ==========================================
    # 7. INFERENCE
    # ==========================================
    print("\n[Step 6/6] Testing EnsemblePredictor (Inference)...")

    # Initialize Predictor with the trained model
    predictor = EnsemblePredictor(model_paths=[Config.MODEL_CHECKPOINT])

    # Generate Submission
    predictor.generate_submission(
        output_path=Config.SUBMISSION_PATH, batch_size=Config.TRAIN_BATCH_SIZE
    )

    # Validate Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check dimensions
    if len(sub_df) != Config.DEBUG_SAMPLES:
        raise AssertionError(
            f"Submission rows mismatch. Expected {Config.DEBUG_SAMPLES}, got {len(sub_df)}"
        )

    # Check columns
    if "_id" not in sub_df.columns or "category_id" not in sub_df.columns:
        raise AssertionError(
            "Submission file missing required columns (_id, category_id)."
        )

    print(f"  Submission Head:\n{sub_df.head(3)}")
    print("EnsemblePredictor validated.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
