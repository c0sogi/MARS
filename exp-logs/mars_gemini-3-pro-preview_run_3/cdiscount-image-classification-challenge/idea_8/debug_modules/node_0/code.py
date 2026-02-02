import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import provided library components
from library.config import Config
from library.data_utils import HierarchyMapper
from library.feature_extractor import run_feature_extraction
from library.dataset import FeatureDataset, MixupCollate
from library.model import HierarchicalMultiTaskNetwork, HierarchicalTrainer
from library.engine import train_one_epoch, evaluate, generate_submission


def main():
    print("=== Starting Demo Execution ===")

    # ==========================================
    # 1. CONFIGURATION OVERRIDE
    # ==========================================
    # Modify Config for a fast demonstration run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200  # Small sample for speed
    Config.BATCH_SIZE = 32
    Config.EPOCHS = 2
    Config.WORKING_DIR = "./working/demo_execution"

    # Update paths in Config based on new WORKING_DIR
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    Config.TRAIN_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "train_features.npy")
    Config.TRAIN_LABELS_PATH = os.path.join(Config.WORKING_DIR, "train_labels.npy")
    Config.VAL_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "val_features.npy")
    Config.VAL_LABELS_PATH = os.path.join(Config.WORKING_DIR, "val_labels.npy")
    Config.TEST_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "test_features.npy")
    Config.TEST_IDS_PATH = os.path.join(Config.WORKING_DIR, "test_ids.npy")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.HIERARCHY_MAP_PATH = os.path.join(
        Config.WORKING_DIR, "hierarchy_map.parquet"
    )
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Setup (seeds, dirs)
    Config.setup()
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # ==========================================
    # 2. HIERARCHY MAPPING
    # ==========================================
    print("\n--- Step 2: Hierarchy Mapping ---")
    mapper = HierarchyMapper(load_cached_data=False)

    # Validation
    num_classes = mapper.get_num_classes()
    print(f"Class Counts: {num_classes}")
    assert num_classes["l1"] == Config.NUM_CLASSES_L1
    assert num_classes["l2"] == Config.NUM_CLASSES_L2
    assert num_classes["l3"] == Config.NUM_CLASSES_L3

    # Check a specific mapping (using first row of category_names.csv from description)
    # category_id: 1000021794 -> ABONNEMENT / SERVICES -> CARTE PREPAYEE -> CARTE PREPAYEE MULTIMEDIA
    # We just ensure it returns valid indices
    l1, l2, l3 = mapper.get_labels(1000021794)
    assert l1 >= 0 and l2 >= 0 and l3 >= 0
    print("HierarchyMapper validation passed.")

    # ==========================================
    # 3. FEATURE EXTRACTION
    # ==========================================
    print("\n--- Step 3: Feature Extraction ---")
    # This will use the BSONIterator and ResNet50 to process the first DEBUG_SAMPLE_SIZE images
    run_feature_extraction(load_cached_data=False)

    # Validate output files
    assert os.path.exists(Config.TRAIN_FEATURES_PATH)
    assert os.path.exists(Config.TRAIN_LABELS_PATH)
    assert os.path.exists(Config.TEST_FEATURES_PATH)

    # Validate Shapes
    train_feats = np.load(Config.TRAIN_FEATURES_PATH)
    train_labels = np.load(Config.TRAIN_LABELS_PATH)
    print(f"Train Features Shape: {train_feats.shape}")
    print(f"Train Labels Shape: {train_labels.shape}")

    # Feature dim should be 2048 (ResNet50)
    assert train_feats.shape[1] == 2048
    # Labels should be (N, 3) for L1, L2, L3
    assert train_labels.shape[1] == 3
    # Number of samples should match (or be close to, depending on filtering) the debug size
    # Note: It might be slightly less if some products have no images, but usually matches.
    assert train_feats.shape[0] > 0

    # ==========================================
    # 4. DATASET & DATALOADER
    # ==========================================
    print("\n--- Step 4: Dataset & DataLoader ---")

    # Instantiate Dataset
    train_ds = FeatureDataset(
        Config.TRAIN_FEATURES_PATH, Config.TRAIN_LABELS_PATH, load_in_memory=True
    )
    val_ds = FeatureDataset(
        Config.VAL_FEATURES_PATH, Config.VAL_LABELS_PATH, load_in_memory=True
    )

    # Instantiate Collate
    mixup_collate = MixupCollate(alpha=0.2)

    # Create Loader
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=mixup_collate
    )

    # Fetch one batch to verify MixUp
    features, y1, y2, y3 = next(iter(train_loader))

    print(f"Batch Features Shape: {features.shape}")
    print(f"Batch L1 Targets Shape: {y1.shape}")

    # Assertions
    assert features.shape[0] == Config.BATCH_SIZE or features.shape[0] == len(train_ds)
    assert features.shape[1] == 2048
    # Targets should be soft (float), not indices (long)
    assert y1.dtype == torch.float32
    assert y1.shape[1] == Config.NUM_CLASSES_L1

    print("Dataset and MixUp validation passed.")

    # ==========================================
    # 5. MODEL INITIALIZATION
    # ==========================================
    print("\n--- Step 5: Model Initialization ---")
    model = HierarchicalMultiTaskNetwork()
    model.to(Config.DEVICE)

    # Dummy Forward Pass
    dummy_input = torch.randn(4, 2048).to(Config.DEVICE)
    out_l1, out_l2, out_l3 = model(dummy_input)

    print(f"Output L1 Shape: {out_l1.shape}")
    assert out_l1.shape == (4, Config.NUM_CLASSES_L1)
    assert out_l2.shape == (4, Config.NUM_CLASSES_L2)
    assert out_l3.shape == (4, Config.NUM_CLASSES_L3)

    print("Model forward pass validation passed.")

    # ==========================================
    # 6. TRAINING LOOP
    # ==========================================
    print("\n--- Step 6: Training Loop ---")

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    trainer = HierarchicalTrainer(model, device=Config.DEVICE)

    # Run fit (Config.EPOCHS is set to 2)
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # Verify model checkpoint creation
    assert os.path.exists(Config.MODEL_SAVE_PATH)
    print("Training complete and model saved.")

    # ==========================================
    # 7. INFERENCE & SUBMISSION
    # ==========================================
    print("\n--- Step 7: Inference & Submission ---")

    # We use the trainer's predict_submission method which handles loading test data
    trainer.predict_submission(
        Config.TEST_FEATURES_PATH, Config.TEST_IDS_PATH, Config.SUBMISSION_PATH
    )

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH)
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    print(f"Submission shape: {sub_df.shape}")
    print(sub_df.head())

    assert "_id" in sub_df.columns
    assert "category_id" in sub_df.columns
    assert len(sub_df) > 0
    # Check types
    assert pd.api.types.is_integer_dtype(sub_df["_id"])
    assert pd.api.types.is_integer_dtype(sub_df["category_id"])

    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    main()
