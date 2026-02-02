import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil
import time

# Ensure the current directory is in the path for library imports
sys.path.append(os.getcwd())

from library.config import Config, set_seed
from library.utils import HierarchyMapper, BSONImageLoader
from library.feature_extract import FeatureExtractor
from library.dataset import FeatureDataset, TestFeatureDataset
from library.model import DualStreamMultiTaskNetwork
from library.trainer import Trainer


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup & Configuration Override
    # We modify Config attributes at runtime to create a fast debug environment
    print("\n[Step 1] Configuring Environment...")
    set_seed(42)

    Config.DEBUG_SIZE = 200  # Process only 200 samples
    Config.EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 32  # Smaller batch size for debug
    Config.NUM_WORKERS = 2  # Reduce workers to avoid overhead on small data

    # Ensure clean working directory for this run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Size: {Config.DEBUG_SIZE}")

    # 2. Verify Hierarchy Mapper
    print("\n[Step 2] Verifying HierarchyMapper...")
    mapper = HierarchyMapper(load_cached_data=False)  # Force compute from CSV

    # Validation: Check consistency
    assert mapper.num_classes_l3 == len(
        mapper.l3_id_to_idx
    ), "Mismatch in L3 classes count"
    assert mapper.num_classes_l1 > 0, "No Level 1 classes found"

    # Test mapping a specific category
    # Pick the first category from the dataframe used inside mapper (re-reading for validation)
    df_cats = pd.read_csv(Config.CATEGORY_NAMES)
    sample_cat_id = df_cats.iloc[0]["category_id"]
    sample_l1_name = df_cats.iloc[0]["category_level1"]

    l3_idx = mapper.l3_id_to_idx[sample_cat_id]
    l1_idx, l2_idx = mapper.get_parent_labels(np.array([l3_idx]))

    print(f"Category ID: {sample_cat_id} -> L3 Index: {l3_idx}")
    print(f"Mapped Parents -> L1 Index: {l1_idx[0]}, L2 Index: {l2_idx[0]}")

    assert l3_idx >= 0
    assert l1_idx[0] < mapper.num_classes_l1
    print("HierarchyMapper verification passed.")

    # 3. Verify BSON Image Loading
    print("\n[Step 3] Verifying BSONImageLoader...")
    train_meta = pd.read_csv(Config.TRAIN_META)
    sample_row = train_meta.iloc[0]

    loader = BSONImageLoader(Config.TRAIN_BSON)
    images = loader.read_images(sample_row["bson_offset"], sample_row["bson_length"])
    loader.close()

    assert len(images) > 0, "Failed to extract images from BSON"
    img_shape = images[0].shape
    print(f"Extracted {len(images)} images. Shape of first image: {img_shape}")

    assert img_shape == (
        Config.IMG_SIZE,
        Config.IMG_SIZE,
        3,
    ), f"Incorrect image shape: {img_shape}"
    print("BSONImageLoader verification passed.")

    # 4. Feature Extraction
    print("\n[Step 4] Running Feature Extraction (Debug Mode)...")
    extractor = FeatureExtractor(debug_size=Config.DEBUG_SIZE)

    # This will generate .npy files in Config.WORKING_DIR
    extractor.extract_all(load_cached_data=False)

    # Verify files exist
    expected_files = [
        Config.TRAIN_FEATS_RESNET,
        Config.TRAIN_FEATS_EFFNET,
        Config.TRAIN_LABELS,
        Config.VAL_FEATS_RESNET,
        Config.VAL_FEATS_EFFNET,
        Config.VAL_LABELS,
        Config.TEST_FEATS_RESNET,
        Config.TEST_FEATS_EFFNET,
        Config.TEST_IDS,
    ]

    for f in expected_files:
        assert os.path.exists(f), f"Missing expected feature file: {f}"

    # Verify shapes
    train_resnet = np.load(Config.TRAIN_FEATS_RESNET)
    print(f"Train ResNet Features Shape: {train_resnet.shape}")
    assert train_resnet.shape == (Config.DEBUG_SIZE, 2048), "Incorrect feature shape"
    print("Feature Extraction verification passed.")

    # 5. Dataset Verification
    print("\n[Step 5] Verifying Dataset Classes...")
    train_ds = FeatureDataset(
        Config.TRAIN_FEATS_RESNET, Config.TRAIN_FEATS_EFFNET, Config.TRAIN_LABELS
    )

    assert len(train_ds) == Config.DEBUG_SIZE
    sample_item = train_ds[0]

    assert "resnet_feat" in sample_item
    assert "effnet_feat" in sample_item
    assert "label_l3" in sample_item

    print(f"Dataset Item Keys: {list(sample_item.keys())}")
    print(f"Label L3: {sample_item['label_l3']}")
    print("FeatureDataset verification passed.")

    # 6. Model Architecture Verification
    print("\n[Step 6] Verifying Model Architecture...")
    model = DualStreamMultiTaskNetwork().to(Config.DEVICE)

    # Create dummy batch
    dummy_resnet = torch.randn(4, Config.RESNET_DIM).to(Config.DEVICE)
    dummy_effnet = torch.randn(4, Config.EFFNET_DIM).to(Config.DEVICE)

    l1_out, l2_out, l3_out = model(dummy_resnet, dummy_effnet)

    print(
        f"Model Output Shapes: L1={l1_out.shape}, L2={l2_out.shape}, L3={l3_out.shape}"
    )
    assert l1_out.shape == (4, Config.NUM_CLASSES_L1)
    assert l2_out.shape == (4, Config.NUM_CLASSES_L2)
    assert l3_out.shape == (4, Config.NUM_CLASSES_L3)
    print("Model architecture verification passed.")

    # 7. Training Loop Verification
    print("\n[Step 7] Verifying Training Loop...")

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True
    )

    val_ds = FeatureDataset(
        Config.VAL_FEATS_RESNET, Config.VAL_FEATS_EFFNET, Config.VAL_LABELS
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    trainer = Trainer(model, Config.DEVICE, train_loader, val_loader)

    print("Starting short training run...")
    start_time = time.time()
    best_acc = trainer.fit()
    duration = time.time() - start_time

    print(f"Training completed in {duration:.2f} seconds.")
    print(f"Best Validation Accuracy: {best_acc}")

    # Basic check that training happened (accuracy is a float between 0 and 1)
    assert 0.0 <= best_acc <= 1.0
    print("Training loop verification passed.")

    # 8. Inference Verification
    print("\n[Step 8] Verifying Inference...")
    test_ds = TestFeatureDataset(
        Config.TEST_FEATS_RESNET, Config.TEST_FEATS_EFFNET, Config.TEST_IDS
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            r_feat = batch["resnet_feat"].to(Config.DEVICE)
            e_feat = batch["effnet_feat"].to(Config.DEVICE)
            _, _, logits_l3 = model(r_feat, e_feat)
            preds = torch.argmax(logits_l3, dim=1).cpu().numpy()
            all_preds.extend(preds)

    assert len(all_preds) == Config.DEBUG_SIZE
    print(f"Generated {len(all_preds)} predictions.")
    print("Inference verification passed.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
