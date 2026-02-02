import os
import shutil
import numpy as np
import torch
import pandas as pd
from library.config import Config
from library.utils import seed_everything, HierarchyMapper
from library.feature_engine import FeatureEngine
from library.dataset import create_dataloader
from library.model import ProjectedMultiTaskMLP
from library.core import CombinedLoss, train_one_epoch, evaluate


def main():
    # 1. Setup and Configuration Overrides for Demo
    print("Setting up demonstration configuration...")
    seed_everything(42)

    # Create a demo working directory to avoid interfering with existing files
    demo_dir = os.path.join(Config.WORKING_DIR, "demo_run")
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config paths to use the demo directory
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_FEATURES = os.path.join(demo_dir, "train_features.npy")
    Config.TRAIN_LABELS = os.path.join(demo_dir, "train_labels.npy")
    Config.VAL_FEATURES = os.path.join(demo_dir, "val_features.npy")
    Config.VAL_LABELS = os.path.join(demo_dir, "val_labels.npy")
    Config.TEST_FEATURES = os.path.join(demo_dir, "test_features.npy")
    Config.TEST_IDS = os.path.join(demo_dir, "test_ids.npy")
    Config.HIERARCHY_MAPPING = os.path.join(demo_dir, "hierarchy_mapping.parquet")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Override Config parameters for speed and debug execution
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 50  # Process only 50 samples for demonstration
    Config.TRAIN_BATCH_SIZE = 16
    Config.EXTRACT_BATCH_SIZE = 16
    Config.EPOCHS = 1
    Config.NUM_MODELS = 1
    Config.NUM_WORKERS = 2

    print(f"Demo directory: {demo_dir}")
    print(f"Debug mode: {Config.DEBUG}")

    # 2. Hierarchy Mapper Demonstration
    print("\n--- Demonstrating HierarchyMapper ---")
    mapper = HierarchyMapper(Config.CATEGORY_NAMES)
    # Process and cache the hierarchy mapping
    df_mapping = mapper.process(load_cached=False)

    print(f"Hierarchy mapping created. Shape: {df_mapping.shape}")

    # Validation of mapping structure
    assert "l1_idx" in df_mapping.columns
    assert "l2_idx" in df_mapping.columns
    assert "l3_idx" in df_mapping.columns

    # Check a specific mapping consistency (Round-trip check)
    if len(df_mapping) > 0:
        sample_l3 = df_mapping["l3_idx"].iloc[0]
        sample_raw = df_mapping["category_id"].iloc[0]

        retrieved_raw = mapper.get_raw_category_id(sample_l3)
        assert retrieved_raw == sample_raw, "Mapping L3 -> Raw failed"

        l1, l2, l3 = mapper.get_hierarchical_labels(sample_raw)
        assert l3 == sample_l3, "Mapping Raw -> L3 failed"
        print("HierarchyMapper validation passed.")

    # 3. Feature Extraction Demonstration
    print("\n--- Demonstrating FeatureEngine ---")
    # This will extract features for the first 50 samples of train, val, and test
    # It uses the real BSON files but limits the count via Config.DEBUG_SAMPLES
    engine = FeatureEngine()
    engine.generate_features(load_cached_data=False)

    # Verify files exist
    assert os.path.exists(Config.TRAIN_FEATURES), "Train features not saved"
    assert os.path.exists(Config.TRAIN_LABELS), "Train labels not saved"

    # Load and verify shapes
    train_feats = np.load(Config.TRAIN_FEATURES)
    train_lbls = np.load(Config.TRAIN_LABELS)
    print(f"Extracted Train Features Shape: {train_feats.shape}")
    print(f"Extracted Train Labels Shape: {train_lbls.shape}")

    # Check dimensions match Config
    assert train_feats.shape == (Config.DEBUG_SAMPLES, Config.INPUT_DIM)
    assert train_lbls.shape == (Config.DEBUG_SAMPLES,)
    print("FeatureEngine validation passed.")

    # 4. Dataset and DataLoader Demonstration
    print("\n--- Demonstrating Dataset and DataLoader ---")
    train_loader = create_dataloader(
        Config.TRAIN_FEATURES,
        Config.TRAIN_LABELS,
        mapper,
        mode="train",
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Use main thread for stability in demo
    )

    # Fetch one batch to verify structure
    features, targets = next(iter(train_loader))
    target_l1, target_l2, target_l3 = targets

    print(f"Batch Features Shape: {features.shape}")
    print(f"Batch Targets L3 Shape: {target_l3.shape}")

    # Batch size might be smaller if it's the last batch, but here we check against expected max
    assert features.shape[0] <= Config.TRAIN_BATCH_SIZE
    assert features.shape[1] == Config.INPUT_DIM
    assert len(targets) == 3  # Should contain L1, L2, L3 targets
    print("DataLoader validation passed.")

    # 5. Model Demonstration
    print("\n--- Demonstrating ProjectedMultiTaskMLP ---")
    device = torch.device("cpu")  # Use CPU for simple demo verification
    model = ProjectedMultiTaskMLP().to(device)

    # Forward pass with the batch fetched earlier
    logits_l1, logits_l2, logits_l3 = model(features.to(device))

    print(f"Logits L1 Shape: {logits_l1.shape}")
    print(f"Logits L2 Shape: {logits_l2.shape}")
    print(f"Logits L3 Shape: {logits_l3.shape}")

    assert logits_l1.shape == (features.shape[0], Config.NUM_CLASSES_L1)
    assert logits_l2.shape == (features.shape[0], Config.NUM_CLASSES_L2)
    assert logits_l3.shape == (features.shape[0], Config.NUM_CLASSES_L3)
    print("Model forward pass validation passed.")

    # 6. Training Loop Demonstration
    print("\n--- Demonstrating Training Loop ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = CombinedLoss(label_smoothing=0.1)

    # Train for 1 epoch
    print("Training for 1 epoch...")
    loss = train_one_epoch(model, train_loader, optimizer, criterion, device, alpha=0.2)
    print(f"Epoch Loss: {loss:.4f}")
    assert not np.isnan(loss), "Training loss is NaN"

    # Evaluate (Using train_loader as validation for demo simplicity)
    print("Evaluating...")
    val_loss, val_acc = evaluate(model, train_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy (L3): {val_acc:.4f}")
    assert 0.0 <= val_acc <= 1.0, "Accuracy out of bounds"
    print("Training loop validation passed.")

    # 7. Inference Demonstration
    print("\n--- Demonstrating Inference ---")
    test_loader = create_dataloader(
        Config.TEST_FEATURES,
        Config.TEST_IDS,
        mapper,
        mode="test",
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for features, ids in test_loader:
            features = features.to(device)
            _, _, logits_l3 = model(features)

            # Get predictions
            preds_idx = torch.argmax(logits_l3, dim=1).cpu().numpy()

            # Map back to raw ID using the mapper
            raw_preds = [mapper.get_raw_category_id(idx) for idx in preds_idx]

            all_preds.extend(raw_preds)
            all_ids.extend(ids.numpy())

    print(f"Generated {len(all_preds)} predictions.")
    assert len(all_preds) == Config.DEBUG_SAMPLES

    # Create submission dataframe
    df_sub = pd.DataFrame({"_id": all_ids, "category_id": all_preds})
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
