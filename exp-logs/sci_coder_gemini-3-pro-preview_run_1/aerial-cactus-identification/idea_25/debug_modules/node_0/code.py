import os
import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, calculate_roc_auc, FileSizeScaler, get_logger
from library.dataset import get_dataloaders, get_test_dataloader
from library.model import MultiTaskRepVGG
from library.engine import Engine


def main():
    print("=== Starting Demonstration of Cactus Classifier Library ===")

    # ---------------------------------------------------------
    # 1. Configuration Setup for Fast Demonstration
    # ---------------------------------------------------------
    print("\n[Step 1] Configuring environment for speed...")

    # Modify Config attributes for a quick run
    Config.DEBUG = True
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 2
    Config.CONVERGENCE_EPOCHS = 1
    Config.SWA_EPOCHS = 1
    Config.TOTAL_EPOCHS = Config.CONVERGENCE_EPOCHS + Config.SWA_EPOCHS
    Config.SWA_START_EPOCH = Config.CONVERGENCE_EPOCHS
    Config.SWA_CYCLE_LEN = 1

    # Use a specific working directory for this demo
    Config.IDEA_ID = "demo_run_script"
    Config.WORKING_DIR = f"./working/{Config.IDEA_ID}"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")

    # Re-create directories since we changed the path
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    Config.print_config()
    seed_everything(Config.SEED)

    # ---------------------------------------------------------
    # 2. Verify Utility Functions
    # ---------------------------------------------------------
    print("\n[Step 2] Verifying Utilities...")

    # Test FileSizeScaler
    raw_size = 1024.0  # 1KB
    norm_size = FileSizeScaler.transform([raw_size])[0]
    rec_size = FileSizeScaler.inverse_transform([norm_size])[0]

    print(f"  FileSizeScaler: {raw_size} -> {norm_size:.4f} -> {rec_size:.4f}")
    assert 0.0 <= norm_size <= 1.0, "Normalized size out of bounds"
    assert np.isclose(raw_size, rec_size, rtol=1e-5), "Inverse transform failed"

    # Test ROC AUC
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    auc = calculate_roc_auc(y_true, y_pred)
    print(f"  ROC AUC Calculation: {auc:.4f}")
    assert 0.0 <= auc <= 1.0, "AUC score out of bounds"

    # ---------------------------------------------------------
    # 3. Verify Dataset and DataLoader
    # ---------------------------------------------------------
    print("\n[Step 3] Verifying Data Loading...")

    # Initialize DataLoaders (DEBUG mode is on, so this will be fast)
    train_loader, val_loader = get_dataloaders(load_cached_data=True)

    print(f"  Train Batches: {len(train_loader)}")
    print(f"  Val Batches: {len(val_loader)}")

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    images = batch["image"]
    labels = batch["label"]
    quality = batch["quality_target"]

    print(f"  Batch Keys: {batch.keys()}")
    print(f"  Image Shape: {images.shape}")
    print(f"  Label Shape: {labels.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect image shape"
    assert labels.shape[0] == Config.BATCH_SIZE, "Incorrect label batch size"
    assert quality.shape[0] == Config.BATCH_SIZE, "Incorrect quality batch size"
    assert images.dtype == torch.float32, "Images should be float32"

    # ---------------------------------------------------------
    # 4. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[Step 4] Verifying Model Architecture...")

    device = Config.DEVICE
    model = MultiTaskRepVGG(deploy=False).to(device)

    # Forward pass with dummy data
    dummy_input = torch.randn(2, 3, 32, 32).to(device)
    outputs = model(dummy_input)

    print("  Forward pass outputs:")
    for key, val in outputs.items():
        if val is not None:
            print(f"    {key}: {val.shape}")
            # Texture and Semantic heads are (B, 1), Quality is (B, 1)
            assert val.shape == (2, 1), f"Output shape mismatch for {key}"

    # Test Reparameterization
    print("  Testing reparameterization...")
    # Check if a block has training attributes
    block = model.stage1[0]
    assert hasattr(block, "rbr_dense"), "Model should have rbr_dense before reparam"

    model.reparameterize()

    # Check if attributes are removed/fused
    block = model.stage1[0]
    assert not hasattr(block, "rbr_dense"), "rbr_dense should be removed after reparam"
    assert hasattr(block, "rbr_reparam"), "rbr_reparam should exist after reparam"
    print("  Reparameterization successful.")

    # ---------------------------------------------------------
    # 5. Verify Engine and Training Loop
    # ---------------------------------------------------------
    print("\n[Step 5] Verifying Engine and Training Loop...")

    # Re-initialize model for training (since we reparameterized the previous one)
    model = MultiTaskRepVGG(deploy=False).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Initialize Engine
    engine = Engine(model, device, optimizer=optimizer)

    # Run Fit (Training + Validation + SWA)
    # This runs for Config.TOTAL_EPOCHS (set to 2 for this demo)
    print("  Starting Engine.fit()...")
    best_auc = engine.fit(train_loader, val_loader, fold_idx=0)

    print(f"  Training complete. Best AUC: {best_auc:.4f}")

    # Check if checkpoints were created
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_fold0.pth")
    swa_model_path = os.path.join(Config.CHECKPOINT_DIR, "swa_fold0.pth")

    assert os.path.exists(best_model_path), "Best model checkpoint not found"
    assert os.path.exists(swa_model_path), "SWA model checkpoint not found"
    print("  Checkpoints verified.")

    # ---------------------------------------------------------
    # 6. Verify Inference
    # ---------------------------------------------------------
    print("\n[Step 6] Verifying Inference...")

    # Get test loader
    test_loader, test_ids = get_test_dataloader(load_cached_data=True)

    # Load best model weights
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Predict
    preds = engine.predict(test_loader)

    print(f"  Predictions shape: {preds.shape}")
    print(f"  First 5 predictions: {preds[:5]}")

    assert len(preds) == len(
        test_ids
    ), "Number of predictions does not match number of test IDs"
    assert np.all(
        (preds >= 0.0) & (preds <= 1.0)
    ), "Predictions out of probability range [0, 1]"

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
