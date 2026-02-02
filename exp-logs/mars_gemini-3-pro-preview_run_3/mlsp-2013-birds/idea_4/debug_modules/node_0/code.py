import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, save_checkpoint
from library.dataset import BirdDataset, get_dataloaders
from library.model import BirdResNet
from library.train import train_one_epoch, validate
from library.inference import predict_ensemble


def run_demonstration():
    print("==== Starting Bird Species Classification Demonstration ====")

    # 1. Setup and Configuration Overrides
    # We modify the global Config to run a fast, minimal demonstration.
    print("[Step 1] Configuring environment for fast demonstration...")
    set_seed(42)

    # Override Config for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_FOLDS = 1  # Only simulate one fold
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.DEBUG = True

    # Ensure directories exist (Config.setup() is called on import, but good to confirm)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Data Loading Demonstration
    print("\n[Step 2] Loading and slicing metadata...")
    # Load original metadata
    train_df_full = pd.read_csv(Config.TRAIN_CSV)
    val_df_full = pd.read_csv(Config.VAL_CSV)
    test_df_full = pd.read_csv(Config.TEST_CSV)

    # Slice data for demonstration (use small subset)
    train_df_demo = train_df_full.head(12).copy()
    val_df_demo = val_df_full.head(8).copy()
    test_df_demo = test_df_full.head(8).copy()

    print(f"  Train subset size: {len(train_df_demo)}")
    print(f"  Val subset size: {len(val_df_demo)}")
    print(f"  Test subset size: {len(test_df_demo)}")

    # Instantiate DataLoaders
    print("  Creating DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df_demo, val_df_demo, test_df_demo, load_cached_data=True
    )

    # Validate DataLoader output
    images, targets, rec_ids = next(iter(train_loader))
    print(f"  Batch shapes - Images: {images.shape}, Targets: {targets.shape}")

    # Assertions for data shapes
    # Expected: (Batch, 3, Freq, Time)
    # Freq = 128 (N_MELS), Time approx 501 (16000*10/320 + 1)
    assert images.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert images.shape[1] == 3, "Channel dimension mismatch (should be 3)"
    assert images.shape[2] == Config.N_MELS, "Frequency dimension mismatch"
    assert targets.shape[1] == Config.NUM_CLASSES, "Target class dimension mismatch"

    # 3. Model Instantiation
    print("\n[Step 3] Instantiating Model...")
    device = Config.DEVICE
    # Using pretrained=False for speed/offline safety in demo, though True is standard
    model = BirdResNet(pretrained=False, num_classes=Config.NUM_CLASSES)
    model.to(device)

    # Verify Forward Pass
    with torch.no_grad():
        demo_output = model(images.to(device))

    print(f"  Model output shape: {demo_output.shape}")
    assert demo_output.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"

    # 4. Training Loop Demonstration
    print("\n[Step 4] Running Training Step (1 Epoch)...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Run one epoch of training
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"  Train Loss: {train_loss:.4f}")
    assert train_loss > 0, "Training loss should be positive"

    # Run validation
    val_loss, val_auc = validate(model, val_loader, criterion, device)
    print(f"  Validation Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # 5. Checkpointing
    print("\n[Step 5] Saving Checkpoint...")
    # Simulate saving the best model for Fold 0
    checkpoint_name = "fold_0_best.pth"
    save_checkpoint(
        {"state_dict": model.state_dict(), "auc": val_auc, "epoch": 0}, checkpoint_name
    )
    expected_ckpt_path = os.path.join(Config.CHECKPOINT_DIR, checkpoint_name)
    assert os.path.exists(expected_ckpt_path), "Checkpoint file was not created"
    print(f"  Checkpoint saved to {expected_ckpt_path}")

    # 6. Inference Demonstration
    print("\n[Step 6] Running Inference...")
    # We need to ensure the Config points to the correct test metadata for the library function
    # Since predict_ensemble reads from Config.TEST_CSV, we temporarily overwrite the file
    # or just rely on the fact that we can't easily change the file path in Config without
    # editing the class. However, predict_ensemble loads the CSV from disk.
    # For this demo, we will execute the logic of predict_ensemble manually using our sliced test_df
    # to avoid processing the full test set, OR we just let it run on the full test set (64 samples is fast).
    # Let's run the provided library function `predict_ensemble` directly.
    # It uses Config.TEST_CSV (64 samples), which is small enough.

    predict_ensemble(load_cached_data=True)

    # Verify Submission
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file not found"

    sub_df = pd.read_csv(submission_path)
    print(f"  Submission generated with shape: {sub_df.shape}")

    # Expected rows: 64 test samples * 19 classes = 1216 rows
    expected_rows = 64 * Config.NUM_CLASSES
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Check columns
    assert (
        "Id" in sub_df.columns and "Probability" in sub_df.columns
    ), "Submission columns mismatch"

    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    run_demonstration()
