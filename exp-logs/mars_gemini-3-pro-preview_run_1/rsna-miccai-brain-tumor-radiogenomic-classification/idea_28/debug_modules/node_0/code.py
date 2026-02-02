import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config, setup_system
from library.utils import (
    seed_everything,
    get_device,
    calculate_roc_auc,
    save_submission,
)
from library.model import SIRVEfficientNet
from library.data import get_dataloader, read_dicom
from library.train import train_one_epoch, validate_one_epoch


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("--- 1. Setting up System and Configuration ---")

    # Modify Config for a fast demonstration
    Config.DEBUG = True
    Config.DEBUG_DATASET_SIZE = 12  # Small subset for speed
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.N_FOLDS = 2

    # Use a specific cache directory for this demo to avoid conflicts
    Config.CACHE_DIR = "./working/demo_cache"
    Config.SUBMISSION_PATH = "./working/demo_submission.csv"

    setup_system(seed=42)
    device = get_device()
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Verify Utilities
    # ==========================================
    print("\n--- 2. Verifying Utilities ---")

    # Test ROC AUC calculation
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    auc = calculate_roc_auc(y_true, y_pred)
    print(f"Test AUC calculation: {auc}")
    assert 0.0 <= auc <= 1.0, "AUC score out of range"

    # ==========================================
    # 3. Verify Data Loading
    # ==========================================
    print("\n--- 3. Verifying Data Loading (Train) ---")

    # Initialize Train Loader (Fold 0)
    # This will trigger ROI cache generation for the debug subset
    train_loader = get_dataloader(
        split="train",
        fold_idx=0,
        batch_size=Config.BATCH_SIZE,
        num_workers=0,  # Use 0 workers for simple debugging to avoid multiprocessing overhead
        load_cached_data=False,  # Force regeneration for demo purposes
    )

    # Fetch one batch
    images, targets = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    # Assertions
    # Expected: (Batch, 9, 224, 224) -> 9 channels = 3 modalities * 3 depths
    assert images.shape == (
        Config.BATCH_SIZE,
        9,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Incorrect image shape. Expected {(Config.BATCH_SIZE, 9, Config.IMAGE_SIZE, Config.IMAGE_SIZE)}, got {images.shape}"
    assert targets.shape == (Config.BATCH_SIZE,), "Incorrect target shape"
    assert images.dtype == torch.float32, "Images should be float32"

    # ==========================================
    # 4. Verify Model Architecture
    # ==========================================
    print("\n--- 4. Verifying Model Architecture ---")

    model = SIRVEfficientNet(
        model_name=Config.MODEL_NAME,
        pretrained=False,  # False for speed in demo, True in real training
        num_classes=1,
        dropout_rate=Config.DROPOUT_RATE,
    )
    model = model.to(device)

    # Check if the first layer was correctly modified to accept 9 channels
    first_conv = model.backbone.conv_stem
    print(f"First Conv Layer: {first_conv}")
    assert first_conv.in_channels == 9, "Model input channels not modified to 9"

    # Check Forward Pass
    with torch.no_grad():
        images = images.to(device)
        logits = model(images)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (Config.BATCH_SIZE, 1), "Output shape mismatch"

    # ==========================================
    # 5. Verify Training Loop Components
    # ==========================================
    print("\n--- 5. Verifying Training Loop ---")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Train for one epoch (on the debug subset)
    print("Running Train Step...")
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}")
    assert train_loss > 0, "Train loss should be positive"

    # Validate for one epoch
    print("Running Validation Step...")
    val_loader = get_dataloader(
        split="val", fold_idx=0, batch_size=Config.BATCH_SIZE, num_workers=0
    )
    val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

    # ==========================================
    # 6. Verify Inference and Submission
    # ==========================================
    print("\n--- 6. Verifying Inference and Submission ---")

    test_loader = get_dataloader(
        split="test", batch_size=Config.BATCH_SIZE, num_workers=0
    )

    model.eval()
    all_ids = []
    all_preds = []

    # We need the BraTS21IDs corresponding to the test loader.
    # Since the loader shuffles=False for test, we can get IDs from the dataset dataframe.
    # Note: In a real scenario, the dataset or loader might not expose IDs directly in __getitem__
    # unless modified, but we can access the underlying dataframe.
    test_ids = test_loader.dataset.df["BraTS21ID"].values

    print(f"Inference on {len(test_ids)} test subjects...")

    with torch.no_grad():
        batch_idx = 0
        for images, _ in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            # Get IDs for this batch
            start = batch_idx * Config.BATCH_SIZE
            end = start + images.size(0)
            batch_ids = test_ids[start:end]

            all_ids.extend(batch_ids)
            all_preds.extend(probs)

            batch_idx += 1

    # Verify we have predictions for all loaded subjects
    assert len(all_ids) == len(test_loader.dataset), "Mismatch in prediction count"

    # Save Submission
    save_submission(all_ids, all_preds, Config.SUBMISSION_PATH)

    # Verify file creation
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file created with {len(df_sub)} rows.")
    print(df_sub.head())

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
