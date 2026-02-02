import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import build_model
from library.train import train_one_epoch, validate_one_epoch
from library.inference import predict_and_submit


def main():
    print("Starting WSIL Pipeline Demonstration...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[Step 1] Configuring environment for rapid demonstration...")

    # Override Config for speed and isolation
    Config.MAX_SAMPLES = 10  # Process only 10 subjects per split
    Config.NUM_EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 2  # Reduce workers for small data

    # Redirect outputs to working directory
    Config.CACHE_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = "./working/demo_run/demo_submission.csv"

    # Update cache paths based on new CACHE_DIR
    Config.CACHE_TRAIN_IMAGES = os.path.join(Config.CACHE_DIR, "train_images.npy")
    Config.CACHE_TRAIN_LABELS = os.path.join(Config.CACHE_DIR, "train_labels.npy")
    Config.CACHE_TRAIN_IDS = os.path.join(Config.CACHE_DIR, "train_ids.npy")
    Config.CACHE_VAL_IMAGES = os.path.join(Config.CACHE_DIR, "val_images.npy")
    Config.CACHE_VAL_LABELS = os.path.join(Config.CACHE_DIR, "val_labels.npy")
    Config.CACHE_VAL_IDS = os.path.join(Config.CACHE_DIR, "val_ids.npy")
    Config.CACHE_TEST_IMAGES = os.path.join(Config.CACHE_DIR, "test_images.npy")
    Config.CACHE_TEST_IDS = os.path.join(Config.CACHE_DIR, "test_ids.npy")

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Configuration complete. Device: {device}")

    # ==========================================
    # 2. Data Loading & Verification
    # ==========================================
    print("\n[Step 2] Loading and processing data...")

    # Force reload to demonstrate processing logic (load_cached_data=False)
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Verify Train Batch
    images, targets = next(iter(train_loader))
    print(f"Sample Batch Shape - Images: {images.shape}, Targets: {targets.shape}")

    # Assertions
    # Shape: (Batch_Size, Channels=3, Height=224, Width=224)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape: {images.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect target shape: {targets.shape}"
    assert images.dtype == torch.float32, "Images should be float32"

    print("Data loading verification passed.")

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\n[Step 3] Initializing Model...")

    model = build_model(device=device)

    # Verify Forward Pass
    images = images.to(device)
    with torch.no_grad():
        logits = model(images)

    print(f"Logits shape: {logits.shape}")

    # Assertions
    # Output should be (Batch_Size, 1)
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Incorrect output shape: {logits.shape}"

    print("Model initialization verification passed.")

    # ==========================================
    # 4. Training Loop Demonstration
    # ==========================================
    print("\n[Step 4] Running Training Loop...")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    best_val_auc = 0.0
    save_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            label_smoothing=Config.LABEL_SMOOTHING,
        )

        # Validate
        val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}"
        )

        # Basic sanity checks
        if np.isnan(train_loss) or np.isnan(val_loss):
            raise ValueError("Loss is NaN, training failed.")

        # Save best model logic
        if val_auc >= best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)

    # Ensure model was saved
    if not os.path.exists(save_path):
        # If AUC didn't improve (unlikely with init 0.0), force save for inference demo
        torch.save(model.state_dict(), save_path)

    print(f"Training complete. Best model saved to {save_path}")

    # ==========================================
    # 5. Inference & Submission
    # ==========================================
    print("\n[Step 5] Running Inference and Generating Submission...")

    # Run inference using the cached data we just generated
    predict_and_submit(load_cached_data=True)

    # Verify Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission File Head:")
    print(df_sub.head())

    # Assertions
    required_cols = ["BraTS21ID", "MGMT_value"]
    assert all(
        col in df_sub.columns for col in required_cols
    ), f"Submission missing columns. Found: {df_sub.columns}"

    # Check that we have predictions for the test IDs processed
    # Note: test_ids contains 3 instances per subject, so we need unique subjects
    unique_test_subjects = np.unique(test_ids)
    assert len(df_sub) == len(
        unique_test_subjects
    ), f"Mismatch in prediction count. Expected {len(unique_test_subjects)}, got {len(df_sub)}"

    # Check probability range
    assert (
        df_sub["MGMT_value"].min() >= 0.0 and df_sub["MGMT_value"].max() <= 1.0
    ), "Predictions out of probability range [0, 1]"

    print("\nInference verification passed.")
    print("Demonstration completed successfully.")


if __name__ == "__main__":
    main()
