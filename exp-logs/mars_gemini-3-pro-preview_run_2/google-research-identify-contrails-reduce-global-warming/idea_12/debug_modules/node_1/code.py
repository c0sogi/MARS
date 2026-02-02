import os
import sys
import torch
import pandas as pd
import numpy as np
import time

# Import library modules
from library.config import Config
from library.utils import seed_everything, rle_encode
from library.dataset import get_train_val_loaders, get_test_loader
from library.model import ConvNeXtUNet
from library.loss import HybridLoss
from library.train import train_one_epoch, validate, train_loop
from library.inference import make_predictions


def run_demo():
    print("===========================================================")
    print("       Contrail Identification Pipeline Demonstration      ")
    print("===========================================================")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Modify Config parameters for speed and demo purposes
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.MAX_TRAIN_SAMPLES = 50  # Limit training data for speed
    Config.MAX_VAL_SAMPLES = 20  # Limit validation data for speed
    Config.NUM_WORKERS = 2  # Reduce overhead

    # Define demo-specific paths to avoid conflicts with main runs
    Config.WORKING_DIR = "./working/demo_run"
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_FILE_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print(f"    Device: {Config.DEVICE}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Dataset & DataLoader Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading...")
    train_loader, val_loader = get_train_val_loaders()

    # Fetch a single batch
    images, masks = next(iter(train_loader))

    # Verify Shapes
    print(f"    Batch Image Shape: {images.shape}")
    print(f"    Batch Mask Shape:  {masks.shape}")

    # Expected: (Batch, 6 Channels, 256, 256)
    assert images.shape == (
        Config.BATCH_SIZE,
        Config.IN_CHANNELS,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Expected image shape {(Config.BATCH_SIZE, Config.IN_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE)}, got {images.shape}"

    # Expected: (Batch, 1 Channel, 256, 256)
    assert masks.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Expected mask shape {(Config.BATCH_SIZE, 1, Config.IMG_SIZE, Config.IMG_SIZE)}, got {masks.shape}"

    # Verify Data Range (Normalized inputs should be approx 0-1)
    print(f"    Image Value Range: [{images.min():.4f}, {images.max():.4f}]")
    assert (
        images.min() >= 0.0 and images.max() <= 1.0
    ), "Image data is not properly normalized to [0, 1]."

    print("    -> Data Loading Verified.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")
    model = ConvNeXtUNet().to(Config.DEVICE)

    # Move batch to device
    images = images.to(Config.DEVICE)
    masks = masks.to(Config.DEVICE)

    # Forward Pass
    logits = model(images)
    print(f"    Logits Shape: {logits.shape}")

    assert (
        logits.shape == masks.shape
    ), "Model output shape does not match ground truth shape."
    print("    -> Model Forward Pass Verified.")

    # -------------------------------------------------------------------------
    # 4. Loss Function Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Loss Function...")
    criterion = HybridLoss()
    loss = criterion(logits, masks)

    print(f"    Calculated Loss: {loss.item():.6f}")
    assert not torch.isnan(loss), "Loss is NaN."
    assert loss.item() >= 0, "Loss is negative."
    print("    -> Loss Calculation Verified.")

    # -------------------------------------------------------------------------
    # 5. Training Step Verification
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Training Step (Optimizer & Backprop)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    # Capture state before update to verify learning
    head_weight_before = model.segmentation_head.weight.clone()

    # Run one training step
    epoch_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, Config.DEVICE, scaler
    )
    print(f"    One Epoch Loss: {epoch_loss:.6f}")

    # Check if weights changed
    head_weight_after = model.segmentation_head.weight
    assert not torch.equal(
        head_weight_before, head_weight_after
    ), "Model weights did not update after training step."
    print("    -> Training Step Verified.")

    # -------------------------------------------------------------------------
    # 6. Validation & Metric Verification
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Validation & Dice Metric...")
    val_loss, val_dice = validate(model, val_loader, criterion, Config.DEVICE)
    print(f"    Validation Loss: {val_loss:.6f}")
    print(f"    Validation Dice: {val_dice:.6f}")

    assert 0.0 <= val_dice <= 1.0, "Dice coefficient must be between 0 and 1."
    print("    -> Validation Logic Verified.")

    # -------------------------------------------------------------------------
    # 7. Full Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n[7] Executing Full Training Loop (1 Epoch)...")
    # We pass epochs explicitly because default args in train_loop might be bound to old Config values
    train_loop(epochs=1, patience=1)

    print("\n[8] Executing Inference Pipeline...")
    # Ensure the model file exists
    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not saved."

    # Run inference
    make_predictions(
        model_path=Config.BEST_MODEL_PATH, output_path=Config.SUBMISSION_FILE_PATH
    )

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_FILE_PATH), "Submission CSV not found."
    df = pd.read_csv(Config.SUBMISSION_FILE_PATH)
    print(f"    Submission Rows: {len(df)}")
    print(f"    Columns: {list(df.columns)}")

    # Check content format
    sample_rle = df.iloc[0]["encoded_pixels"]
    print(f"    Sample RLE: {sample_rle}")
    if sample_rle != "-":
        rle_parts = sample_rle.split()
        assert all(x.isdigit() for x in rle_parts), "RLE contains non-digit characters."

    print("    -> Full Pipeline Verified.")

    print("\n===========================================================")
    print("       Demo Completed Successfully")
    print("===========================================================")


if __name__ == "__main__":
    run_demo()
