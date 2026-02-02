import os
import sys
import torch
import pandas as pd
import warnings
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import get_model
from library.engine import train_one_epoch, validate
from library.inference import generate_submission


def run_demo():
    print("Starting iWildCam Pipeline Demo...")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Override Config for a fast demo run
    print("\n[1] Configuring environment...")

    # Use a separate working directory for the demo
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update paths to use the demo directory
    Config.CACHED_BBOXES_PATH = os.path.join(
        Config.WORKING_DIR, "megadetector_boxes.parquet"
    )
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Reduce computational load
    Config.BATCH_SIZE = 8
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 2  # Reduce workers to avoid overhead on small data

    # Set seed
    seed_everything(Config.SEED)
    device = get_device()
    print(f"    Device: {device}")
    print(f"    Working Dir: {Config.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\n[2] Loading Data (Subset)...")

    # Load only 50 samples for train/val/test to ensure speed
    SAMPLE_SIZE = 50

    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, sample_size=SAMPLE_SIZE
    )

    # Verification: Check Loader Lengths
    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches: {len(val_loader)}")

    # Verification: Check Batch Structure (Mixup Collator)
    images, targets = next(iter(train_loader))

    # Expected: [Batch_Size, 3, 448, 448]
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Image shape mismatch: {images.shape}"

    # Expected: [Batch_Size, Num_Classes] (Soft targets from Mixup)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Target shape mismatch: {targets.shape}"

    print("    Data integrity check passed.")

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    print("\n[3] Initializing Model...")

    # We use pretrained=False to avoid downloading weights during this short demo,
    # assuming we just want to verify pipeline mechanics.
    # In a real run, pretrained=True is preferred.
    model = get_model(pretrained=True)

    # Verification: Check Parameter Count > 0
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0, "Model has no parameters!"
    print(f"    Model loaded. Parameters: {param_count:,}")

    # --------------------------------------------------------------------------
    # 4. Training Loop (1 Epoch)
    # --------------------------------------------------------------------------
    print("\n[4] Running Training Step...")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    train_metrics = train_one_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        device=device,
        scaler=scaler,
        epoch=1,
    )

    # Verification: Loss should be a valid float
    assert isinstance(train_metrics["loss"], float), "Training loss is not a float"
    assert train_metrics["loss"] > 0, "Training loss should be positive"
    print(f"    Training finished. Loss: {train_metrics['loss']:.4f}")

    # --------------------------------------------------------------------------
    # 5. Validation Loop
    # --------------------------------------------------------------------------
    print("\n[5] Running Validation Step...")

    val_metrics = validate(model=model, dataloader=val_loader, device=device)

    # Verification: Accuracy between 0 and 1
    assert 0.0 <= val_metrics["accuracy"] <= 1.0, "Validation accuracy out of bounds"
    print(f"    Validation finished. Accuracy: {val_metrics['accuracy']:.4f}")

    # --------------------------------------------------------------------------
    # 6. Inference & Submission
    # --------------------------------------------------------------------------
    print("\n[6] Generating Submission...")

    generate_submission(
        model=model,
        dataloader=test_loader,
        device=device,
        save_path=Config.SUBMISSION_PATH,
    )

    # Verification: File Existence and Format
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission loaded. Shape: {df_sub.shape}")

    # Check columns
    expected_cols = ["Id", "Category"]
    assert all(
        col in df_sub.columns for col in expected_cols
    ), f"Missing columns. Expected {expected_cols}, got {df_sub.columns.tolist()}"

    # Check row count (should match sample size, or close to it depending on batch drop_last settings in loader)
    # The test loader does not drop last, so it should match SAMPLE_SIZE exactly.
    assert (
        len(df_sub) == SAMPLE_SIZE
    ), f"Submission row count mismatch. Expected {SAMPLE_SIZE}, got {len(df_sub)}"

    print("\n[SUCCESS] Pipeline demo completed successfully.")


if __name__ == "__main__":
    run_demo()
