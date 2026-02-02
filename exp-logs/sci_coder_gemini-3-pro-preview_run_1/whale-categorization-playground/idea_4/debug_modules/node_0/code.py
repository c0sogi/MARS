import sys
import os
import torch
import pandas as pd
import numpy as np
import shutil

# Ensure the current directory is in the path to import the library modules
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, save_checkpoint, load_checkpoint
from library.dataset import get_dataloaders
from library.model import WhaleConvNeXt
from library.train import train_one_epoch
from library.evaluate import validate, inference


def run_demo():
    print("=== Starting Whale Species Prediction Demo ===")

    # ---------------------------------------------------------
    # 1. Configure for Speed/Demo
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Override Config defaults for a quick run
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 16  # Use only 16 images
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo

    # Change working directory to avoid overwriting main experiment files
    Config.WORKING_DIR = "./working/demo_execution"
    Config.setup()  # Create the directory

    seed_everything(Config.SEED)
    print(
        f"Configuration updated: DEBUG={Config.DEBUG}, BATCH_SIZE={Config.BATCH_SIZE}"
    )

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\n[2] Initializing DataLoaders...")

    # load_cached_data=False forces re-computation of class mapping for this demo run
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        load_cached_data=False
    )

    print(f"Number of Classes: {len(classes)}")
    print(f"Train Batches: {len(train_loader)}")

    # Verify Data Shapes
    images, labels = next(iter(train_loader))

    # Expected shape: (Batch, 3, H, W)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch. Expected {(Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)}, got {images.shape}"

    # Expected shape: (Batch,)
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Label shape mismatch. Expected {(Config.BATCH_SIZE,)}, got {labels.shape}"

    print("DataLoader verification passed.")

    # ---------------------------------------------------------
    # 3. Model Initialization & Forward Pass Checks
    # ---------------------------------------------------------
    print("\n[3] Initializing Model (ConvNeXt + ArcFace)...")
    device = Config.DEVICE
    model = WhaleConvNeXt().to(device)

    # Check 1: Inference Mode (No labels) -> Returns scaled cosine similarities
    with torch.no_grad():
        logits_inference = model(images.to(device), labels=None)

    assert logits_inference.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Inference logits shape mismatch. Got {logits_inference.shape}"

    # Check 2: Training Mode (With labels) -> Returns ArcFace margin logits
    logits_train = model(images.to(device), labels=labels.to(device))

    assert logits_train.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Training logits shape mismatch. Got {logits_train.shape}"

    print("Model forward pass verified.")

    # ---------------------------------------------------------
    # 4. Training Loop Demo
    # ---------------------------------------------------------
    print("\n[4] Running Training Step (1 Epoch)...")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)

    # Run training for one epoch on the subset
    avg_loss = train_one_epoch(
        train_loader, model, criterion, optimizer, device, epoch=1
    )

    print(f"Training finished. Average Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Training loss resulted in NaN."

    # ---------------------------------------------------------
    # 5. Validation Demo
    # ---------------------------------------------------------
    print("\n[5] Running Validation Step...")
    val_loss, val_map5 = validate(val_loader, model, criterion, device, classes)

    print(f"Validation finished. Loss: {val_loss:.4f}, MAP@5: {val_map5:.4f}")
    assert 0.0 <= val_map5 <= 1.0, "MAP@5 score is out of valid range [0, 1]."

    # ---------------------------------------------------------
    # 6. Checkpointing
    # ---------------------------------------------------------
    print("\n[6] Testing Checkpoint Save/Load...")
    checkpoint_filename = "model_best.pth"

    state = {
        "epoch": 1,
        "state_dict": model.state_dict(),
        "best_map5": val_map5,
        "optimizer": optimizer.state_dict(),
    }

    # Save
    save_checkpoint(state, is_best=True, filename=checkpoint_filename)

    # Load
    model_new = WhaleConvNeXt().to(device)
    loaded_checkpoint = load_checkpoint(
        model_new, filename=checkpoint_filename, device=device
    )

    assert loaded_checkpoint is not None, "Failed to load checkpoint."
    assert loaded_checkpoint["epoch"] == 1, "Loaded checkpoint epoch mismatch."
    print("Checkpoint save/load verified.")

    # ---------------------------------------------------------
    # 7. Inference / Submission
    # ---------------------------------------------------------
    print("\n[7] Generating Submission...")

    # Run inference using the loaded model
    inference(test_loader, model_new, device, classes)

    submission_file = "./submission/submission.csv"
    assert os.path.exists(submission_file), "Submission file was not created."

    df_sub = pd.read_csv(submission_file)
    print(f"Submission file created successfully with {len(df_sub)} rows.")

    # Verify format
    assert (
        "Image" in df_sub.columns and "Id" in df_sub.columns
    ), "Submission columns missing."
    print("Submission format verified.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
