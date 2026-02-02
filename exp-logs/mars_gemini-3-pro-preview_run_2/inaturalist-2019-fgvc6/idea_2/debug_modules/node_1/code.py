import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import classes and functions from the provided library files
from library.config import Config
from library.utils import set_seed, save_checkpoint
from library.dataset import INatDataset, get_transforms
from library.model import create_model
from library.engine import train_one_epoch, validate, inference


def run_demo():
    print("--- Starting Library Integration Demo ---")

    # ==========================================
    # 1. Configuration Setup
    # ==========================================
    print("\n[1] Configuring environment for rapid demonstration...")
    # Override Config values to ensure the script runs fast (Speed Optimization)
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 images
    Config.BATCH_SIZE = 8
    Config.NUM_EPOCHS = 1
    Config.PRETRAINED = False  # Skip downloading weights for speed
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Redirect output to a specific demo directory
    Config.WORKING_DIR = "./working/demo_run"
    Config.CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "checkpoint.pth")
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "model_best.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Create working directory
    if not os.path.exists(Config.WORKING_DIR):
        os.makedirs(Config.WORKING_DIR)

    # Set reproducibility seed
    set_seed(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Compute Device: {Config.DEVICE}")

    # ==========================================
    # 2. Dataset & DataLoader Instantiation
    # ==========================================
    print("\n[2] Initializing Datasets & DataLoaders...")

    # Train Dataset
    train_transform = get_transforms(mode="train", image_size=Config.IMAGE_SIZE)
    train_dataset = INatDataset(
        metadata_file=Config.TRAIN_METADATA,
        root_dir=Config.INPUT_DIR,
        transform=train_transform,
        mode="train",
        debug=Config.DEBUG,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Validation Dataset
    val_transform = get_transforms(mode="val", image_size=Config.IMAGE_SIZE)
    val_dataset = INatDataset(
        metadata_file=Config.VAL_METADATA,
        root_dir=Config.INPUT_DIR,
        transform=val_transform,
        mode="val",
        debug=Config.DEBUG,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train Subset Size: {len(train_dataset)}")
    print(f"Val Subset Size: {len(val_dataset)}")

    # Logic Verification: Check batch shapes
    dummy_images, dummy_targets = next(iter(train_loader))
    print(f"Batch Image Shape: {dummy_images.shape}")
    print(f"Batch Target Shape: {dummy_targets.shape}")

    assert dummy_images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect image tensor shape."
    assert dummy_targets.shape == (Config.BATCH_SIZE,), "Incorrect target tensor shape."

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\n[3] Initializing Model...")
    model = create_model(num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED)
    model.to(Config.DEVICE)

    # Logic Verification: Forward pass check
    with torch.no_grad():
        dummy_output = model(dummy_images.to(Config.DEVICE))

    print(f"Model Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output layer dimension mismatch."

    # ==========================================
    # 4. Training Loop Execution
    # ==========================================
    print("\n[4] Executing Training Loop (1 Epoch)...")

    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    # Initialize Scaler for Mixed Precision
    scaler = torch.cuda.amp.GradScaler(enabled=(Config.DEVICE == "cuda"))

    # Train for one epoch
    train_loss, train_acc = train_one_epoch(
        train_loader, model, criterion, optimizer, scaler, Config.DEVICE, epoch=1
    )
    print(f"Training Completed -> Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

    # Save Checkpoint
    save_checkpoint(
        {"state_dict": model.state_dict(), "epoch": 1},
        is_best=True,
        checkpoint_path=Config.CHECKPOINT_PATH,
        best_model_path=Config.BEST_MODEL_PATH,
    )

    # Logic Verification: Checkpoint existence
    assert os.path.exists(Config.CHECKPOINT_PATH), "Checkpoint file was not created."
    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not created."

    # ==========================================
    # 5. Validation Execution
    # ==========================================
    print("\n[5] Executing Validation...")
    val_loss, val_acc = validate(val_loader, model, criterion, Config.DEVICE)
    print(f"Validation Completed -> Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    # ==========================================
    # 6. Inference & Submission
    # ==========================================
    print("\n[6] Generating Submission (Inference)...")

    test_transform = get_transforms(mode="test", image_size=Config.IMAGE_SIZE)
    test_dataset = INatDataset(
        metadata_file=Config.TEST_METADATA,
        root_dir=Config.INPUT_DIR,
        transform=test_transform,
        mode="test",
        debug=Config.DEBUG,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    inference(test_loader, model, Config.DEVICE)

    # Logic Verification: Submission file check
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission CSV was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission File: {Config.SUBMISSION_PATH}")
    print(f"Rows: {len(df_sub)}")
    print(df_sub.head())

    assert len(df_sub) == len(
        test_dataset
    ), "Submission row count does not match test dataset."
    assert list(df_sub.columns) == [
        "id",
        "predicted",
    ], "Submission columns are incorrect."

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
