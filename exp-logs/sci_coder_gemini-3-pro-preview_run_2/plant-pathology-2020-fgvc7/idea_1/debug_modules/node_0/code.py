import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Import library modules
from library.config import Config
from library.utils import seed_everything, calculate_class_weights
from library.data_setup import create_dataloaders
from library.model_setup import build_model
from library.engine import train_one_epoch, evaluate, fit, generate_submission


def main():
    print("Starting Apple Disease Detection Demo...")

    # 1. Configuration & Setup
    # Override Config for a fast demonstration
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Use only 20 images for speed
    Config.NUM_WORKERS = 0  # Use main process to avoid overhead in demo

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print("Configuration set for fast execution.")

    # 2. Data Pipeline Verification
    print("\n--- Data Pipeline Verification ---")
    train_loader, val_loader, test_loader = create_dataloaders(debug=Config.DEBUG)

    # Fetch a batch to verify shapes
    images, labels = next(iter(train_loader))

    # Assertions for data shapes
    # Expected: (Batch_Size, Channels, Height, Width) -> (8, 3, 224, 224)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image batch shape: {images.shape}"
    # Expected: (Batch_Size,)
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect label batch shape: {labels.shape}"

    print(
        f"Data loaded successfully. Batch Shape: {images.shape}, Labels Shape: {labels.shape}"
    )

    # 3. Model Initialization & Forward Pass
    print("\n--- Model Initialization ---")
    # Use pretrained=False to avoid downloading weights during this quick demo
    model = build_model(pretrained=False, num_classes=Config.NUM_CLASSES)
    model = model.to(Config.DEVICE)

    # Verify forward pass
    with torch.no_grad():
        dummy_output = model(images.to(Config.DEVICE))

    # Expected Output: (Batch_Size, Num_Classes) -> (8, 4)
    assert dummy_output.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Incorrect model output shape: {dummy_output.shape}"

    print(
        f"Model initialized and forward pass verified. Output Shape: {dummy_output.shape}"
    )

    # 4. Training Engine Demonstration
    print("\n--- Training Engine Demonstration ---")

    # Setup Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Calculate class weights for loss function (handling imbalance)
    # We read the metadata file directly as required by the utility
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    class_weights = calculate_class_weights(train_df, device=Config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Test 'train_one_epoch'
    print("Running single epoch training step...")
    loss = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
    print(f"Single epoch training completed. Average Loss: {loss:.4f}")

    # Test 'fit' function (Full loop with validation)
    print("Running full fit cycle (1 epoch)...")
    save_path = os.path.join(Config.WORKING_DIR, "demo_best_model.pth")

    best_auc = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        device=Config.DEVICE,
        epochs=Config.EPOCHS,
        patience=1,
        save_path=save_path,
    )

    assert os.path.exists(save_path), "Model checkpoint was not saved."
    print(f"Fit cycle completed. Best ROC AUC: {best_auc:.4f}")

    # 5. Inference & Submission
    print("\n--- Inference & Submission ---")
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Generate submission file
    generate_submission(model, test_loader, Config.DEVICE, submission_path)

    # Verify submission file
    assert os.path.exists(submission_path), "Submission file not found."

    sub_df = pd.read_csv(submission_path)
    print(f"Submission file loaded. Shape: {sub_df.shape}")

    # Check columns
    expected_cols = ["image_id"] + Config.TARGET_COLS
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    # Check row count (should match debug subset size)
    assert (
        len(sub_df) == Config.DEBUG_SUBSET_SIZE
    ), f"Submission row count mismatch. Expected {Config.DEBUG_SUBSET_SIZE}, got {len(sub_df)}"

    print("\nAll demonstration steps completed successfully!")


if __name__ == "__main__":
    main()
