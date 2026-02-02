import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_loaders
from library.model import AppleClassifier
from library.engine import train_model, predict_with_tta


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("--- Setting up Configuration ---")

    # Override Config for a fast demonstration
    Config.EPOCHS = 1
    Config.WORKING_DIR = "./working/demo_run"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Disable pretrained weights download to ensure offline execution speed
    Config.PRETRAINED = False

    # Create directories
    Config.setup()

    # Set seeds
    seed_everything(Config.SEED)

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")

    # ==========================================
    # 2. Data Loading (Debug Mode)
    # ==========================================
    print("\n--- Initializing DataLoaders (Debug Mode) ---")

    # debug=True loads only 100 samples per dataset
    train_loader, val_loader, test_loader = get_loaders(
        load_cached_data=False, debug=True
    )

    # Verify DataLoaders are not empty
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Val loader is empty"
    assert len(test_loader) > 0, "Test loader is empty"

    # Verify Batch Structure
    images, targets = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")  # Expected: (Batch, 3, 224, 224)
    print(f"Batch Target Shape: {targets.shape}")  # Expected: (Batch, 6)

    assert images.dim() == 4, "Images should be 4-dimensional (B, C, H, W)"
    assert images.size(1) == 3, "Images should have 3 channels"
    assert (
        images.size(2) == Config.IMAGE_SIZE and images.size(3) == Config.IMAGE_SIZE
    ), f"Images should be {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}"
    assert (
        targets.size(1) == Config.NUM_CLASSES
    ), f"Targets should have {Config.NUM_CLASSES} classes"

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\n--- Initializing Model ---")

    model = AppleClassifier(pretrained=Config.PRETRAINED)
    model.to(device)

    # Verify Forward Pass
    with torch.no_grad():
        # Pass the batch fetched earlier
        dummy_logits = model(images.to(device))

    assert dummy_logits.shape == (
        images.size(0),
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"
    print("Model forward pass successful.")

    # ==========================================
    # 4. Training Loop Execution
    # ==========================================
    print("\n--- Starting Training Demo ---")

    # Define Optimizer and Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Multi-label classification loss
    criterion = nn.BCEWithLogitsLoss()

    # Run training
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=Config.EPOCHS,
        patience=1,
    )

    # Verify Checkpoint Creation
    if not os.path.exists(Config.BEST_MODEL_PATH):
        # In case validation didn't improve (unlikely with random init vs data),
        # force save for the sake of the demo flow or check logic.
        # However, train_model saves if val_f1 > -1.0. Initial best is -1.0.
        # So it should save at least once unless F1 is exactly -1.0 (impossible).
        raise FileNotFoundError(f"Checkpoint not found at {Config.BEST_MODEL_PATH}")

    print("Training demo complete. Checkpoint verified.")

    # ==========================================
    # 5. Inference & Submission
    # ==========================================
    print("\n--- Starting Inference Demo ---")

    predict_with_tta(
        model=model,
        loader=test_loader,
        device=device,
        output_path=Config.SUBMISSION_PATH,
    )

    # Verify Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Head:\n{df_sub.head()}")

    # Check Columns
    assert "image" in df_sub.columns, "Submission missing 'image' column"
    assert "labels" in df_sub.columns, "Submission missing 'labels' column"

    # Check Row Count (Should match the debug subset size)
    # The debug subset is usually 100, or length of test file if smaller.
    # Sample submission has 3727 rows, debug takes head(100).
    assert len(df_sub) > 0, "Submission dataframe is empty"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
