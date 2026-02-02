import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import load_dataset_dataframe, get_transforms, AppleDataset
from library.model import AppleMaxViT
from library.engine import fit, predict


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print("Initializing Configuration...")

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Override Config parameters for a fast demonstration
    Config.EPOCHS = 1
    Config.DEBUG = True

    # Define a specific working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update paths to point to the demo working directory
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")
    Config.LOG_PATH = os.path.join(Config.WORKING_DIR, "demo_log.txt")

    # Use distinct cache files for the demo
    Config.TRAIN_CACHE_PATH = os.path.join(Config.WORKING_DIR, "train_demo.parquet")
    Config.VAL_CACHE_PATH = os.path.join(Config.WORKING_DIR, "val_demo.parquet")
    Config.TEST_CACHE_PATH = os.path.join(Config.WORKING_DIR, "test_demo.parquet")

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # =========================================================================
    # 2. Data Loading & Processing
    # =========================================================================
    print("\nLoading Data...")

    # Load metadata
    # We set load_cached_data=False to ensure we load from CSV and then subset manually
    df_train_full = load_dataset_dataframe(
        Config.TRAIN_METADATA_PATH, Config.TRAIN_CACHE_PATH, load_cached_data=False
    )
    df_val_full = load_dataset_dataframe(
        Config.VAL_METADATA_PATH, Config.VAL_CACHE_PATH, load_cached_data=False
    )

    # Subset data for speed (50 samples for train, 20 for val)
    df_train = df_train_full.head(50).reset_index(drop=True)
    df_val = df_val_full.head(20).reset_index(drop=True)

    print(f"Training samples (subset): {len(df_train)}")
    print(f"Validation samples (subset): {len(df_val)}")

    # Instantiate Datasets
    train_dataset = AppleDataset(df_train, transforms=get_transforms("train"))
    val_dataset = AppleDataset(df_val, transforms=get_transforms("valid"))

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # =========================================================================
    # 3. Model Initialization & logic Verification
    # =========================================================================
    print("\nInitializing Model...")

    model = AppleMaxViT(pretrained=True)
    model.to(Config.DEVICE)

    # Verify Forward Pass Logic
    # Create a dummy input tensor matching the expected shape (Batch, Channels, Height, Width)
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(Config.DEVICE)

    print("Verifying model forward pass...")
    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape: (Batch Size, Num Classes)
    expected_shape = (2, Config.NUM_CLASSES)
    if output.shape != expected_shape:
        raise AssertionError(
            f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"
        )

    print("Model logic verified successfully.")

    # =========================================================================
    # 4. Training Loop
    # =========================================================================
    print("\nStarting Training...")

    # Setup Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # Run Training
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        epochs=Config.EPOCHS,
    )

    # Verify Model Artifact
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH} after training."
        )

    print("Training complete. Best model saved.")

    # =========================================================================
    # 5. Inference & Submission
    # =========================================================================
    print("\nStarting Inference...")

    # Load Test Data
    df_test_full = load_dataset_dataframe(
        Config.TEST_METADATA_PATH, Config.TEST_CACHE_PATH, load_cached_data=False
    )
    df_test = df_test_full.head(20).reset_index(drop=True)  # Subset for speed

    test_dataset = AppleDataset(df_test, transforms=get_transforms("valid"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model Weights
    # Note: In a real scenario, we would reload the weights.
    # Since fit() saves the model, we load it back to ensure the saved file is valid.
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))

    # Generate Predictions
    predict(model, test_loader, Config.DEVICE)

    # Verify Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    # Validate Submission Format
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {submission_df.shape}")

    if list(submission_df.columns) != ["image", "labels"]:
        raise AssertionError(f"Invalid submission columns: {submission_df.columns}")

    if len(submission_df) != len(df_test):
        raise AssertionError(
            f"Submission row count mismatch. Expected {len(df_test)}, got {len(submission_df)}"
        )

    print("\nDemo execution completed successfully!")


if __name__ == "__main__":
    main()
