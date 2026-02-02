import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_dataloaders, IcebergDataset
from library.model import QPWBN
from library.train import train_one_epoch, validate


def run_demo():
    # ==========================================
    # 1. SETUP & CONFIGURATION
    # ==========================================
    print("\n[1] Setup and Configuration")

    # Set seeds for reproducibility
    seed_everything(42)

    # Override Config for Speed/Demo purposes
    # We use the DEBUG flag to load a tiny subset of data (50 samples)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50
    Config.BATCH_SIZE = 8
    Config.NUM_EPOCHS = 1  # Run only 1 epoch for demonstration
    Config.NUM_FOLDS = 2  # Not used in this direct demo, but good practice

    # Ensure working directory exists for artifacts
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    # ==========================================
    # 2. DATA LOADING
    # ==========================================
    print("\n[2] Data Loading and Processing")

    # get_dataloaders handles:
    # - Loading JSONs
    # - Creating 3-channel images (Band1, Band2, Avg)
    # - Imputing incidence angles
    # - Normalization
    # - Splitting into Train/Val/Test loaders
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify DataLoaders
    print("Verifying DataLoader outputs...")

    # Fetch one batch from training
    train_batch = next(iter(train_loader))
    images, inc_angles, labels = train_batch

    # Assertions for shapes
    # Images: (B, 3, 75, 75)
    assert images.dim() == 4, f"Expected 4D image tensor, got {images.dim()}"
    assert images.shape[1] == 3, f"Expected 3 channels, got {images.shape[1]}"
    assert (
        images.shape[2] == 75 and images.shape[3] == 75
    ), f"Expected 75x75 resolution, got {images.shape[2:]}"

    # Incidence Angles: (B,)
    assert (
        inc_angles.dim() == 1
    ), f"Expected 1D incidence angle tensor, got {inc_angles.dim()}"
    assert (
        inc_angles.shape[0] == images.shape[0]
    ), "Batch size mismatch between images and angles"

    # Labels: (B,)
    assert labels.dim() == 1, f"Expected 1D label tensor, got {labels.dim()}"

    print(
        f"Train Batch Shape: Images {images.shape}, Angles {inc_angles.shape}, Labels {labels.shape}"
    )
    print("Data loading verification successful.")

    # ==========================================
    # 3. MODEL INSTANTIATION
    # ==========================================
    print("\n[3] Model Initialization")

    model = QPWBN().to(device)

    # Print model summary (number of parameters)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized QPWBN Model with {num_params:,} trainable parameters.")

    # ==========================================
    # 4. FORWARD PASS CHECK
    # ==========================================
    print("\n[4] Forward Pass Verification")

    # Move batch to device
    images = images.to(device)
    inc_angles = inc_angles.to(device)

    # Perform forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(images, inc_angles)

    # Assert output shape: (Batch_Size,)
    assert outputs.dim() == 1, f"Expected 1D output logits, got {outputs.dim()}"
    assert (
        outputs.shape[0] == images.shape[0]
    ), f"Output batch size mismatch. Expected {images.shape[0]}, got {outputs.shape[0]}"

    print(f"Forward pass successful. Output shape: {outputs.shape}")
    print(f"Sample logits: {outputs[:3].cpu().numpy()}")

    # ==========================================
    # 5. TRAINING LOOP DEMONSTRATION
    # ==========================================
    print("\n[5] Training Loop Demonstration (1 Epoch)")

    # Setup criterion and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Run Training Step
    print("Running training epoch...")
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Epoch 1 Train Loss: {train_loss:.4f}")

    # Run Validation Step
    print("Running validation step...")
    val_loss = validate(model, val_loader, criterion, device)
    print(f"Epoch 1 Val Loss: {val_loss:.4f}")

    # Assert losses are valid floats
    assert not np.isnan(train_loss), "Training loss is NaN"
    assert not np.isnan(val_loss), "Validation loss is NaN"

    # ==========================================
    # 6. INFERENCE & SUBMISSION
    # ==========================================
    print("\n[6] Inference and Submission Generation")

    model.eval()
    predictions = []

    # We need the IDs to map predictions back to the submission file.
    # The current Dataset implementation returns (X, inc) for test set.
    # We can retrieve IDs from the metadata file corresponding to the test indices.
    # Since we are in DEBUG mode, we took the first N samples.

    print("Generating predictions on test set...")
    with torch.no_grad():
        for inputs, inc_angles in test_loader:
            inputs = inputs.to(device)
            inc_angles = inc_angles.to(device)

            # Forward pass
            logits = model(inputs, inc_angles)

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits)
            predictions.extend(probs.cpu().numpy())

    predictions = np.array(predictions)
    print(f"Generated {len(predictions)} predictions.")

    # Load Test Metadata to get IDs
    df_test = pd.read_csv(Config.TEST_CSV)

    # Adjust for DEBUG mode truncation
    if Config.DEBUG:
        df_test = df_test.iloc[: Config.DEBUG_SAMPLE_SIZE]

    assert len(df_test) == len(
        predictions
    ), f"Mismatch between test metadata rows ({len(df_test)}) and predictions ({len(predictions)})"

    # Create Submission DataFrame
    submission = pd.DataFrame({"id": df_test["id"], "is_iceberg": predictions})

    # Verify values are probabilities
    assert submission["is_iceberg"].min() >= 0.0, "Probabilities < 0 detected"
    assert submission["is_iceberg"].max() <= 1.0, "Probabilities > 1 detected"

    print("Sample Submission Head:")
    print(submission.head())

    # Save to file (in a temp location for demo)
    output_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    run_demo()
