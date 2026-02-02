import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.dataset import get_loaders
from library.model import WhaleConvNeXt
from library.train import train_one_epoch, validate, predict_test

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Demonstration of Right Whale Call Detection Pipeline ===\n")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Debugging
    # ---------------------------------------------------------
    print("[1] Configuring environment for fast demonstration...")

    # Enable Debug mode to use a small subset of data (defined in Config as 500, we reduce to 20)
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Very small subset for instant execution

    # Reduce training parameters
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Disable downloading pretrained weights to ensure offline execution capability and speed
    Config.PRETRAINED = False

    # Set device
    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Subset Size: {Config.DEBUG_SUBSET_SIZE}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")

    # Ensure reproducibility
    seed_everything(Config.SEED)
    print("    Configuration complete.\n")

    # ---------------------------------------------------------
    # 2. Verify Utility Functions
    # ---------------------------------------------------------
    print("[2] Verifying Utility Functions...")

    # Test ROC AUC calculation
    y_true_dummy = np.array([0, 1, 0, 1])
    y_pred_dummy = np.array([0.1, 0.9, 0.2, 0.8])
    auc_score = calculate_roc_auc(y_true_dummy, y_pred_dummy)

    print(f"    Dummy ROC AUC Score: {auc_score}")

    # Assert correctness for a perfect prediction scenario
    if auc_score != 1.0:
        raise AssertionError(
            f"ROC AUC calculation failed. Expected 1.0, got {auc_score}"
        )

    # Test with Tensors
    y_true_tensor = torch.tensor([0, 1], dtype=torch.float32)
    y_pred_tensor = torch.tensor([0.2, 0.8], dtype=torch.float32)
    auc_tensor = calculate_roc_auc(y_true_tensor, y_pred_tensor)

    if auc_tensor != 1.0:
        raise AssertionError("ROC AUC calculation with Tensors failed.")

    print("    Utility functions verified.\n")

    # ---------------------------------------------------------
    # 3. Verify Dataset and DataLoaders
    # ---------------------------------------------------------
    print("[3] Initializing DataLoaders and Preprocessing...")

    # We force load_cached_data=False to ensure the preprocessing logic runs and works
    # This reads audio files, computes spectrograms, and resizes them.
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    print(f"    Train Loader Batches: {len(train_loader)}")
    print(f"    Val Loader Batches: {len(val_loader)}")
    print(f"    Test Loader Batches: {len(test_loader)}")

    # Fetch a single batch to verify shapes
    images, labels = next(iter(train_loader))

    print(f"    Batch Image Shape: {images.shape}")
    print(f"    Batch Label Shape: {labels.shape}")

    # Assertions
    expected_shape = (Config.BATCH_SIZE, 1, 224, 224)
    if images.shape != expected_shape:
        raise AssertionError(
            f"Incorrect image shape. Expected {expected_shape}, got {images.shape}"
        )

    if labels.shape[0] != Config.BATCH_SIZE:
        raise AssertionError(
            f"Incorrect label batch size. Expected {Config.BATCH_SIZE}, got {labels.shape[0]}"
        )

    print("    Data loading and preprocessing verified.\n")

    # ---------------------------------------------------------
    # 4. Verify Model Architecture
    # ---------------------------------------------------------
    print("[4] Initializing Model...")

    model = WhaleConvNeXt()
    model = model.to(device)

    # Forward pass check
    images = images.to(device)
    logits = model(images)

    print(f"    Output Logits Shape: {logits.shape}")

    # Assertions
    if logits.shape != (Config.BATCH_SIZE, Config.NUM_CLASSES):
        raise AssertionError(
            f"Incorrect model output shape. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {logits.shape}"
        )

    print("    Model architecture verified.\n")

    # ---------------------------------------------------------
    # 5. Verify Training Loop Logic
    # ---------------------------------------------------------
    print("[5] Running Training Loop (1 Epoch)...")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Train for one epoch
    train_loss, train_auc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    print(f"    Train Loss: {train_loss:.4f}")
    print(f"    Train AUC: {train_auc:.4f}")

    # Validate
    val_loss, val_auc = validate(model, val_loader, criterion, device)

    print(f"    Val Loss: {val_loss:.4f}")
    print(f"    Val AUC: {val_auc:.4f}")

    # Basic sanity checks
    if not np.isfinite(train_loss):
        raise AssertionError("Training loss is not finite (NaN or Inf).")

    print("    Training logic verified.\n")

    # ---------------------------------------------------------
    # 6. Verify Prediction and Submission Generation
    # ---------------------------------------------------------
    print("[6] Generating Predictions on Test Set...")

    # Generate predictions
    predictions = predict_test(model, test_loader, device)

    # Flatten predictions
    predictions = predictions.flatten()

    print(f"    Number of predictions: {len(predictions)}")

    # Check against test dataset size
    # Note: In Debug mode, test dataset size is Config.DEBUG_SUBSET_SIZE
    test_df = pd.read_csv(Config.TEST_CSV)
    if Config.DEBUG:
        test_df = test_df.iloc[: Config.DEBUG_SUBSET_SIZE]

    if len(predictions) != len(test_df):
        raise AssertionError(
            f"Prediction count mismatch. Expected {len(test_df)}, got {len(predictions)}"
        )

    # Create submission dataframe
    submission = pd.DataFrame({"clip": test_df["clip"], "probability": predictions})

    # Save to a temp location to verify writing works
    output_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission.to_csv(output_path, index=False)

    print(f"    Demo submission saved to: {output_path}")
    print(f"    Submission Head:\n{submission.head()}")

    print("\n=== Demonstration Complete: All Systems Go ===")


if __name__ == "__main__":
    run_demo()
