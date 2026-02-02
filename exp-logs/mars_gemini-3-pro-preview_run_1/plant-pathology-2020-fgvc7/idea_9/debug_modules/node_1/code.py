import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, SWAHelper, average_model_weights
from library.dataset import get_datasets
from library.model import AppleResNet34
from library.loss import get_weighted_loss
from library.engine import train_one_epoch, predict_tta, save_submission


def verify_swa_logic():
    """
    Verifies that the SWA weight averaging logic works mathematically.
    """
    print("\n[1/6] Verifying SWA Logic...")

    # Create two simple linear models
    model1 = nn.Linear(2, 2)
    model2 = nn.Linear(2, 2)

    # Set weights manually: Model 1 has all 1.0, Model 2 has all 3.0
    with torch.no_grad():
        model1.weight.fill_(1.0)
        model1.bias.fill_(1.0)
        model2.weight.fill_(3.0)
        model2.bias.fill_(3.0)

    # Use helper to average
    helper = SWAHelper()
    helper.update(model1)
    helper.update(model2)

    avg_state = helper.get_averaged_weights()

    # Expected average is 2.0
    expected_val = 2.0

    avg_weight = avg_state["weight"]
    avg_bias = avg_state["bias"]

    if not torch.allclose(avg_weight, torch.tensor(expected_val)):
        raise AssertionError(
            f"SWA Weight Averaging failed. Expected {expected_val}, got {avg_weight}"
        )

    if not torch.allclose(avg_bias, torch.tensor(expected_val)):
        raise AssertionError(
            f"SWA Bias Averaging failed. Expected {expected_val}, got {avg_bias}"
        )

    print("SWA Logic Verified: (1.0 + 3.0) / 2 = 2.0")


def main():
    # ==========================================
    # 0. Configuration Overrides for Speed
    # ==========================================
    print("Configuring environment for demonstration...")

    # Enable Debug mode to use a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 0.05  # Use 5% of data

    # Reduce training duration
    Config.EPOCHS_CONVERGENCE = 1
    Config.EPOCHS_SWA = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2

    # Ensure output directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed
    seed_everything(Config.SEED)

    # ==========================================
    # 1. Verify SWA Utility
    # ==========================================
    verify_swa_logic()

    # ==========================================
    # 2. Verify Dataset & Dataloader
    # ==========================================
    print("\n[2/6] Verifying Dataset and DataLoader...")

    # Get datasets (using split mode to have a validation set for demo)
    train_ds, val_ds, test_ds = get_datasets(use_full_data=False)

    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples: {len(test_ds)}")

    if len(train_ds) == 0:
        raise AssertionError(
            "Training dataset is empty. Check DEBUG_SAMPLE_SIZE or metadata."
        )

    # Create DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    # Check one batch
    images, labels = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    # Assertions
    expected_shape = (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    # Note: If dataset size < batch size, shape[0] will be smaller.
    # Since we use 5% of ~1300 (~65 images), batch size 8 is fine.
    if images.shape[1:] != expected_shape[1:]:
        raise AssertionError(
            f"Incorrect image dimensions. Expected {expected_shape[1:]}, got {images.shape[1:]}"
        )

    if labels.max() >= Config.NUM_CLASSES:
        raise AssertionError(
            f"Label index out of bounds. Max label {labels.max()} >= {Config.NUM_CLASSES}"
        )

    # ==========================================
    # 3. Verify Model Architecture
    # ==========================================
    print("\n[3/6] Verifying Model Architecture...")

    device = Config.DEVICE
    model = AppleResNet34(pretrained=False)  # False for speed in demo
    model.to(device)

    # Forward pass check
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    if output.shape != (2, Config.NUM_CLASSES):
        raise AssertionError(
            f"Model output shape mismatch. Expected (2, {Config.NUM_CLASSES}), got {output.shape}"
        )

    # ==========================================
    # 4. Verify Loss Function
    # ==========================================
    print("\n[4/6] Verifying Weighted Loss...")

    # Load metadata df to calculate weights
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    if Config.DEBUG:
        train_df = train_df.sample(
            frac=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
        )

    criterion = get_weighted_loss(train_df, device=device, load_cached_data=False)

    if not isinstance(criterion, nn.CrossEntropyLoss):
        raise AssertionError("Loss function is not CrossEntropyLoss")

    if criterion.weight is None:
        raise AssertionError("Class weights were not assigned to the loss function.")

    print(f"Class Weights: {criterion.weight.cpu().numpy()}")

    # ==========================================
    # 5. Execution: Training Loop (Convergence + SWA)
    # ==========================================
    print("\n[5/6] Executing Training Loop...")

    optimizer = optim.Adam(model.parameters(), lr=Config.LR_CONVERGENCE)
    swa_helper = SWAHelper()

    # Phase 1: Convergence
    print("Phase 1: Convergence")
    for epoch in range(Config.EPOCHS_CONVERGENCE):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"  Epoch {epoch+1}/{Config.EPOCHS_CONVERGENCE} Loss: {loss:.4f}")

    # Phase 2: SWA
    print("Phase 2: SWA")
    # Update LR for SWA
    for param_group in optimizer.param_groups:
        param_group["lr"] = Config.LR_SWA

    for epoch in range(Config.EPOCHS_SWA):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"  SWA Epoch {epoch+1}/{Config.EPOCHS_SWA} Loss: {loss:.4f}")
        swa_helper.update(model)

    # Update model with averaged weights
    print("Updating model with SWA weights...")
    avg_weights = swa_helper.get_averaged_weights()
    if avg_weights:
        model.load_state_dict(avg_weights)
    else:
        raise AssertionError("SWA Helper failed to capture weights.")

    # ==========================================
    # 6. Execution: Inference & Submission
    # ==========================================
    print("\n[6/6] Executing Inference and Saving Submission...")

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Run TTA Prediction
    preds_df = predict_tta(model, test_loader, device)

    print("Predictions head:")
    print(preds_df.head())

    # Validate Output Format
    expected_cols = ["image_id"] + Config.CLASS_LABELS
    if list(preds_df.columns) != expected_cols:
        raise AssertionError(
            f"Submission columns mismatch. Expected {expected_cols}, got {list(preds_df.columns)}"
        )

    if len(preds_df) != len(test_ds):
        raise AssertionError(
            f"Number of predictions ({len(preds_df)}) does not match test set size ({len(test_ds)})"
        )

    # Save
    save_submission(preds_df)

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
