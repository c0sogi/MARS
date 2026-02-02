import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_score
from library.dataset import get_dataloaders
from library.model import WhaleDetector
from library.loss import WeightedBCELoss
from library.train import train_one_epoch, validate, inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("Starting Right Whale Detection Library Demo...")

    # =========================================================================
    # 1. Configuration Setup for Demo
    # =========================================================================
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config for speed and debugging
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 32  # Small subset for quick execution
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script
    Config.PRETRAINED = False  # Avoid downloading weights for this demo

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print("    Configuration updated: DEBUG=True, BATCH_SIZE=8, EPOCHS=1")

    # =========================================================================
    # 2. Data Loading Verification
    # =========================================================================
    print("\n[2] Verifying Data Loading...")

    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.DEBUG, batch_size=Config.BATCH_SIZE
    )

    # Verify Train Loader
    try:
        images, labels = next(iter(train_loader))
        print(f"    Train Batch - Images Shape: {images.shape}")
        print(f"    Train Batch - Labels Shape: {labels.shape}")

        # Assertions
        # Expected: (Batch, 1, 224, 224)
        expected_img_shape = (Config.BATCH_SIZE, 1, 224, 224)
        assert (
            images.shape == expected_img_shape
        ), f"Image shape mismatch. Expected {expected_img_shape}, got {images.shape}"

        # Expected: (Batch,) or (Batch, 1) depending on collation, usually (Batch,) from dataset
        assert (
            labels.shape[0] == Config.BATCH_SIZE
        ), f"Label batch size mismatch. Expected {Config.BATCH_SIZE}, got {labels.shape[0]}"

        print("    Data Loading assertions passed.")
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    # =========================================================================
    # 3. Model Instantiation & Forward Pass
    # =========================================================================
    print("\n[3] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = WhaleDetector(pretrained=Config.PRETRAINED)
    model.to(device)

    # Move batch to device
    images = images.to(device)

    # Forward pass
    logits = model(images)
    print(f"    Model Output (Logits) Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {logits.shape}"

    print("    Model forward pass successful.")

    # =========================================================================
    # 4. Loss Function Verification
    # =========================================================================
    print("\n[4] Verifying Loss Function...")

    criterion = WeightedBCELoss(device=device)

    # Move labels to device
    labels = labels.to(device)

    # Calculate loss
    loss = criterion(logits, labels)
    print(f"    Calculated Loss: {loss.item():.4f}")

    # Assertions
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    print("    Loss function verification successful.")

    # =========================================================================
    # 5. Training Loop Demonstration
    # =========================================================================
    print("\n[5] Demonstrating Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Train for one epoch
    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, device, epoch=1
    )
    print(f"    Epoch 1 Train Loss: {train_loss:.4f}")

    # Validate
    val_loss, val_auc = validate(model, val_loader, criterion, device)
    print(f"    Epoch 1 Val Loss: {val_loss:.4f}")
    print(f"    Epoch 1 Val AUC: {val_auc:.4f}")

    # Assertions
    assert train_loss > 0, "Train loss should be positive"
    assert val_loss > 0, "Validation loss should be positive"
    assert 0 <= val_auc <= 1, "AUC should be between 0 and 1"

    print("    Training loop execution successful.")

    # =========================================================================
    # 6. Inference & Submission Generation
    # =========================================================================
    print("\n[6] Demonstrating Inference and Submission...")

    # Run inference
    preds = inference(model, test_loader, device)
    print(f"    Inference completed. Predictions shape: {preds.shape}")

    # Verify predictions count matches the debug subset size for test
    # Note: dataset.process_and_cache_data slices the dataframe by DEBUG_SUBSET_SIZE
    expected_preds = min(Config.DEBUG_SUBSET_SIZE, len(pd.read_csv(Config.TEST_CSV)))

    # Allow for slight mismatch if batch drop_last affects things (though test loader usually doesn't drop last)
    # In this specific codebase, test loader does not drop last.
    assert (
        len(preds) == expected_preds
    ), f"Prediction count mismatch. Expected {expected_preds}, got {len(preds)}"

    # Generate dummy submission
    test_df = pd.read_csv(Config.TEST_CSV).head(expected_preds)
    submission = pd.DataFrame({"clip": test_df["clip_name"], "probability": preds})

    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"    Submission file generated at: {submission_path}")
    print("    First 5 rows:")
    print(submission.head())

    # Verify file existence
    assert os.path.exists(submission_path), "Submission file was not created."

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
