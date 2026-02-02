import os
import sys
import torch
import pandas as pd
import numpy as np
import logging

# Import from the provided library
from library.config import Config
from library.utils import set_seed, get_device, ProbabilisticF1
from library.data import get_dataloaders, MammographyDataset
from library.model import MTSIN
from library.train import MultiTaskLoss, run_training

# Configure Logging to suppress non-error messages for cleaner output during demo
logging.getLogger("data").setLevel(logging.ERROR)
logging.getLogger("train").setLevel(logging.INFO)
logging.getLogger("timm").setLevel(logging.ERROR)


def demonstrate_usage():
    print("=== Starting Demonstration of Breast Cancer Detection Pipeline ===\n")

    # 1. Configuration Overrides for Fast Demonstration
    print("1. Configuring environment for rapid execution...")
    # Modify Config attributes directly to speed up the demo
    Config.DEBUG = True  # Use subsampled data
    Config.NUM_EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.IMG_SIZE = (256, 256)  # Smaller images for faster processing
    Config.PRETRAINED = False  # Skip downloading weights for demo speed/offline safety
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    device = get_device()
    print(f"   Device: {device}")
    print(f"   Debug Mode: {Config.DEBUG}")
    print(f"   Image Size: {Config.IMG_SIZE}")
    print("   Configuration complete.\n")

    # 2. Data Loading Demonstration
    print("2. Verifying Data Loading and Processing...")
    # We call get_dataloaders with debug=True to load a small subset
    # load_cached_data=False forces reprocessing to ensure data logic works
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True
    )

    # Verify Loaders are not empty
    assert len(train_loader) > 0, "Train loader is empty!"
    assert len(val_loader) > 0, "Val loader is empty!"
    assert len(test_loader) > 0, "Test loader is empty!"

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))
    images = batch["image"]
    meta = batch["meta"]
    target_cancer = batch["target_cancer"]

    print(f"   Batch keys: {list(batch.keys())}")
    print(f"   Image shape: {images.shape}")  # Should be (B, 3, 256, 256)
    print(f"   Meta shape: {meta.shape}")  # Should be (B, 5)

    # Assertions for Data
    expected_img_shape = (Config.BATCH_SIZE, 3, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
    assert (
        images.shape == expected_img_shape
    ), f"Image shape mismatch. Expected {expected_img_shape}, got {images.shape}"
    assert meta.shape[1] == len(
        Config.META_FEATURES
    ), "Meta feature dimension mismatch."
    assert target_cancer.shape[0] == Config.BATCH_SIZE, "Target batch size mismatch."
    print("   Data Loading verification passed.\n")

    # 3. Model Instantiation and Forward Pass
    print("3. Verifying Model Architecture and Forward Pass...")
    model = MTSIN().to(device)

    # Move batch to device
    images = images.to(device)
    meta = meta.to(device)

    # Forward pass
    outputs = model(images, meta)

    print(f"   Output keys: {list(outputs.keys())}")

    # Assertions for Model
    assert "cancer" in outputs, "Model output missing 'cancer' logits."
    assert "birads" in outputs, "Model output missing 'birads' logits."
    assert "density" in outputs, "Model output missing 'density' logits."

    # Check shape of cancer logits (B, 1)
    assert outputs["cancer"].shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Cancer output shape mismatch. Got {outputs['cancer'].shape}"
    print("   Model verification passed.\n")

    # 4. Loss Function Verification
    print("4. Verifying Loss Calculation...")
    criterion = MultiTaskLoss(device)

    # Prepare targets dictionary
    targets = {
        "cancer": batch["target_cancer"].to(device),
        "birads": batch["target_birads"].to(device),
        "density": batch["target_density"].to(device),
    }

    # Compute loss
    loss, c_loss, b_loss, d_loss = criterion(outputs, targets)

    print(f"   Total Loss: {loss.item():.4f}")
    print(f"   Cancer Loss: {c_loss:.4f}")

    # Assertions for Loss
    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.item() > 0, "Loss should be positive."
    print("   Loss verification passed.\n")

    # 5. Metric Verification (Probabilistic F1)
    print("5. Verifying Probabilistic F1 Metric...")
    pf1 = ProbabilisticF1()

    # Create dummy deterministic predictions for testing
    # Preds: 0.8, 0.8, 0.2, 0.2
    # Targets: 1, 0, 1, 0
    # pTP = 0.8*1 + 0.8*0 + 0.2*1 + 0.2*0 = 0.8 + 0 + 0.2 + 0 = 1.0
    # pPreds = 0.8 + 0.8 + 0.2 + 0.2 = 2.0
    # pTargets = 1 + 0 + 1 + 0 = 2.0
    # pPrec = 1.0 / 2.0 = 0.5
    # pRec = 1.0 / 2.0 = 0.5
    # pF1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5

    dummy_preds = torch.tensor([0.8, 0.8, 0.2, 0.2])
    dummy_targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

    pf1.update(dummy_preds, dummy_targets)
    score = pf1.compute()

    print(f"   Computed pF1 Score: {score:.4f}")

    # Assertion for Metric
    assert 0.0 <= score <= 1.0, "pF1 score out of range [0, 1]."
    assert (
        abs(score - 0.5) < 1e-5
    ), f"pF1 calculation incorrect. Expected 0.5, got {score}"
    print("   Metric verification passed.\n")

    # 6. Full Training Loop Integration
    print("6. Running Full Training Loop (Debug Mode)...")
    # This runs the logic in library/train.py: get_dataloaders -> train -> validate -> save -> predict
    run_training(debug=True)
    print("   Training loop completed.\n")

    # 7. Submission Verification
    print("7. Verifying Submission File...")
    submission_path = Config.SUBMISSION_PATH

    if os.path.exists(submission_path):
        sub_df = pd.read_csv(submission_path)
        print(f"   Submission file found at {submission_path}")
        print(f"   Shape: {sub_df.shape}")
        print(f"   Columns: {list(sub_df.columns)}")
        print(sub_df.head(3))

        # Assertions
        assert "prediction_id" in sub_df.columns, "Missing prediction_id column."
        assert "cancer" in sub_df.columns, "Missing cancer column."
        assert not sub_df.empty, "Submission file is empty."

        # Check values are probabilities
        assert sub_df["cancer"].min() >= 0.0, "Probabilities < 0 found."
        assert sub_df["cancer"].max() <= 1.0, "Probabilities > 1 found."

        print("   Submission verification passed.")
    else:
        raise FileNotFoundError(f"Submission file not generated at {submission_path}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    demonstrate_usage()
