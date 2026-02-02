import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import get_cosine_schedule_with_warmup

# Import library modules
from library.config import Config, seed_everything
from library.data import get_dataloaders, get_test_dataloader
from library.models import PawpularityModel
from library.train import train_one_epoch, valid_one_epoch
from library.utils import get_score


def demonstrate_pawpularity_pipeline():
    print("Starting Pawpularity Pipeline Demonstration...")

    # ------------------------------------------------------------------------
    # 1. Configuration and Setup
    # ------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Override Config for speed and debugging
    Config.debug = True
    Config.debug_sample_size = 50  # Very small subset
    Config.epochs = 1
    Config.batch_size = 4
    Config.num_folds = 5
    Config.num_workers = 0  # Avoid multiprocessing overhead for small demo

    # Use a lightweight model for the demo instead of the large ones in default config
    # We use resnet18 as it is standard and likely available in timm
    demo_model_name = "resnet18"
    Config.model_names = [demo_model_name]

    seed_everything(Config.seed)
    Config.create_dirs()

    print(f"    Debug Mode: {Config.debug}")
    print(f"    Batch Size: {Config.batch_size}")
    print(f"    Device: {Config.device}")

    # ------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # ------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Get dataloaders for Fold 0
    train_loader, val_loader = get_dataloaders(fold_idx=0)

    # Fetch one batch to verify structure
    images, metadata, targets = next(iter(train_loader))

    print(f"    Image Batch Shape: {images.shape}")
    print(f"    Metadata Batch Shape: {metadata.shape}")
    print(f"    Targets Batch Shape: {targets.shape}")

    # Assertions to verify data integrity
    assert images.shape == (
        Config.batch_size,
        3,
        Config.image_size,
        Config.image_size,
    ), "Image batch shape mismatch"
    assert metadata.shape == (
        Config.batch_size,
        12,
    ), "Metadata batch shape mismatch (expected 12 features)"
    assert targets.shape == (Config.batch_size,), "Targets batch shape mismatch"
    # Check normalization/scaling of targets (should be 0-1 for training)
    assert (
        targets.max() <= 1.0 and targets.min() >= 0.0
    ), "Training targets should be scaled to [0, 1]"

    # Verify Test Loader
    test_loader, test_df = get_test_dataloader()
    test_imgs, test_meta, test_targs = next(iter(test_loader))
    assert test_imgs.shape[0] == Config.batch_size, "Test loader batch size mismatch"
    print("    Data Loading verified successfully.")

    # ------------------------------------------------------------------------
    # 3. Model Demonstration
    # ------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    # Instantiate model
    # pretrained=False to avoid downloading weights during this quick demo if not cached
    model = PawpularityModel(model_name=demo_model_name, pretrained=False)
    model.to(Config.device)

    # Move batch to device
    images = images.to(Config.device)
    metadata = metadata.to(Config.device)

    # Forward pass
    logits = model(images, metadata)

    print(f"    Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (Config.batch_size, 1), "Model output shape mismatch"
    print("    Model forward pass verified successfully.")

    # ------------------------------------------------------------------------
    # 4. Training Loop Component Demonstration
    # ------------------------------------------------------------------------
    print("\n[4] Demonstrating Training Components (One Epoch)...")

    # Setup training components
    criterion = nn.BCEWithLogitsLoss()

    # Simple optimizer setup for demo
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)

    # Scheduler
    num_train_steps = len(train_loader) * Config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    # Run Train Step
    print("    Running training epoch...")
    train_loss = train_one_epoch(
        model, optimizer, scheduler, train_loader, Config.device, criterion
    )
    print(f"    Train Loss: {train_loss:.4f}")

    # Run Validation Step
    print("    Running validation epoch...")
    val_loss, val_preds, val_targets = valid_one_epoch(
        model, val_loader, Config.device, criterion
    )
    print(f"    Val Loss: {val_loss:.4f}")

    # Assertions
    assert not np.isnan(train_loss), "Training loss returned NaN"
    assert len(val_preds) == len(
        val_targets
    ), "Validation predictions/targets length mismatch"
    assert np.all(
        (val_preds >= 0) & (val_preds <= 1)
    ), "Validation predictions not in [0, 1]"

    # ------------------------------------------------------------------------
    # 5. Metric Verification
    # ------------------------------------------------------------------------
    print("\n[5] Verifying Metric Calculation (RMSE)...")

    # Calculate RMSE on the validation batch
    val_rmse = get_score(val_targets, val_preds)
    print(f"    Validation RMSE (calculated): {val_rmse:.4f}")

    # Test Metric Logic with synthetic data
    # Case 1: Ground truth is scaled [0, 1] (like in training/validation loop)
    # y_true = 0.5 (represents 50), y_pred = 0.6 (represents 60) -> Error is 10
    y_true_scaled = np.array([0.5])
    y_pred_scaled = np.array([0.6])
    rmse_scaled = get_score(y_true_scaled, y_pred_scaled)

    # Case 2: Ground truth is original scale [1, 100] (like in final evaluation)
    y_true_orig = np.array([50.0])
    y_pred_probs = np.array([0.6])  # Preds are always probabilities
    rmse_orig = get_score(y_true_orig, y_pred_probs)

    print(f"    Test RMSE (Scaled Input): {rmse_scaled:.4f}")
    print(f"    Test RMSE (Original Input): {rmse_orig:.4f}")

    # Assertions
    # 0.6 * 100 - 0.5 * 100 = 60 - 50 = 10. RMSE should be 10.
    assert np.isclose(
        rmse_scaled, 10.0, atol=1e-4
    ), f"Metric calculation failed for scaled inputs. Got {rmse_scaled}, expected 10.0"
    assert np.isclose(
        rmse_orig, 10.0, atol=1e-4
    ), f"Metric calculation failed for original inputs. Got {rmse_orig}, expected 10.0"

    print("    Metric calculation verified successfully.")

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    demonstrate_pawpularity_pipeline()
