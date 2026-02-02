import os
import torch
import numpy as np
import torch.nn as nn
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, count_parameters
from library.data_loader import get_loaders
from library.model import IcebergResNet18
from library.engine import (
    train_one_epoch,
    validate_with_tta,
    predict_tta,
    DistillationLoss,
)


def run_demo():
    # 1. Setup and Configuration Override
    print("--- 1. Initializing Configuration ---")

    # Override Config for a fast demo run
    Config.WORKING_DIR = "./working/demo_execution"
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.N_FOLDS = 5
    Config.DEBUG = True

    # Clean up demo directory if it exists to ensure fresh run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.create_directories()

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print("Configuration configured for fast execution.")

    # 2. Data Loading Demonstration
    print("\n--- 2. Demonstrating Data Loading ---")

    # We use load_cached_data=False to force the processing logic to run
    # This verifies process_json_data, imputation, and normalization
    loaders = get_loaders(fold_idx=0, stage="teacher", load_cached_data=False)

    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]

    print(f"Train Loader batches: {len(train_loader)}")
    print(f"Val Loader batches: {len(val_loader)}")

    # Fetch a single batch to verify shapes
    batch = next(iter(train_loader))
    images = batch["image"]
    angles = batch["angle"]
    labels = batch["label"]

    # Verify Image Shapes: [Batch, 3, 224, 224] (after Albumentations resize)
    assert images.dim() == 4, f"Expected 4D image tensor, got {images.dim()}"
    assert images.shape[1] == 3, f"Expected 3 channels, got {images.shape[1]}"
    assert images.shape[2] == Config.IMAGE_SIZE, "Image height mismatch"
    assert images.shape[3] == Config.IMAGE_SIZE, "Image width mismatch"

    # Verify Angle Shapes: [Batch]
    assert angles.dim() == 1, f"Expected 1D angle tensor, got {angles.dim()}"
    assert (
        angles.shape[0] == images.shape[0]
    ), "Batch size mismatch between images and angles"

    # Verify Label Shapes: [Batch]
    assert labels.dim() == 1, f"Expected 1D label tensor, got {labels.dim()}"

    print("Data batch shapes verified successfully.")

    # 3. Model Instantiation and Forward Pass
    print("\n--- 3. Demonstrating Model Logic ---")

    device = Config.DEVICE
    model = IcebergResNet18().to(device)

    # Check parameter count
    num_params = count_parameters(model)
    print(f"Model instantiated with {num_params:,} trainable parameters.")
    assert num_params > 0, "Model has no trainable parameters!"

    # Move batch to device
    images = images.to(device)
    angles = angles.to(device)

    # Perform Forward Pass
    logits = model(images, angles)

    # Verify Output Shape: [Batch, 1]
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected output shape {(Config.BATCH_SIZE, 1)}, got {logits.shape}"

    print("Model forward pass successful.")

    # 4. Loss Function Verification
    print("\n--- 4. Demonstrating Loss Function (Distillation) ---")

    # Initialize Loss
    criterion = DistillationLoss(alpha=0.5)

    # Create dummy targets
    dummy_logits = torch.randn(Config.BATCH_SIZE, 1, device=device)
    dummy_targets = torch.randint(0, 2, (Config.BATCH_SIZE, 1), device=device).float()
    dummy_soft = torch.rand(
        Config.BATCH_SIZE, 1, device=device
    )  # Teacher probabilities

    # Test Hard Label Loss (Standard BCE)
    loss_hard = criterion(dummy_logits, dummy_targets)
    assert loss_hard.item() > 0, "Hard loss should be positive"

    # Test Distillation Loss (Hard + Soft)
    loss_distill = criterion(dummy_logits, dummy_targets, soft_targets=dummy_soft)
    assert loss_distill.item() > 0, "Distillation loss should be positive"

    print("DistillationLoss logic verified.")

    # 5. Training Engine Demonstration
    print("\n--- 5. Demonstrating Training Engine ---")

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Train for 1 epoch
    # We use the train_loader from step 2
    avg_loss = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
        epoch=1,
        use_distillation=False,
    )

    assert not np.isnan(avg_loss), "Training loss returned NaN"
    print(f"Training Epoch completed. Average Loss: {avg_loss:.4f}")

    # 6. Validation Engine Demonstration (TTA)
    print("\n--- 6. Demonstrating Validation with TTA ---")

    # Validate on the validation set
    val_score = validate_with_tta(model, val_loader, device)

    assert not np.isnan(val_score), "Validation score returned NaN"
    assert val_score >= 0, "Log loss cannot be negative"
    print(f"Validation completed. Log Loss: {val_score:.4f}")

    # 7. Inference Demonstration
    print("\n--- 7. Demonstrating Inference ---")

    test_loaders = get_loaders(stage="test", load_cached_data=True)
    test_loader = test_loaders["test_loader"]

    # Predict on test set
    # Note: This runs on the full test set. Since it's small (321 images), it's fast.
    preds = predict_tta(model, test_loader, device)

    assert len(preds) == 321, f"Expected 321 predictions, got {len(preds)}"
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Predictions must be probabilities [0, 1]"

    print(f"Inference completed. Generated {len(preds)} predictions.")
    print(f"Sample predictions: {preds[:5]}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
