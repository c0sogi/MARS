import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import provided library modules
from library import config
from library import utils
from library import data
from library import models
from library import losses
from library import engine


def run_demo():
    print("Starting Library Demo...")

    # 1. Setup and Reproducibility
    utils.seed_everything(config.SEED)
    device = config.DEVICE
    print(f"Device: {device}")

    # Override config for speed in this demo
    config.MAX_DEBUG_SAMPLES = 20  # Use only 20 samples for data loading
    config.EPOCHS = 1  # Run only 1 epoch
    config.BATCH_SIZE = 4  # Small batch size

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # ------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # ------------------------------------------------------------------------
    print("\n[Demo] Data Loading...")

    # Test get_folds
    df_folds = data.get_folds(load_cached_data=False)
    assert "fold" in df_folds.columns, "Fold column missing in dataframe"
    print(f"Folds generated. Shape: {df_folds.shape}")

    # Test create_loaders (using Fold 0)
    # We use debug=True to limit the dataset size based on config.MAX_DEBUG_SAMPLES
    train_loader, val_loader = data.create_loaders(
        fold=0, model_name="resnet18", batch_size=config.BATCH_SIZE, debug=True
    )

    print(f"Train Loader batches: {len(train_loader)}")
    print(f"Val Loader batches: {len(val_loader)}")

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    images = batch["image"]
    targets = batch["target"]
    soft_targets = batch["soft_target"]
    rec_ids = batch["rec_id"]

    print(f"Batch Image Shape: {images.shape}")  # Expected: [B, 3, H, W]
    print(f"Batch Target Shape: {targets.shape}")  # Expected: [B, Num_Species]

    # Assertions
    assert images.dim() == 4, "Images should be 4D tensors [B, C, H, W]"
    assert targets.dim() == 2, "Targets should be 2D tensors [B, Num_Classes]"
    assert (
        targets.shape[1] == config.NUM_SPECIES
    ), f"Target classes mismatch. Expected {config.NUM_SPECIES}"
    assert images.shape[1] == 3, "Images should have 3 channels (Pseudo-RGB)"

    # ------------------------------------------------------------------------
    # 3. Model Demonstration
    # ------------------------------------------------------------------------
    print("\n[Demo] Model Instantiation...")

    # Instantiate the classifier
    model = models.BirdClassifier(
        model_name="resnet18", num_classes=config.NUM_SPECIES, pretrained=False
    )
    model.to(device)

    # Forward pass check
    images = images.to(device)
    logits = model(images)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (
        config.BATCH_SIZE,
        config.NUM_SPECIES,
    ), "Output shape mismatch"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"

    # ------------------------------------------------------------------------
    # 4. Loss Function Demonstration
    # ------------------------------------------------------------------------
    print("\n[Demo] Loss Calculation...")

    # Instantiate Distillation Loss
    loss_fn = losses.DistillationLoss(gamma=0.5)

    # Calculate loss
    targets = targets.to(device)
    soft_targets = soft_targets.to(device)

    # Test with teacher probabilities (Stage 2 style)
    loss_val = loss_fn(logits, targets, teacher_probs=soft_targets)
    print(f"Computed Loss (with soft targets): {loss_val.item():.4f}")
    assert loss_val.ndim == 0, "Loss should be a scalar"

    # Test without teacher probabilities (Stage 1 style)
    loss_val_hard = loss_fn(logits, targets, teacher_probs=None)
    print(f"Computed Loss (hard only): {loss_val_hard.item():.4f}")

    # ------------------------------------------------------------------------
    # 5. Engine (Training/Validation Loop) Demonstration
    # ------------------------------------------------------------------------
    print("\n[Demo] Training Loop (1 Epoch)...")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Run training epoch
    train_loss = engine.train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
        loss_fn=loss_fn,
        scheduler=None,
    )
    print(f"Train Epoch Loss: {train_loss:.4f}")

    # Run validation epoch
    print("[Demo] Validation Loop...")
    val_loss, val_auc = engine.valid_one_epoch(
        model=model, loader=val_loader, device=device, loss_fn=loss_fn
    )
    print(f"Val Epoch Loss: {val_loss:.4f}")
    print(f"Val Epoch AUC: {val_auc:.4f}")

    # ------------------------------------------------------------------------
    # 6. Inference Demonstration
    # ------------------------------------------------------------------------
    print("\n[Demo] Inference (TTA)...")

    # Create test loader (using a small subset via manual slicing isn't supported by create_test_loader
    # directly without modifying the file, but we can rely on the fact that the test set is small ~64 samples)
    test_loader = data.create_test_loader(
        model_name="resnet18", batch_size=config.BATCH_SIZE
    )

    # Run prediction
    predictions = engine.predict(model, test_loader, device)

    # Verify predictions
    print(f"Number of predictions: {len(predictions)}")
    sample_rec_id = list(predictions.keys())[0]
    sample_pred = predictions[sample_rec_id]

    print(f"Sample Prediction (rec_id={sample_rec_id}): {sample_pred}")
    assert len(sample_pred) == config.NUM_SPECIES, "Prediction vector length mismatch"
    assert np.all(
        (sample_pred >= 0) & (sample_pred <= 1)
    ), "Probabilities out of range [0, 1]"

    # ------------------------------------------------------------------------
    # 7. Metric Utility Demonstration
    # ------------------------------------------------------------------------
    print("\n[Demo] Metric Utility...")

    # Create dummy data to test get_score robustness
    # Case: 2 samples, 3 classes.
    # Class 0: Mixed (0, 1) -> Valid AUC
    # Class 1: All 0s -> Skipped (returns 0 contribution or handled)
    # Class 2: Mixed (1, 0) -> Valid AUC
    y_true_dummy = np.array([[0, 0, 1], [1, 0, 0]])
    y_pred_dummy = np.array([[0.2, 0.1, 0.8], [0.9, 0.1, 0.3]])

    score = utils.get_score(y_true_dummy, y_pred_dummy)
    print(f"Robust AUC Score: {score:.4f}")

    # Manual check:
    # Class 0: true=[0,1], pred=[0.2, 0.9] -> Correct order -> AUC=1.0
    # Class 1: true=[0,0] -> Skipped
    # Class 2: true=[1,0], pred=[0.8, 0.3] -> Correct order -> AUC=1.0
    # Average should be 1.0
    assert score == 1.0, "Metric calculation logic failed on dummy data"

    print("\n" + "=" * 50)
    print("Demo Completed Successfully!")
    print("=" * 50)


if __name__ == "__main__":
    run_demo()
