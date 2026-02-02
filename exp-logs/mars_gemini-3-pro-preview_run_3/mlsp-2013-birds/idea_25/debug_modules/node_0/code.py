import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Ensure the current directory is in the python path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.data import get_data
from library.models import BirdModel
from library.optimization import get_optimizer, get_scheduler
from library.engine import train_one_epoch, validate, inference_with_tta


def run_demonstration():
    print("==== Starting Demonstration ====")

    # 1. Setup and Configuration Overrides for Speed
    print("\n[Step 1] Configuring environment...")
    seed_everything(42)

    # Override Config for a fast demo run
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Use only 20 samples
    Config.BATCH_SIZE = 4  # Small batch size
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.MODEL_BACKBONES = ["resnet18"]  # Use the smallest backbone for demo

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Loading
    print("\n[Step 2] Loading Data...")
    # Force reload from scratch to demonstrate processing logic (load_cached_data=False)
    train_ds, val_ds, test_ds = get_data(load_cached_data=False, debug=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Verification: Check Dataset Sizes
    print(
        f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}"
    )
    assert len(train_ds) == Config.DEBUG_SUBSET_SIZE, "Train dataset size mismatch"
    assert len(val_ds) == Config.DEBUG_SUBSET_SIZE, "Val dataset size mismatch"
    # Test set might be smaller than debug subset size if original is small, but here we expect subset

    # Verification: Check Batch Shapes
    dummy_images, dummy_labels = next(iter(train_loader))
    print(f"Batch Image Shape: {dummy_images.shape}")
    print(f"Batch Label Shape: {dummy_labels.shape}")

    assert dummy_images.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), "Incorrect image tensor shape"
    assert dummy_labels.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Incorrect label tensor shape"
    assert dummy_labels.dtype == torch.float32, "Labels should be float32"

    # 3. Model Instantiation
    print("\n[Step 3] Instantiating Model...")
    # Using ResNet18 as defined in Config override
    model = BirdModel(
        backbone_name="resnet18", num_classes=Config.NUM_CLASSES, pretrained=False
    )
    model.to(device)

    # Verification: Forward pass
    with torch.no_grad():
        output = model(dummy_images.to(device))
    assert output.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"
    print("Model instantiated and forward pass successful.")

    # 4. Optimizer and Scheduler (LLRD Verification)
    print("\n[Step 4] Configuring Optimizer (LLRD) and Scheduler...")
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, epochs=Config.EPOCHS)

    # Verification: Check LLRD (Layer-Wise Learning Rate Decay)
    # The optimizer should have multiple param groups with different LRs.
    # Group 0 (Head) should have the highest LR (Base LR).
    # Subsequent groups should have lower LRs.
    param_groups = optimizer.param_groups
    print(f"Number of parameter groups: {len(param_groups)}")
    assert len(param_groups) > 1, "Optimizer should have multiple param groups for LLRD"

    head_lr = param_groups[0]["lr"]
    stem_lr = param_groups[-1]["lr"]
    print(f"Head LR: {head_lr}, Stem LR: {stem_lr}")

    # Allow for floating point tolerance, but Head LR should be approx Config.LEARNING_RATE
    assert (
        abs(head_lr - Config.LEARNING_RATE) < 1e-6
    ), "Head LR does not match base learning rate"
    assert head_lr > stem_lr, "LLRD failed: Head LR should be greater than Stem LR"

    print("Optimizer configuration verified.")

    # 5. Training Loop
    print("\n[Step 5] Running Training Loop (1 Epoch)...")
    train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch=0)
    print(f"Train Loss: {train_loss:.4f}")
    assert isinstance(train_loss, float), "Train loss should be a float"
    assert train_loss > 0, "Train loss should be positive"

    # Step Scheduler
    scheduler.step()

    # 6. Validation Loop
    print("\n[Step 6] Running Validation...")
    val_loss, val_auc = validate(model, val_loader, device)
    print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
    assert isinstance(val_loss, float), "Val loss should be a float"
    assert isinstance(val_auc, float), "Val AUC should be a float"
    # AUC is 0.5 if undefined/constant, or between 0 and 1.
    assert 0.0 <= val_auc <= 1.0, "AUC score out of range"

    # 7. Inference with TTA
    print("\n[Step 7] Running Inference with TTA...")
    # TTA averages 3 passes (Original, Left Shift, Right Shift)
    predictions = inference_with_tta(model, test_loader, device)

    print(f"Predictions Shape: {predictions.shape}")
    assert predictions.shape == (
        len(test_ds),
        Config.NUM_CLASSES,
    ), "Prediction shape mismatch"
    assert (
        predictions.min() >= 0.0 and predictions.max() <= 1.0
    ), "Predictions should be probabilities [0, 1]"

    # 8. Formatting Submission
    print("\n[Step 8] Formatting Submission...")
    # Map predictions to submission format
    # Submission requires 'Id' = rec_id * 100 + species_id

    # Load test metadata to get rec_ids
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    # Since we used debug mode, we need to slice the metadata to match the subset used in DataLoader
    # Note: get_data slices the arrays, but the dataset object doesn't hold the rec_ids directly.
    # However, the order is preserved.
    test_meta_subset = test_meta.iloc[: len(test_ds)]

    submission_rows = []
    for i, row in test_meta_subset.iterrows():
        rec_id = row["rec_id"]
        probs = predictions[i]
        for species_idx, prob in enumerate(probs):
            row_id = int(rec_id * 100 + species_idx)
            submission_rows.append({"Id": row_id, "Probability": prob})

    submission_df = pd.DataFrame(submission_rows)
    print(f"Submission DataFrame Head:\n{submission_df.head()}")

    assert (
        "Id" in submission_df.columns and "Probability" in submission_df.columns
    ), "Submission columns missing"
    assert (
        len(submission_df) == len(test_ds) * Config.NUM_CLASSES
    ), "Incorrect number of submission rows"

    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    run_demonstration()
