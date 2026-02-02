import os
import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

# Import library modules
from library import config
from library import utils
from library import data_factory
from library import model_factory
from library import sam
from library import engine
from library import workflow


def run_demo():
    print("=== Starting Demonstration of Iceberg Classification Pipeline ===\n")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed
    # ---------------------------------------------------------
    print("[1] Overriding Configuration for Demo Speed...")

    # Redirect outputs to a demo directory
    config.WORKING_DIR = "./working/demo_execution"
    config.CHECKPOINT_DIR = os.path.join(config.WORKING_DIR, "checkpoints")
    config.SUBMISSION_DIR = os.path.join(config.WORKING_DIR, "submission")
    config.CACHE_DIR = os.path.join(config.WORKING_DIR, "cache")

    # Ensure directories exist
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Reduce computational load
    config.IMG_SIZE = 75  # Keep small to avoid resize overhead during demo, or 224 if needed by model.
    # Model uses ResNet, so 224 is standard, but let's stick to config default 224
    # to ensure model compatibility (ResNet expects > 32x32).
    config.BATCH_SIZE = 8
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    config.MAX_EPOCHS_PHASE_1 = 1
    config.SWA_DURATION_EPOCHS = 1
    config.EARLY_STOPPING_PATIENCE = 1

    # Set seeds
    utils.seed_everything(42)
    print("Configuration updated.\n")

    # ---------------------------------------------------------
    # 2. Data Factory Verification
    # ---------------------------------------------------------
    print("[2] Verifying Data Factory...")

    # Load data (this will process raw JSONs as cache is empty in demo dir)
    t_imgs, t_angs, t_lbls, t_ids, test_imgs, test_angs, test_ids = (
        data_factory.load_processed_data(load_cached_data=True)
    )

    # Assertions for raw data loading
    assert (
        len(t_imgs) == len(t_angs) == len(t_lbls) == len(t_ids)
    ), "Training data arrays must have same length"
    assert (
        len(test_imgs) == len(test_angs) == len(test_ids)
    ), "Test data arrays must have same length"
    assert t_imgs.shape[1:] == (
        75,
        75,
        2,
    ), f"Expected (75, 75, 2), got {t_imgs.shape[1:]}"
    print(f"Loaded {len(t_imgs)} training samples and {len(test_imgs)} test samples.")

    # Verify Dataset and DataLoader
    # Create a small subset for verification
    subset_indices = np.arange(16)
    train_loader, val_loader, test_loader = data_factory.get_dataloaders(
        batch_size=config.BATCH_SIZE, train_idxs=subset_indices, val_idxs=subset_indices
    )

    # Check one batch
    images, angles, labels = next(iter(train_loader))

    # Assertions for DataLoader output
    # Note: IcebergDataset resizes to config.IMG_SIZE (224) and stacks to 3 channels
    assert images.shape == (
        config.BATCH_SIZE,
        3,
        config.IMG_SIZE,
        config.IMG_SIZE,
    ), f"Batch shape mismatch. Expected ({config.BATCH_SIZE}, 3, {config.IMG_SIZE}, {config.IMG_SIZE}), got {images.shape}"
    assert angles.shape == (config.BATCH_SIZE,), "Angles shape mismatch"
    assert labels.shape == (config.BATCH_SIZE,), "Labels shape mismatch"
    print("Data Factory and DataLoader verified successfully.\n")

    # ---------------------------------------------------------
    # 3. Model Factory Verification
    # ---------------------------------------------------------
    print("[3] Verifying Model Factory...")

    device = utils.get_device()
    model = model_factory.get_model()
    model.to(device)

    # Test Forward Pass
    dummy_img = images.to(device)
    dummy_ang = angles.to(device)

    with torch.no_grad():
        output = model(dummy_img, dummy_ang)

    assert output.shape == (
        config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected ({config.BATCH_SIZE}, 1), got {output.shape}"
    print("Model instantiated and forward pass verified.\n")

    # ---------------------------------------------------------
    # 4. SAM Optimizer & Engine Verification
    # ---------------------------------------------------------
    print("[4] Verifying SAM Optimizer and Training Engine...")

    # Setup components
    base_optimizer = torch.optim.AdamW
    optimizer = sam.SAM(
        model.parameters(), base_optimizer, rho=0.05, lr=1e-3, weight_decay=0.0
    )
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.base_optimizer)

    trainer = engine.Engine(
        model=model,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
    )

    # Run one training epoch on the subset loader
    print("Running 1 epoch of training on subset...")
    loss, acc = trainer.train_one_epoch(train_loader, epoch=1)

    assert not np.isnan(loss), "Training loss is NaN"
    assert 0 <= acc <= 1, "Accuracy should be between 0 and 1"
    print(f"Train Step Verified. Loss: {loss:.4f}, Acc: {acc:.4f}")

    # Verify TTA Validation
    print("Running TTA Validation on subset...")
    val_loss, val_acc, val_preds, val_targets = trainer.validate_tta(val_loader)
    assert len(val_preds) == len(
        subset_indices
    ), "Validation predictions count mismatch"
    print(f"Validation TTA Verified. Loss: {val_loss:.4f}, Acc: {val_acc:.4f}\n")

    # ---------------------------------------------------------
    # 5. SWA Handler Verification
    # ---------------------------------------------------------
    print("[5] Verifying SWA Handler...")

    swa_handler = engine.SWAHandler(model, device)

    # Update SWA parameters
    swa_handler.update(model)

    # Update BN statistics
    swa_handler.update_bn(train_loader)

    swa_model = swa_handler.get_model()
    assert isinstance(swa_model, torch.nn.Module), "SWA model is not a torch Module"

    # Check if SWA model works
    with torch.no_grad():
        swa_out = swa_model(dummy_img, dummy_ang)
    assert swa_out.shape == (config.BATCH_SIZE, 1), "SWA model output shape mismatch"
    print("SWA Handler verified.\n")

    # ---------------------------------------------------------
    # 6. Workflow Integration (Mini-Run)
    # ---------------------------------------------------------
    print("[6] Verifying Workflow Integration...")

    # We will run the workflow functions with minimal parameters to ensure they execute correctly.
    # Note: run_phase_1_calibration uses StratifiedKFold. We pass n_splits=2 for speed.

    print("-> Running Phase 1 (Calibration) with 2 folds...")
    optimal_epoch = workflow.run_phase_1_calibration(n_splits=2)
    assert (
        isinstance(optimal_epoch, int) and optimal_epoch > 0
    ), "Optimal epoch should be a positive integer"
    print(f"Phase 1 returned optimal epoch: {optimal_epoch}")

    print("-> Running Phase 2 (Production) with 1 model...")
    # This will train 1 model for optimal_epoch + swa_duration (1+1=2 epochs total)
    workflow.run_phase_2_production(optimal_epochs=optimal_epoch, n_models=1)

    expected_checkpoint = os.path.join(config.CHECKPOINT_DIR, "swa_model_0.pth")
    assert os.path.exists(
        expected_checkpoint
    ), f"Checkpoint not found at {expected_checkpoint}"
    print("Phase 2 completed and checkpoint saved.")

    print("-> Generating Submission...")
    workflow.generate_submission(n_models=1)

    expected_submission = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(
        expected_submission
    ), f"Submission file not found at {expected_submission}"

    df_sub = pd.read_csv(expected_submission)
    assert list(df_sub.columns) == ["id", "is_iceberg"], "Submission columns mismatch"
    assert len(df_sub) == len(
        test_ids
    ), f"Submission length mismatch. Expected {len(test_ids)}, got {len(df_sub)}"
    print("Submission generated successfully.\n")

    print("=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
