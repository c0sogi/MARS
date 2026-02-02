import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_score
from library.data import get_loaders
from library.models import BirdModel
from library.loss import WeightedBCELoss, AnchorDistillationLoss
from library.sam import SAM
from library.engine import (
    train_one_epoch,
    valid_one_epoch,
    inference_fn,
    save_submission,
)


def main():
    print("Starting Demonstration Script...")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for speed and demonstration purposes
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Use a very small subset
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.WORKING_DIR = "./working/demo_execution"
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directory exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set random seed
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[Step 2] Initializing Data Loaders...")

    # We pass soft_targets=None for the initial stage (no distillation yet)
    train_loader, val_loader, test_loader = get_loaders(
        fold=0,
        load_cached_data=False,  # Force reload to demonstrate processing
        soft_targets=None,
        debug=Config.DEBUG,
    )

    # Validation: Check batch structure
    images, hard_targets, soft_targets, rec_ids = next(iter(train_loader))

    print(f"Train Batch - Images: {images.shape}")
    print(f"Train Batch - Hard Targets: {hard_targets.shape}")

    # Assertions to ensure data pipeline is correct
    assert images.dim() == 4, "Images must be 4D tensors (B, C, H, W)"
    assert images.shape[1] == 3, "Images must have 3 channels (RGB)"
    assert (
        hard_targets.shape[1] == Config.NUM_CLASSES
    ), f"Targets must have {Config.NUM_CLASSES} classes"
    assert rec_ids.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\n[Step 3] Initializing Model...")

    # Using ResNet18 for speed
    model = BirdModel(
        model_name="resnet18",
        pretrained=False,  # False for speed/offline demo
        num_classes=Config.NUM_CLASSES,
    ).to(device)

    # Validation: Check forward pass
    with torch.no_grad():
        dummy_input = images.to(device)
        logits = model(dummy_input)

    print(f"Model Output Logits: {logits.shape}")
    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"

    # -------------------------------------------------------------------------
    # 4. Optimizer & Loss Setup
    # -------------------------------------------------------------------------
    print("\n[Step 4] Setting up Optimizer (SAM) and Loss...")

    # Define Base Optimizer
    base_optimizer = torch.optim.AdamW

    # Wrap with SAM
    optimizer = SAM(
        model.parameters(),
        base_optimizer,
        lr=Config.LEARNING_RATE,
        rho=Config.SAM_RHO,
        adaptive=Config.SAM_ADAPTIVE,
    )

    # Define Loss Function (Weighted BCE)
    # Calculate dummy pos_weights for demonstration
    pos_weights = torch.ones(Config.NUM_CLASSES).to(device)
    loss_fn = WeightedBCELoss(pos_weights=pos_weights)

    # -------------------------------------------------------------------------
    # 5. Training Loop (1 Epoch)
    # -------------------------------------------------------------------------
    print("\n[Step 5] Running Training Loop...")

    train_loss = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        scheduler=None,  # No scheduler for this short demo
        loss_fn=loss_fn,
        device=device,
        epoch=1,
    )

    print(f"Training Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # -------------------------------------------------------------------------
    # 6. Validation Loop
    # -------------------------------------------------------------------------
    print("\n[Step 6] Running Validation Loop...")

    val_loss, val_auc = valid_one_epoch(
        model=model, loader=val_loader, loss_fn=loss_fn, device=device
    )

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")

    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0.0 <= val_auc <= 1.0, "AUC score out of range"

    # -------------------------------------------------------------------------
    # 7. Inference & Submission
    # -------------------------------------------------------------------------
    print("\n[Step 7] Running Inference and Generating Submission...")

    preds, ids = inference_fn(model=model, loader=test_loader, device=device)

    print(f"Predictions Shape: {preds.shape}")
    print(f"IDs Shape: {ids.shape}")

    assert preds.shape[0] == len(ids), "Number of predictions and IDs mismatch"
    assert preds.shape[1] == Config.NUM_CLASSES, "Prediction classes mismatch"

    # Save Submission
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    save_submission(preds, ids, submission_path)

    assert os.path.exists(submission_path), "Submission file was not created"

    # Verify file content format
    df_sub = pd.read_csv(submission_path)
    print("Submission Head:")
    print(df_sub.head())

    assert (
        "Id" in df_sub.columns and "Probability" in df_sub.columns
    ), "Submission columns invalid"
    assert (
        len(df_sub) == len(ids) * Config.NUM_CLASSES
    ), "Submission row count incorrect"

    # -------------------------------------------------------------------------
    # 8. Advanced Feature Demo: Anchor Distillation
    # -------------------------------------------------------------------------
    print("\n[Step 8] Demonstrating Anchor Distillation Loss...")

    # Create dummy soft targets for the training batch
    # In a real scenario, these come from pre-trained anchor models
    dummy_soft_targets = torch.rand(Config.BATCH_SIZE, Config.NUM_CLASSES).to(device)

    distillation_loss_fn = AnchorDistillationLoss(
        pos_weights=pos_weights, distillation_lambda=0.5
    )

    # Forward pass with distillation loss
    # Note: train_one_epoch handles the unpacking, here we test the loss function directly
    logits = model(images.to(device))
    loss_distill = distillation_loss_fn(
        student_logits=logits,
        hard_targets=hard_targets.to(device),
        soft_targets=dummy_soft_targets,
    )

    print(f"Distillation Loss: {loss_distill.item():.4f}")
    assert loss_distill.item() > 0, "Distillation loss should be positive"

    print("\nDemonstration Completed Successfully.")


if __name__ == "__main__":
    main()
