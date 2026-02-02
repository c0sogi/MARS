import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys

# Import from the provided library files
from library.config import (
    DEVICE,
    NUM_CLASSES,
    BATCH_SIZE,
    setup_directories,
    SUBMISSION_PATH,
    WORKING_DIR,
)
from library.utils import seed_everything
from library.dataset import get_loaders, get_test_loader
from library.models import CactusRepVGG_DS, CactusResNet_DS
from library.engine import train_one_epoch, evaluate, predict_tta


def run_demo():
    print(f"Running on device: {DEVICE}")

    # 1. Setup and Initialization
    # ---------------------------
    print("\n[1] Setting up environment...")
    seed_everything(42)
    setup_directories()

    # 2. Data Loading Verification
    # ----------------------------
    print("\n[2] Verifying Data Loaders...")
    # We use a slightly smaller batch size for the demo to ensure we see multiple batches if data is small
    demo_batch_size = 64
    train_loader, val_loader = get_loaders(fold_idx=0, batch_size=demo_batch_size)

    # Fetch a single batch to verify shapes and types
    imgs, labels = next(iter(train_loader))

    print(f"    Batch Image Shape: {imgs.shape}")
    print(f"    Batch Label Shape: {labels.shape}")

    # Assertions
    assert imgs.shape == (demo_batch_size, 3, 32, 32), "Incorrect image batch shape"
    assert labels.shape == (demo_batch_size,), "Incorrect label batch shape"
    assert imgs.dtype == torch.float32, "Images should be float32"
    # Check normalization roughly (standardized data should have mean ~0, std ~1)
    # Since we normalized with specific mean/std, values can be negative.
    assert (
        imgs.max() > 1.0 or imgs.min() < 0.0
    ), "Images do not appear to be normalized (expected range outside [0,1])"

    print("    Data loading verification passed.")

    # 3. Model Architecture Verification
    # ----------------------------------
    print("\n[3] Verifying Model Architectures...")

    # Instantiate models
    repvgg = CactusRepVGG_DS(num_classes=NUM_CLASSES).to(DEVICE)
    resnet = CactusResNet_DS(num_classes=NUM_CLASSES).to(DEVICE)

    # Move dummy batch to device
    dummy_input = imgs.to(DEVICE)

    # Test RepVGG Forward Pass (Training Mode -> Returns Main + Aux)
    repvgg.train()
    main_out, aux_out = repvgg(dummy_input)
    assert main_out.shape == (
        demo_batch_size,
        NUM_CLASSES,
    ), "RepVGG Main output shape mismatch"
    assert aux_out.shape == (
        demo_batch_size,
        NUM_CLASSES,
    ), "RepVGG Aux output shape mismatch"

    # Test ResNet Forward Pass (Training Mode)
    resnet.train()
    main_out_res, aux_out_res = resnet(dummy_input)
    assert main_out_res.shape == (
        demo_batch_size,
        NUM_CLASSES,
    ), "ResNet Main output shape mismatch"

    print("    Model architecture verification passed.")

    # 4. Training Loop Demonstration
    # ------------------------------
    print("\n[4] Starting Training Demo (CactusRepVGG_DS)...")

    # We will use the RepVGG model for the training demo
    model = repvgg
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    # Run for 2 epochs to demonstrate the loop and metric calculation
    demo_epochs = 2

    for epoch in range(1, demo_epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)

        # Evaluate
        val_loss, val_auc = evaluate(model, val_loader, criterion, DEVICE)

        print(
            f"    Epoch {epoch}/{demo_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )

        # Basic sanity check: Loss should not be NaN
        if np.isnan(train_loss) or np.isnan(val_loss):
            raise ValueError("Loss is NaN, training failed.")

    print("    Training demonstration complete.")

    # 5. Inference and TTA Verification
    # ---------------------------------
    print("\n[5] Verifying Inference and Test Time Augmentation (TTA)...")

    test_loader = get_test_loader(batch_size=demo_batch_size)

    # Verify RepVGG structural re-parameterization switch
    # The predict_tta function handles the switch_to_deploy call internally
    assert model.deploy is False, "Model should be in training mode before inference"

    preds, ids = predict_tta(model, test_loader, DEVICE)

    # Verify model was switched to deploy mode
    # Note: If model was wrapped in DataParallel or AveragedModel, we'd check the inner module.
    # Here it's direct.
    assert model.deploy is True, "Model should be in deploy mode after inference"

    print(f"    Predictions shape: {preds.shape}")
    print(f"    IDs count: {len(ids)}")

    assert len(preds) == len(ids), "Mismatch between predictions and IDs"
    assert len(preds) == 3325, f"Expected 3325 test samples, got {len(preds)}"
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Predictions not in probability range [0, 1]"

    print("    Inference verification passed.")

    # 6. Submission Generation
    # ------------------------
    print("\n[6] Generating Submission File...")

    submission_df = pd.DataFrame({"id": ids, "has_cactus": preds})

    # Ensure output directory exists (handled by setup_directories, but good to be safe)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(SUBMISSION_PATH, index=False)

    # Verify file creation
    if os.path.exists(SUBMISSION_PATH):
        print(f"    Submission saved to: {SUBMISSION_PATH}")
        print("    First 5 rows:")
        print(submission_df.head().to_string())
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    run_demo()
