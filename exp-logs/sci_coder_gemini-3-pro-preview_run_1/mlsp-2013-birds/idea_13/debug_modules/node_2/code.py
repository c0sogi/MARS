import os
import sys
import shutil
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, AverageMeter, save_checkpoint, load_checkpoint
from library.data import get_dataloaders
from library.model import SEResNet34
from library.engine import (
    train_one_epoch,
    valid_one_epoch,
    inference_fn,
    generate_submission,
    SWAManager,
    mixup_data,
)


def main():
    print("Initializing Demonstration...")

    # 1. Setup & Configuration Override for Speed
    # We modify the global Config to run a lightweight version of the pipeline
    Config.WORKING_DIR = "./working/demo_run"
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8  # Small batch size for demonstration
    Config.SWA_START_EPOCH_TEACHER = 1  # Start SWA immediately after epoch 0
    Config.SWA_LR = 1e-4

    # Ensure working directory exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)

    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading Demonstration
    print("\n--- Testing Data Loading ---")
    train_loader, val_loader, test_loader = get_dataloaders()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Fetch a single batch to verify shapes
    images, labels, rec_ids = next(iter(train_loader))

    # Assertions for data integrity
    assert images.dim() == 4, f"Expected 4D image tensor, got {images.dim()}"
    assert images.shape[1] == 3, f"Expected 3 channels (RGB), got {images.shape[1]}"
    assert images.shape[2] == Config.IMG_HEIGHT, f"Height mismatch: {images.shape[2]}"
    assert images.shape[3] == Config.IMG_WIDTH, f"Width mismatch: {images.shape[3]}"
    assert (
        labels.shape[1] == Config.NUM_CLASSES
    ), f"Label class count mismatch: {labels.shape[1]}"
    assert len(rec_ids) == Config.BATCH_SIZE, "Batch size mismatch in rec_ids"

    print("Data shapes verified successfully.")
    print(f"Image Batch Shape: {images.shape}")
    print(f"Label Batch Shape: {labels.shape}")

    # 3. Model Instantiation & Forward Pass
    print("\n--- Testing Model Architecture ---")
    model = SEResNet34(pretrained=False)  # False for speed, usually True
    model.to(device)

    # Test forward pass with the fetched batch
    with torch.no_grad():
        logits = model(images.to(device))

    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {logits.shape}"

    print("Model forward pass successful.")

    # 4. Training Loop & SWA Demonstration
    print("\n--- Testing Training Loop & SWA ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Initialize SWA Manager
    swa_manager = SWAManager(
        model=model,
        optimizer=optimizer,
        swa_start_epoch=Config.SWA_START_EPOCH_TEACHER,
        swa_lr=Config.SWA_LR,
        device=device,
    )

    best_auc = 0.0

    for epoch in range(Config.EPOCHS):
        print(f"\n[Epoch {epoch}]")

        # Train
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            use_mixup=True,
        )

        # SWA Step (updates SWA model if epoch >= start_epoch)
        swa_manager.step(epoch, model)

        # Validation
        val_loss, val_auc = valid_one_epoch(
            model=model, loader=val_loader, device=device, epoch=epoch
        )

        # Checkpoint logic
        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_auc": best_auc,
            },
            is_best=is_best,
            filename=f"checkpoint_epoch_{epoch}.pth",
        )

        # Verify metrics are valid numbers
        assert not np.isnan(train_loss), "Training loss is NaN"
        assert not np.isnan(val_loss), "Validation loss is NaN"
        assert 0.0 <= val_auc <= 1.0, "AUC score out of range"

    # Update SWA Batch Norms
    print("\nUpdating SWA Batch Norm statistics...")
    swa_manager.update_bn(train_loader)

    # Verify SWA model weights are different/exist
    assert swa_manager.swa_model is not None, "SWA model not initialized"
    print("SWA update successful.")

    # 5. Inference Demonstration
    print("\n--- Testing Inference ---")
    # Use the best model for inference
    best_model_path = os.path.join(Config.WORKING_DIR, "model_best.pth")
    if os.path.exists(best_model_path):
        checkpoint = load_checkpoint(model, "model_best.pth", device=device)
        print(f"Loaded best model with AUC: {checkpoint.get('best_auc', 0.0):.4f}")

    # Run inference on validation set (as a proxy for test to verify logic)
    # We use valid_loader here just to check the function works
    probs, r_ids = inference_fn(model, val_loader, device=device, use_tta=True)

    assert probs.shape[1] == Config.NUM_CLASSES, "Inference probability shape mismatch"
    assert len(probs) == len(r_ids), "Mismatch between probabilities and rec_ids count"
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of [0, 1] range"

    print(f"Inference output shape: {probs.shape}")

    # 6. Submission Generation
    print("\n--- Testing Submission Generation ---")
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    generate_submission(probs, r_ids, submission_path)

    # Verify submission file
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    expected_rows = len(r_ids) * Config.NUM_CLASSES
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"
    assert (
        "Id" in df_sub.columns and "Probability" in df_sub.columns
    ), "Submission columns missing"

    print("Submission file verified.")

    print("\n============================================")
    print("       DEMONSTRATION COMPLETE               ")
    print("============================================")


if __name__ == "__main__":
    main()
