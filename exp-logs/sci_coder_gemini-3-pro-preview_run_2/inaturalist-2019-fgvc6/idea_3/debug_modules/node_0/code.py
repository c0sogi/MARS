import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path

# Import from the provided library files
from library.config import CFG
from library.utils import seed_everything, save_checkpoint
from library.dataset import get_loaders
from library.model import build_model
from library.engine import train_one_epoch, validate


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Configure for Speed/Debug
    # We override the default configuration to run a quick sanity check
    print("\n[Step 1] Configuring environment for fast demonstration...")
    CFG.debug = True
    CFG.debug_sample_size = 50  # Use only 50 images per split
    CFG.batch_size = 8  # Small batch size
    CFG.epochs = 1  # Only 1 epoch
    CFG.num_workers = 2  # Reduce workers to minimize overhead for small data
    CFG.output_dir = "./working/demo_run"
    CFG.setup()  # Ensure directories exist

    # Set random seed for reproducibility
    seed_everything(CFG.seed)
    print(
        f"Configuration set: Debug={CFG.debug}, Device={CFG.device}, Output Dir={CFG.output_dir}"
    )

    # 2. Data Loading Demonstration
    print("\n[Step 2] Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_loaders()

    # Verification: Check loader lengths
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    if len(train_loader) == 0:
        raise RuntimeError(
            "Train loader is empty. Check dataset path or debug_sample_size."
        )

    # Verification: Inspect a single training batch
    images, targets = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    # Assertions for data integrity
    # Expected shape: [Batch_Size, 3, Train_Size, Train_Size]
    assert images.shape == (
        CFG.batch_size,
        3,
        CFG.train_size,
        CFG.train_size,
    ), f"Incorrect image shape: {images.shape}"
    assert targets.shape == (
        CFG.batch_size,
    ), f"Incorrect target shape: {targets.shape}"
    assert targets.max() < CFG.num_classes, "Target label exceeds number of classes"
    print("Data loading verification passed.")

    # 3. Model Construction Demonstration
    print("\n[Step 3] Building Model...")
    model = build_model()
    model = model.to(CFG.device)

    # Verification: Check model output shape
    # We use the batch fetched in Step 2
    with torch.no_grad():
        dummy_input = images.to(CFG.device)
        dummy_output = model(dummy_input)

    print(f"Model Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (
        CFG.batch_size,
        CFG.num_classes,
    ), f"Model output shape mismatch. Expected {(CFG.batch_size, CFG.num_classes)}, got {dummy_output.shape}"
    print("Model construction verification passed.")

    # 4. Training Engine Demonstration
    print("\n[Step 4] Running Training Loop (1 Epoch)...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )

    # Run training for one epoch
    train_loss, train_acc = train_one_epoch(
        model, optimizer, train_loader, CFG.device, epoch=0
    )

    print(f"Training completed. Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

    # Assertions for training metrics
    assert not np.isnan(train_loss), "Training loss is NaN"
    assert 0 <= train_acc <= 100, "Training accuracy is out of bounds (0-100)"
    print("Training engine verification passed.")

    # 5. Validation Engine Demonstration
    print("\n[Step 5] Running Validation Loop...")
    val_loss, val_acc = validate(model, val_loader, CFG.device)

    print(f"Validation completed. Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

    # Assertions for validation metrics
    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0 <= val_acc <= 100, "Validation accuracy is out of bounds (0-100)"
    print("Validation engine verification passed.")

    # 6. Checkpointing Demonstration
    print("\n[Step 6] Saving Checkpoint...")
    state = {
        "epoch": 0,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_acc": val_acc,
    }

    save_checkpoint(state, is_best=True, output_dir=CFG.output_dir)

    # Verification: Check if files were created
    checkpoint_path = os.path.join(CFG.output_dir, "checkpoint.pth")
    best_model_path = os.path.join(CFG.output_dir, "model_best.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model copy not found at {best_model_path}")

    print(f"Checkpoint saved successfully at {checkpoint_path}")

    # 7. Inference Demonstration (Test Set)
    print("\n[Step 7] Generating Test Predictions (Subset)...")
    model.eval()
    predictions = []
    test_ids = []

    # Run inference on a few batches
    with torch.no_grad():
        for i, (images, image_ids) in enumerate(test_loader):
            images = images.to(CFG.device)
            outputs = model(images)

            # Get top 5 predictions
            _, preds = torch.topk(outputs, 5)

            predictions.append(preds.cpu().numpy())
            test_ids.append(image_ids.numpy())

            if i >= 2:
                break  # Limit to a few batches for speed

    predictions = np.concatenate(predictions)
    test_ids = np.concatenate(test_ids)

    # Create submission DataFrame format
    pred_strings = [" ".join(map(str, row)) for row in predictions]
    submission_df = pd.DataFrame({"id": test_ids, "predicted": pred_strings})

    print("Sample Submission Data:")
    print(submission_df.head())

    submission_path = os.path.join(CFG.output_dir, "demo_submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file generated at {submission_path}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
