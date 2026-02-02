import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import CFG
from library.utils import seed_everything, get_logger
from library.dataset import load_dataset_metadata, make_loader
from library.models import build_model
from library.engine import train_one_epoch, valid_one_epoch, predict_tta


def main():
    # 1. Setup
    print("Initializing demonstration...")
    seed_everything(CFG.seed)

    # Override CFG for speed in this demo
    CFG.epochs = 1
    CFG.image_size = 224  # Reduce size for faster processing
    CFG.batch_size = 16  # Small batch size for the demo subset

    device = CFG.device
    print(f"Device: {device}")

    # 2. Data Preparation
    print("\n--- Data Preparation ---")
    # Load full metadata
    df_train_full = load_dataset_metadata(CFG.train_csv)
    df_test_full = load_dataset_metadata(CFG.test_csv)

    # Subset for speed (Simulation of a small experiment)
    # We use 64 samples for train, 32 for valid
    df_train_subset = df_train_full.iloc[:64].reset_index(drop=True)
    df_valid_subset = df_train_full.iloc[64:96].reset_index(drop=True)

    print(f"Train subset shape: {df_train_subset.shape}")
    print(f"Valid subset shape: {df_valid_subset.shape}")

    # Create DataLoaders
    train_loader = make_loader(
        df_train_subset,
        image_size=CFG.image_size,
        batch_size=CFG.batch_size,
        is_train=True,
    )
    valid_loader = make_loader(
        df_valid_subset,
        image_size=CFG.image_size,
        batch_size=CFG.batch_size,
        is_train=False,
    )

    # Verification: Check DataLoader output
    images, labels = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")

    assert images.shape == (
        CFG.batch_size,
        3,
        CFG.image_size,
        CFG.image_size,
    ), "Incorrect image batch shape"
    assert labels.shape == (CFG.batch_size,), "Incorrect label batch shape"
    assert labels.dtype == torch.float32, "Labels should be float32"

    # 3. Model Initialization
    print("\n--- Model Initialization ---")
    model_name = CFG.model_names[0]  # Use the first model in the list
    print(f"Building model: {model_name}")

    model = build_model(model_name, pretrained=True)
    model.to(device)

    assert isinstance(model, nn.Module), "Model is not a torch.nn.Module"

    # 4. Training Loop
    print("\n--- Training Loop Demonstration ---")
    optimizer = optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr
    )

    # Train One Epoch
    avg_loss = train_one_epoch(
        model, optimizer, scheduler, train_loader, device, epoch=1
    )
    print(f"Training completed. Average Loss: {avg_loss:.4f}")
    assert isinstance(avg_loss, float), "Train loss should be a float"
    assert avg_loss > 0, "Train loss should be positive"

    # Validate One Epoch
    val_loss, val_log_loss, val_acc, val_preds = valid_one_epoch(
        model, valid_loader, device
    )
    print(f"Validation completed. LogLoss: {val_log_loss:.4f}, Accuracy: {val_acc:.4f}")

    assert len(val_preds) == len(
        df_valid_subset
    ), "Number of predictions does not match validation set size"
    assert val_preds.shape == (len(df_valid_subset), 1), "Prediction shape mismatch"
    assert 0.0 <= val_acc <= 1.0, "Accuracy should be between 0 and 1"

    # 5. Inference (TTA)
    print("\n--- Inference Demonstration (TTA) ---")
    # Subset test data
    df_test_subset = df_test_full.iloc[:32].reset_index(drop=True)

    test_loader = make_loader(
        df_test_subset,
        image_size=CFG.image_size,
        batch_size=CFG.batch_size,
        is_train=False,
    )

    test_preds, test_ids = predict_tta(model, test_loader, device)

    print(f"Inference completed. Predictions shape: {test_preds.shape}")

    # Verification
    assert len(test_preds) == len(df_test_subset), "Number of test predictions mismatch"
    assert (
        test_preds.min() >= 0.0 and test_preds.max() <= 1.0
    ), "Probabilities must be in [0, 1]"
    assert np.array_equal(test_ids, df_test_subset["id"].values), "Test IDs mismatch"

    # 6. Submission File
    print("\n--- Generating Submission ---")
    submission = pd.DataFrame({"id": test_ids, "label": test_preds.flatten()})

    # Save to working directory
    sub_path = os.path.join(CFG.working_dir, "demo_submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")

    # Verify file existence
    assert os.path.exists(sub_path), "Submission file was not created"

    print("\nDemonstration finished successfully.")


if __name__ == "__main__":
    main()
