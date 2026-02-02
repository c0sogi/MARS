import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import (
    seed_everything,
    get_pos_weight,
    save_oof_preds,
    load_oof_preds,
)
from library.dataset import BirdDataset, get_transforms
from library.models import BirdModel
from library.loss import BornAgainLoss
from library.engine import train_one_epoch, validate, predict


def main():
    print("Starting Demo Script...")

    # 1. Setup Environment
    # --------------------
    # Initialize configuration and directories
    Config.setup()

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    # Override Config parameters for a fast demonstration
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    Config.NUM_WORKERS = 0  # Disable multi-processing for simple script execution

    # Use 'resnet18' as it is lightweight
    Config.MODEL_RESNET = "resnet18"

    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Data Preparation
    # -------------------
    # Load training metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata file not found: {Config.TRAIN_METADATA_PATH}"
        )

    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Select a small subset (e.g., 16 samples) for quick execution
    df_subset = df_train.head(16).copy().reset_index(drop=True)
    print(f"Using subset of {len(df_subset)} samples.")

    # Calculate positive weights for class imbalance handling
    pos_weights = get_pos_weight(df_subset, device)

    # Create Transforms
    train_transforms = get_transforms(mode="train")
    val_transforms = get_transforms(mode="val")  # No random shifting

    # Create Dataset and DataLoader
    train_dataset = BirdDataset(df_subset, transforms=train_transforms, mode="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Model Initialization
    # -----------------------
    print("Initializing BirdModel (ResNet18)...")
    # Using pretrained=False to ensure it runs without internet access
    model = BirdModel(
        model_name="resnet18", num_classes=Config.NUM_CLASSES, pretrained=False
    )
    model.to(device)

    # 4. Training Loop Demonstration
    # ------------------------------
    print("Testing Training Loop...")
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    loss_fn = BornAgainLoss(pos_weight=pos_weights)

    # Run one epoch
    train_loss = train_one_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        device=device,
        epoch=0,
        loss_fn=loss_fn,
    )
    print(f"Training Epoch Loss: {train_loss:.4f}")

    # Verification
    if not np.isfinite(train_loss):
        raise AssertionError("Training loss is not finite.")

    # 5. Validation Loop Demonstration
    # --------------------------------
    print("Testing Validation Loop...")
    val_loss, val_auc = validate(
        model=model,
        dataloader=train_loader,  # Using train loader as proxy for val loader
        device=device,
        loss_fn=loss_fn,
    )
    print(f"Validation Loss: {val_loss:.4f}, ROC AUC: {val_auc:.4f}")

    # 6. Inference Demonstration
    # --------------------------
    print("Testing Prediction...")
    rec_ids, preds = predict(model, train_loader, device)

    # Verification
    assert len(rec_ids) == len(
        df_subset
    ), "Number of predictions does not match dataset size."
    assert preds.shape == (
        len(df_subset),
        Config.NUM_CLASSES,
    ), "Prediction shape mismatch."
    print("Prediction shape verified.")

    # 7. OOF Utilities & Distillation Demonstration
    # ---------------------------------------------
    print("Testing OOF Utilities and Distillation...")

    # Define a temporary path for OOF predictions
    oof_save_path = os.path.join(Config.WORKING_DIR, "demo_oof_preds.parquet")

    # Save predictions
    save_oof_preds(preds, rec_ids, oof_save_path)
    if not os.path.exists(oof_save_path):
        raise AssertionError("OOF file was not created.")

    # Load predictions back
    loaded_preds = load_oof_preds(oof_save_path, rec_ids)

    # Verify consistency
    if not np.allclose(preds, loaded_preds, atol=1e-6):
        raise AssertionError("Loaded OOF predictions do not match saved predictions.")
    print("OOF Save/Load cycle verified.")

    # Create a dataset with soft labels (Distillation)
    distill_dataset = BirdDataset(
        df_subset,
        transforms=train_transforms,
        soft_labels_path=oof_save_path,
        mode="train",
    )
    distill_loader = DataLoader(
        distill_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Run training step with soft targets
    # BornAgainLoss automatically handles the soft targets if present in the batch
    distill_loss = train_one_epoch(
        model=model,
        dataloader=distill_loader,
        optimizer=optimizer,
        device=device,
        epoch=0,
        loss_fn=loss_fn,
    )
    print(f"Distillation Epoch Loss: {distill_loss:.4f}")

    if not np.isfinite(distill_loss):
        raise AssertionError("Distillation loss is not finite.")

    print("-" * 30)
    print("All demonstrations completed successfully.")


if __name__ == "__main__":
    main()
