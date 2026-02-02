import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything
from library.dataset import (
    load_and_process_data,
    AppleDataset,
    get_transforms,
    MixupCutmix,
)
from library.model import AppleClassifier
from library.loss import AsymmetricLoss
from library.engine import fit, inference_tta


def main():
    print("=== Starting Apple Disease Detection Demo ===")

    # 1. Configuration & Setup
    # ----------------------
    cfg = Config()

    # Override configuration for a quick demonstration
    cfg.EPOCHS = 2
    cfg.BATCH_SIZE = 8
    cfg.NUM_WORKERS = 2
    cfg.WORKING_DIR = os.path.join(cfg.WORKING_DIR, "demo")
    os.makedirs(cfg.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(cfg.SEED)

    print(
        f"Configuration: Device={cfg.DEVICE}, Batch Size={cfg.BATCH_SIZE}, Epochs={cfg.EPOCHS}"
    )

    # 2. Data Loading & Processing
    # ----------------------------
    print("\n[Step 1] Loading and processing metadata...")

    # Load metadata using the library function
    # We use a unique cache name for the demo to avoid conflicts
    df_train_full = load_and_process_data(cfg.TRAIN_METADATA, "demo_train_cache")
    df_val_full = load_and_process_data(cfg.VAL_METADATA, "demo_val_cache")

    # Subsample datasets for speed (32 train, 16 val)
    df_train = df_train_full.iloc[:32].reset_index(drop=True)
    df_val = df_val_full.iloc[:16].reset_index(drop=True)

    print(f"Subsampled Train Data: {df_train.shape}")
    print(f"Subsampled Val Data:   {df_val.shape}")

    # Validate that class columns were created
    for cls_name in cfg.CLASSES:
        assert (
            cls_name in df_train.columns
        ), f"Class column {cls_name} missing in train df"

    # 3. Datasets & Dataloaders
    # -------------------------
    print("\n[Step 2] Creating Datasets and Dataloaders...")

    train_transforms = get_transforms("train", cfg)
    val_transforms = get_transforms("val", cfg)

    train_dataset = AppleDataset(df_train, transform=train_transforms)
    val_dataset = AppleDataset(df_val, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    # Verification: Check batch shapes
    images, targets = next(iter(train_loader))
    print(f"Batch Shapes -> Images: {images.shape}, Targets: {targets.shape}")
    assert images.shape == (cfg.BATCH_SIZE, 3, cfg.IMG_SIZE, cfg.IMG_SIZE)
    assert targets.shape == (cfg.BATCH_SIZE, cfg.NUM_CLASSES)

    # 4. Augmentation (Mixup/Cutmix)
    # ------------------------------
    print("\n[Step 3] Verifying Mixup/Cutmix...")
    mixup_fn = MixupCutmix(cfg)

    # Apply mixup to the fetched batch
    mixed_images, mixed_targets = mixup_fn(images, targets)

    # Verification: Shapes should remain the same
    assert mixed_images.shape == images.shape
    assert mixed_targets.shape == targets.shape
    print("Mixup/Cutmix applied successfully.")

    # 5. Model Initialization
    # -----------------------
    print("\n[Step 4] Initializing Model...")
    device = torch.device(cfg.DEVICE)
    model = AppleClassifier(
        model_name=cfg.MODEL_NAME, num_classes=cfg.NUM_CLASSES, pretrained=True
    )
    model.to(device)

    # Verification: Forward pass
    with torch.no_grad():
        dummy_output = model(images.to(device))
    assert dummy_output.shape == (cfg.BATCH_SIZE, cfg.NUM_CLASSES)
    print("Model forward pass successful.")

    # 6. Training Components
    # ----------------------
    print("\n[Step 5] Setting up Loss, Optimizer, and Scheduler...")
    loss_fn = AsymmetricLoss(
        gamma_neg=cfg.ASL_GAMMA_NEG, gamma_pos=cfg.ASL_GAMMA_POS, clip=cfg.ASL_CLIP
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=cfg.MIN_LR
    )

    # 7. Training Loop
    # ----------------
    print("\n[Step 6] Running Training Loop...")
    save_path = os.path.join(cfg.WORKING_DIR, "best_model_demo.pth")

    trained_model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=cfg.EPOCHS,
        mixup_fn=mixup_fn,
        loss_fn=loss_fn,
        patience=1,
        save_path=save_path,
    )

    # Verification: Check if model file exists
    assert os.path.exists(save_path), "Model checkpoint file was not created."
    print(f"Training finished. Best model saved to {save_path}")

    # 8. Inference
    # ------------
    print("\n[Step 7] Running Inference on Test Set...")

    # Load test metadata (subset)
    df_test_full = load_and_process_data(
        cfg.TEST_METADATA, "demo_test_cache", load_cached_data=False
    )
    df_test = df_test_full.iloc[:10].reset_index(drop=True)

    # Create test dataset with return_id=True
    test_dataset = AppleDataset(df_test, transform=val_transforms, return_id=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    submission_path = os.path.join(cfg.WORKING_DIR, "submission_demo.csv")

    # Run TTA Inference
    df_submission = inference_tta(
        model=trained_model,
        loader=test_loader,
        device=device,
        output_path=submission_path,
    )

    # Verification: Submission format
    print(f"Submission shape: {df_submission.shape}")
    print(df_submission.head())

    assert os.path.exists(submission_path), "Submission file was not created."
    assert len(df_submission) == 10, "Submission row count mismatch."
    assert list(df_submission.columns) == [
        "image",
        "labels",
    ], "Submission columns mismatch."

    # Check if labels are strings
    assert isinstance(df_submission.iloc[0]["labels"], str), "Labels should be strings."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
