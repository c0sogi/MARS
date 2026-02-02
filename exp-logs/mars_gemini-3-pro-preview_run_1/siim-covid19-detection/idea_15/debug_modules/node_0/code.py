import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import SIIMDataset
from library.model import DropBlockResNet34UNet
from library.engine import train_one_epoch, evaluate, predict_test


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # -------------------------------------------------------------------------
    print("[Demo] Setting up configuration for rapid execution...")

    # Set paths
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Ensure working directory exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Override Config for speed and deterministic behavior
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 8  # Small subset for demo
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.PRETRAINED = False  # Skip downloading weights for speed
    Config.IMG_SIZE = 256  # Reduce image size for faster processing

    # Initialize environment
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # -------------------------------------------------------------------------
    # 2. Dataset Instantiation & Verification
    # -------------------------------------------------------------------------
    print("\n[Demo] Verifying Dataset...")

    # Instantiate Train Dataset
    train_dataset = SIIMDataset("train", load_cached_data=False)

    # Apply Debug Subsetting manually to match Config.DEBUG logic if it were inside the class
    # (The provided dataset class loads full metadata, so we slice it here for the demo)
    train_dataset.df = train_dataset.df.iloc[: Config.DEBUG_SAMPLE_SIZE].copy()

    print(f"Train Dataset Length: {len(train_dataset)}")

    # Assertions
    assert (
        len(train_dataset) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} samples, got {len(train_dataset)}"

    # Fetch one sample
    sample = train_dataset[0]
    image = sample["image"]
    mask = sample["mask"]
    labels = sample["labels"]

    print(f"Sample Image Shape: {image.shape}")
    print(f"Sample Mask Shape: {mask.shape}")
    print(f"Sample Labels: {labels}")

    # Verify Shapes
    # Image: (3, H, W) -> Albumentations ToTensorV2 produces (C, H, W)
    assert image.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch. Expected (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {image.shape}"

    # Mask: (1, H, W)
    assert mask.shape == (
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Mask shape mismatch. Expected (1, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {mask.shape}"

    # Labels: (4,)
    assert labels.shape == (
        4,
    ), f"Labels shape mismatch. Expected (4,), got {labels.shape}"

    # -------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[Demo] Initializing Model and checking Forward Pass...")

    model = DropBlockResNet34UNet().to(device)

    # Create a small batch
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    batch = next(iter(train_loader))
    images = batch["image"].to(device)

    # Forward pass
    cls_logits, seg_logits = model(images)

    print(f"Class Logits Shape: {cls_logits.shape}")
    print(f"Seg Logits Shape: {seg_logits.shape}")

    # Assert Output Shapes
    # Classification: (Batch_Size, Num_Classes)
    assert cls_logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Classification logits shape incorrect."

    # Segmentation: (Batch_Size, 1, H, W)
    # The model outputs segmentation at the same spatial resolution as input due to U-Net architecture
    assert seg_logits.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Segmentation logits shape incorrect."

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[Demo] Running Training Loop (1 Epoch)...")

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # Run one epoch
    loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch=0)

    print(f"Training finished. Loss: {loss:.4f}")
    assert not np.isnan(loss), "Training Loss is NaN!"

    # -------------------------------------------------------------------------
    # 5. Evaluation Demonstration
    # -------------------------------------------------------------------------
    print("\n[Demo] Running Evaluation...")

    # Setup Validation Dataset (Subset)
    val_dataset = SIIMDataset("val", load_cached_data=False)
    val_dataset.df = val_dataset.df.iloc[: Config.DEBUG_SAMPLE_SIZE].copy()

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Run evaluation
    # Note: evaluate() uses prepare_gt_from_metadata which needs the dataframe passed to it
    # to match the predictions generated.
    map_score = evaluate(model, val_loader, device, val_dataset.df)

    print(f"Evaluation finished. mAP: {map_score:.6f}")
    assert isinstance(map_score, float), "mAP score should be a float."

    # -------------------------------------------------------------------------
    # 6. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n[Demo] Running Inference on Test Set...")

    # Setup Test Dataset (Subset)
    test_dataset = SIIMDataset("test", load_cached_data=False)
    # Test set might be small, but let's limit it just in case
    test_dataset.df = test_dataset.df.iloc[
        : min(len(test_dataset), Config.DEBUG_SAMPLE_SIZE)
    ].copy()

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Run prediction
    predict_test(model, test_loader, device, test_dataset.df)

    # Check if submission file exists
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file was not created at {Config.SUBMISSION_PATH}"

    # Verify submission content format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file created with {len(sub_df)} rows.")
    print("Head of submission:")
    print(sub_df.head())

    # Assert columns
    assert (
        "Id" in sub_df.columns and "PredictionString" in sub_df.columns
    ), "Submission file missing required columns."

    print("\n[Demo] All demonstrations completed successfully.")


if __name__ == "__main__":
    main()
