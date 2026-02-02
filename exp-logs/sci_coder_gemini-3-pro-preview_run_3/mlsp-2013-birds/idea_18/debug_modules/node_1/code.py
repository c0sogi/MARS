import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shutil
import glob

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, RobustAUC
from library.data import get_folds, get_dataloaders, get_test_dataloader, BirdDataset
from library.models import ModelFactory
from library.trainer import Trainer
from library.inference import InferenceEngine


def run_demo():
    print("=" * 50)
    print("Starting Bird Classification Library Demo")
    print("=" * 50)

    # ---------------------------------------------------------
    # 1. Setup and Configuration Override
    # ---------------------------------------------------------
    print("\n[Step 1] Configuring environment for demo...")

    # Override Config for speed and isolation
    Config.PROJECT_NAME = "demo_run"
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Reduce computational load
    Config.NUM_EPOCHS = 1
    Config.NUM_FOLDS = 2  # Use 2 folds to demonstrate the loop, but keep it fast
    Config.BATCH_SIZE = 8
    Config.ARCHITECTURES = ["resnet18"]  # Only use the smallest model
    Config.TOP_K_CHECKPOINTS = 1

    # Ensure directories exist (since we changed the paths after import)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print("Configuration updated. Working directory:", Config.WORKING_DIR)

    # ---------------------------------------------------------
    # 2. Data Pipeline Verification
    # ---------------------------------------------------------
    print("\n[Step 2] Verifying Data Pipeline...")

    # Test get_folds
    print("Generating/Loading folds...")
    folds_df = get_folds(load_cached_data=False)  # Force regeneration
    assert "fold" in folds_df.columns, "Folds DataFrame missing 'fold' column"
    assert len(folds_df) > 0, "Folds DataFrame is empty"
    print(f"Folds generated. Total samples: {len(folds_df)}")

    # Test DataLoaders for Fold 0
    print("Creating DataLoaders for Fold 0...")
    train_loader, val_loader = get_dataloaders(fold_id=0, load_cached_data=True)

    # Fetch a single batch to verify shapes
    images, labels = next(iter(train_loader))
    print(f"Train Batch - Images: {images.shape}, Labels: {labels.shape}")

    # Assertions
    assert images.dim() == 4, "Images should be 4D tensors (B, C, H, W)"
    assert images.shape[1] == 3, "Images should have 3 channels"
    assert (
        images.shape[2] == Config.IMG_SIZE[0] and images.shape[3] == Config.IMG_SIZE[1]
    ), f"Image size mismatch. Expected {Config.IMG_SIZE}, got {images.shape[2:]}"
    assert (
        labels.shape[1] == Config.NUM_CLASSES
    ), f"Label classes mismatch. Expected {Config.NUM_CLASSES}, got {labels.shape[1]}"

    print("Data Pipeline verification successful.")

    # ---------------------------------------------------------
    # 3. Model Instantiation Verification
    # ---------------------------------------------------------
    print("\n[Step 3] Verifying Model Factory...")

    model = ModelFactory.create_model(
        "resnet18", num_classes=Config.NUM_CLASSES, pretrained=True
    )
    model.to(Config.DEVICE)

    # Test forward pass
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, 224, 224).to(Config.DEVICE)
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, Config.NUM_CLASSES), "Model output shape is incorrect"
    print("Model instantiated and tested successfully.")

    # ---------------------------------------------------------
    # 4. Metric Verification (RobustAUC)
    # ---------------------------------------------------------
    print("\n[Step 4] Verifying RobustAUC Metric...")

    metric = RobustAUC()

    # Simulate a scenario: 3 classes.
    # Class 0: Good predictions
    # Class 1: Constant ground truth (should be skipped)
    # Class 2: Random
    y_true = np.array([[0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0]])
    y_pred = np.array(
        [[0.1, 0.5, 0.8], [0.9, 0.5, 0.2], [0.2, 0.5, 0.9], [0.8, 0.5, 0.1]]
    )

    metric.update(y_pred, y_true)
    score = metric.compute()
    print(f"Computed RobustAUC Score: {score:.4f}")
    # Class 1 is constant (0), so it should be skipped.
    # Class 0 and 2 should have high AUC (perfect separation in this dummy data).
    # Expected AUC should be 1.0.
    assert score == 1.0, f"RobustAUC logic failed. Expected 1.0, got {score}"
    print("RobustAUC verification successful.")

    # ---------------------------------------------------------
    # 5. Training Loop Verification
    # ---------------------------------------------------------
    print("\n[Step 5] Running Training Loop (2 Folds, 1 Epoch)...")

    # We will train fold 0 and fold 1 to ensure InferenceEngine has what it needs
    for fold in range(Config.NUM_FOLDS):
        print(f"--- Training Fold {fold} ---")

        # Re-init model and loaders for each fold
        train_loader, val_loader = get_dataloaders(fold_id=fold, load_cached_data=True)
        model = ModelFactory.create_model(
            "resnet18", num_classes=Config.NUM_CLASSES, pretrained=True
        )
        model.to(Config.DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.NUM_EPOCHS
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=Config.DEVICE,
            fold_id=fold,
            architecture_name="resnet18",
        )

        # Fit model
        checkpoints = trainer.fit(
            train_loader, val_loader, num_epochs=Config.NUM_EPOCHS
        )

        assert len(checkpoints) > 0, f"No checkpoints saved for fold {fold}"
        assert os.path.exists(
            checkpoints[0]
        ), f"Checkpoint file missing: {checkpoints[0]}"
        print(f"Fold {fold} complete. Best checkpoint: {checkpoints[0]}")

    print("Training loop verification successful.")

    # ---------------------------------------------------------
    # 6. Inference and Submission Verification
    # ---------------------------------------------------------
    print("\n[Step 6] Verifying Inference Engine and Submission...")

    inference_engine = InferenceEngine()

    # This will load checkpoints from Config.CHECKPOINT_DIR, average them (if applicable),
    # run TTA, and generate submission.csv
    inference_engine.generate_submission()

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    # Verify Submission Content
    sub_df = pd.read_csv(submission_path)
    print(f"Submission generated with {len(sub_df)} rows.")

    # Check format
    assert list(sub_df.columns) == ["Id", "Probability"], "Submission columns mismatch"
    assert (
        sub_df["Probability"].min() >= 0 and sub_df["Probability"].max() <= 1
    ), "Probabilities out of range"

    # Check against sample submission length (Test set size * 19 classes)
    # Test set size is 64 (from metadata/test.csv)
    # 64 * 19 = 1216 rows
    expected_rows = 64 * Config.NUM_CLASSES
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    print("Inference verification successful.")

    print("\n" + "=" * 50)
    print("Demo execution completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    run_demo()
