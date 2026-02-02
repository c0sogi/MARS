import os
import torch
import pandas as pd
import numpy as np
import shutil

# 1. Configuration Override for Speed and Demo
from library.config import Config

# Modify Config for a fast demonstration run
Config.DEBUG = True
Config.DEBUG_SUBSET_SIZE = 100  # Use only 100 images for speed
Config.WORKING_DIR = "./working/demo_run"  # Separate working dir for demo
Config.WARMUP_EPOCHS = 1
Config.FINE_TUNE_EPOCHS = 1
Config.N_FOLDS = 1  # Only run 1 fold
Config.BATCH_SIZE = 8  # Small batch size for demo
Config.NUM_WORKERS = 2  # Reduce workers to avoid overhead in small demo

# Import library modules after config modification
from library.utils import seed_everything, get_logger
from library.dataset import get_dataloaders
from library.model import get_model
from library.engine import train_fold
from library.inference import run_inference


def run_demo():
    print("=== Starting Library Demo Script ===")

    # Setup
    seed_everything(Config.SEED)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    logger = get_logger("demo_script")

    # ==========================================
    # 2. Dataset Verification
    # ==========================================
    logger.info("--- Testing Dataset Loading ---")

    # Initialize DataLoaders
    dataloaders = get_dataloaders(debug=Config.DEBUG)

    # Verify keys
    assert "train" in dataloaders
    assert "val" in dataloaders
    assert "test" in dataloaders
    assert "class_to_idx" in dataloaders

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # Verify Batch Shape
    images, labels, ids = next(iter(train_loader))

    logger.info(f"Train Batch Image Shape: {images.shape}")
    logger.info(f"Train Batch Label Shape: {labels.shape}")

    # Assertions
    # Shape: (Batch_Size, Channels, Height, Width)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Expected image shape {(Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)}, got {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Expected label shape {(Config.BATCH_SIZE,)}, got {labels.shape}"
    assert len(ids) == Config.BATCH_SIZE

    logger.info("Dataset verification successful.")

    # ==========================================
    # 3. Model Verification
    # ==========================================
    logger.info("--- Testing Model Initialization & Forward Pass ---")

    device = Config.DEVICE
    model = get_model(device=device, pretrained=True)

    # Test Forward Pass
    images = images.to(device)
    outputs = model(images)

    logger.info(f"Model Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Expected output shape {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {outputs.shape}"

    # Test Backbone Freezing Logic
    logger.info("Testing backbone freezing...")
    model.set_backbone_trainable(False)
    trainable_params, _ = model.get_param_count()
    logger.info(f"Trainable params (Frozen Backbone): {trainable_params}")

    # We expect significantly fewer trainable params when frozen
    # Just checking it's not 0 and less than total
    _, total_params = model.get_param_count()
    assert trainable_params < total_params
    assert trainable_params > 0

    model.set_backbone_trainable(True)
    trainable_params_unfrozen, _ = model.get_param_count()
    logger.info(f"Trainable params (Unfrozen Backbone): {trainable_params_unfrozen}")
    assert trainable_params_unfrozen == total_params

    logger.info("Model verification successful.")

    # ==========================================
    # 4. Training Loop Execution
    # ==========================================
    logger.info("--- Testing Training Loop (Fold 0) ---")

    # Run training for Fold 0
    # This covers: Warmup, Fine-tuning, Checkpointing, Validation
    best_loss = train_fold(
        fold_idx=0,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    logger.info(f"Training completed. Best Loss: {best_loss}")

    # Verify Checkpoints exist
    expected_checkpoint = os.path.join(Config.WORKING_DIR, "best_model_fold_0.pth")
    assert os.path.exists(
        expected_checkpoint
    ), f"Checkpoint not found at {expected_checkpoint}"

    logger.info("Training loop verification successful.")

    # ==========================================
    # 5. Inference Execution
    # ==========================================
    logger.info("--- Testing Inference Pipeline ---")

    # Run inference
    # This uses the checkpoint generated above
    run_inference()

    # Verify Submission File
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    df_sub = pd.read_csv(submission_path)
    logger.info(f"Submission shape: {df_sub.shape}")

    # Assertions
    # Rows should match test set size (Config.DEBUG_SUBSET_SIZE is applied to test set too in DogDataset)
    # However, DogDataset debug logic takes min(len(df), DEBUG_SUBSET_SIZE).
    # The test.csv has 1023 rows. Debug size is 100. So we expect 100 rows.
    expected_rows = Config.DEBUG_SUBSET_SIZE
    # Columns = 1 (id) + 120 (breeds) = 121
    expected_cols = 1 + Config.NUM_CLASSES

    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(df_sub)}"
    assert (
        df_sub.shape[1] == expected_cols
    ), f"Expected {expected_cols} columns in submission, got {df_sub.shape[1]}"

    # Check probability sum (should be approx 1.0 per row)
    # Exclude 'id' column
    probs = df_sub.iloc[:, 1:].values
    row_sums = np.sum(probs, axis=1)
    # Allow small float error
    assert np.allclose(row_sums, 1.0, atol=1e-4), "Probabilities do not sum to 1.0"

    logger.info("Inference verification successful.")
    print("=== Demo Script Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
