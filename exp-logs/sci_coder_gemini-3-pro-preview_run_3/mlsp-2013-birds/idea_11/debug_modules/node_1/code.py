import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Ensure the current directory is in the python path for library imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.data import make_folds, get_loaders, get_test_loader
from library.models import get_model
from library.engine import train_fold, inference, save_submission


def run_demo():
    print("==== Starting Demo Execution ====")

    # 1. Configure for Speed/Demo
    # Override Config parameters to ensure quick execution
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 32  # Use only 32 samples
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.WORK_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORK_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORK_DIR, "submission")

    # Re-run setup to create new directories
    Config.setup()

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    print(f"Working Directory: {Config.WORK_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")

    # 2. Data Pipeline Verification
    print("\n[1/5] Verifying Data Pipeline...")

    # Create/Load Folds
    df_folds = make_folds(load_cached_data=False)
    assert isinstance(df_folds, pd.DataFrame), "make_folds should return a DataFrame"
    assert "fold" in df_folds.columns, "DataFrame must contain 'fold' column"
    print(f"Folds DataFrame shape: {df_folds.shape}")

    # Get Loaders for Fold 0
    train_loader, valid_loader = get_loaders(
        fold_idx=0, df_folds=df_folds, batch_size=8
    )

    # Verify Train Loader Batch
    images, labels, rec_ids = next(iter(train_loader))
    print(f"Batch shapes - Images: {images.shape}, Labels: {labels.shape}")

    # Assertions for shapes
    # Images: (Batch, 3, 224, 224)
    assert images.dim() == 4, "Images must be 4D tensor"
    assert images.shape[1] == 3, "Images must have 3 channels"
    assert images.shape[2] == 224 and images.shape[3] == 224, "Images must be 224x224"
    # Labels: (Batch, 19)
    assert labels.dim() == 2, "Labels must be 2D tensor"
    assert (
        labels.shape[1] == Config.NUM_CLASSES
    ), f"Labels must have {Config.NUM_CLASSES} classes"

    # Get Test Loader
    test_loader = get_test_loader(batch_size=8)
    test_images, _, _ = next(iter(test_loader))
    assert test_images.shape[1:] == (3, 224, 224), "Test images shape mismatch"
    print("Data Pipeline Verified.")

    # 3. Model Verification
    print("\n[2/5] Verifying Model Architecture...")
    model_name = "resnet18"
    model = get_model(
        model_name, pretrained=False
    )  # No need to download weights for logic check

    # Move to CPU for quick check (or GPU if available, Config handles device)
    device = Config.DEVICE
    model = model.to(device)

    # Forward pass check
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (4, Config.NUM_CLASSES), "Model output shape mismatch"
    print("Model Architecture Verified.")

    # 4. Training Engine Verification
    print("\n[3/5] Verifying Training Loop...")

    # Train for 1 epoch on fold 0
    # Note: Config.DEBUG is True, so this uses the subset
    best_auc, checkpoint_path = train_fold(
        fold_idx=0,
        model_name=model_name,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
    )

    print(f"Training completed. Best AUC: {best_auc}")
    print(f"Checkpoint path: {checkpoint_path}")

    assert os.path.exists(checkpoint_path), "Checkpoint file was not created"
    print("Training Loop Verified.")

    # 5. Inference Verification
    print("\n[4/5] Verifying Inference...")

    rec_ids, preds = inference(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        test_loader=test_loader,
        device=device,
    )

    print(f"Predictions Shape: {preds.shape}")
    print(f"Recording IDs Shape: {rec_ids.shape}")

    # Assertions
    # In Debug mode, test loader also loads a subset (32 samples)
    expected_samples = Config.DEBUG_SUBSET_SIZE
    assert preds.shape == (
        expected_samples,
        Config.NUM_CLASSES,
    ), "Predictions shape mismatch"
    assert len(rec_ids) == expected_samples, "Recording IDs count mismatch"
    print("Inference Verified.")

    # 6. Submission Verification
    print("\n[5/5] Verifying Submission Generation...")

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    save_submission(rec_ids, preds, submission_path)

    assert os.path.exists(submission_path), "Submission file not found"

    # Check content format
    df_sub = pd.read_csv(submission_path)
    print(f"Submission Head:\n{df_sub.head()}")

    expected_cols = ["Id", "Probability"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Columns mismatch. Expected {expected_cols}"

    # Check row count: num_samples * num_classes
    expected_rows = expected_samples * Config.NUM_CLASSES
    assert (
        len(df_sub) == expected_rows
    ), f"Row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    print("Submission Generation Verified.")

    print("\n==== Demo Execution Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
