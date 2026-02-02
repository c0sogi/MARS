import os
import sys
import shutil
import torch
import pandas as pd
import warnings

# Ensure the current directory is in the path so library imports work correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import CassavaConvNext
from library.engine import run_training


def run_demo():
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    print("============================================================")
    print("       Cassava Disease Classification - Pipeline Demo       ")
    print("============================================================")

    # ------------------------------------------------------------------
    # 1. Configuration Setup
    # ------------------------------------------------------------------
    print("\n[Step 1] Initializing Configuration...")

    # Instantiate the default configuration
    cfg = Config()

    # Override settings for a fast, reproducible demonstration
    cfg.debug = True
    cfg.debug_sample_size = 64  # Use a tiny subset of data
    cfg.epochs = 2  # Run only 2 epochs to verify the loop
    cfg.batch_size = 16  # Small batch size for speed
    cfg.pretrained = False  # Disable pretraining to avoid download overhead/errors

    # Set a specific working directory for this demo
    cfg.working_dir = "./working/demo_execution"
    cfg.best_model_path = os.path.join(cfg.working_dir, "best_model.pth")
    cfg.submission_path = os.path.join(cfg.working_dir, "submission.csv")

    # Clean and create the working directory
    if os.path.exists(cfg.working_dir):
        shutil.rmtree(cfg.working_dir)
    os.makedirs(cfg.working_dir, exist_ok=True)

    print(f"    Debug Mode: {cfg.debug}")
    print(f"    Sample Size: {cfg.debug_sample_size}")
    print(f"    Device: {cfg.device}")
    print(f"    Working Directory: {cfg.working_dir}")

    # Set random seeds for full reproducibility
    seed_everything(cfg.seed)

    # ------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # ------------------------------------------------------------------
    print("\n[Step 2] Setting up and Verifying DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # Verify Training Batch (Should have Soft Targets from MixUp/CutMix)
    try:
        images, targets = next(iter(train_loader))
        print(
            f"    Train Batch - Image Shape: {images.shape}, Target Shape: {targets.shape}"
        )

        # Assertions
        assert images.shape == (
            cfg.batch_size,
            3,
            cfg.image_size,
            cfg.image_size,
        ), "Train image shape mismatch"
        assert targets.shape == (
            cfg.batch_size,
            cfg.num_classes,
        ), "Train target shape mismatch"
        assert (
            targets.dtype == torch.float32
        ), "Train targets must be float32 (soft labels)"
        print("    -> Train Loader verification passed (Soft targets detected).")
    except StopIteration:
        raise Exception("Train loader is empty!")

    # Verify Validation Batch (Should have Standard Indices)
    try:
        val_images, val_labels = next(iter(val_loader))
        print(
            f"    Val Batch   - Image Shape: {val_images.shape}, Label Shape: {val_labels.shape}"
        )

        # Assertions
        assert val_images.shape[0] <= cfg.batch_size, "Val batch size mismatch"
        assert val_labels.dim() == 1, "Val labels must be 1D indices"
        assert val_labels.dtype == torch.long, "Val labels must be long/int"
        print(
            "    -> Validation Loader verification passed (Standard indices detected)."
        )
    except StopIteration:
        raise Exception("Validation loader is empty!")

    # ------------------------------------------------------------------
    # 3. Model Initialization Verification
    # ------------------------------------------------------------------
    print("\n[Step 3] Initializing Model and Verifying Forward Pass...")
    model = CassavaConvNext(cfg)
    model.to(cfg.device)

    # Create dummy input
    dummy_input = torch.randn(2, 3, cfg.image_size, cfg.image_size).to(cfg.device)

    # Check forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Model Output Shape: {output.shape}")
    assert output.shape == (2, cfg.num_classes), "Model output shape mismatch"
    print("    -> Model initialization and forward pass verified.")

    # ------------------------------------------------------------------
    # 4. Training Engine Execution
    # ------------------------------------------------------------------
    print("\n[Step 4] Executing Training Loop (Train -> Val -> Inference)...")
    # run_training handles the full loop: training, validation, checkpointing, and TTA inference
    submission_df = run_training(model, train_loader, val_loader, test_loader, cfg)

    # ------------------------------------------------------------------
    # 5. Output Verification
    # ------------------------------------------------------------------
    print("\n[Step 5] Verifying Output Files...")

    # Check Submission File
    assert os.path.exists(cfg.submission_path), "Submission file was not created"

    # Validate Submission Content
    loaded_df = pd.read_csv(cfg.submission_path)
    print(f"    Submission Shape: {loaded_df.shape}")

    expected_cols = ["image_id", "label"]
    assert (
        list(loaded_df.columns) == expected_cols
    ), f"Columns mismatch. Expected {expected_cols}"
    assert (
        len(loaded_df) == cfg.debug_sample_size
    ), f"Length mismatch. Expected {cfg.debug_sample_size}"
    assert not loaded_df.isnull().values.any(), "Submission contains NaN values"

    print("    -> Submission file verified successfully.")

    # Check Model Checkpoint
    # Note: Model is saved only if validation accuracy improves or loss decreases.
    # With random init and 2 epochs, it's possible but not guaranteed to improve over initial random state.
    # However, we check if the file exists if the engine reported saving it.
    if os.path.exists(cfg.best_model_path):
        print("    -> Best model checkpoint found.")
    else:
        print(
            "    -> No model checkpoint found (Metric might not have improved in short debug run)."
        )

    print("\n============================================================")
    print("       Demo Completed Successfully                          ")
    print("============================================================")


if __name__ == "__main__":
    run_demo()
