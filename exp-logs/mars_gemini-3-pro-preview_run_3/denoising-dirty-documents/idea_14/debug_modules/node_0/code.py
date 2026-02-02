import sys
import os
import shutil
import torch
import numpy as np
import pandas as pd

# Ensure the current directory is in the path to import from 'library'
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.data_loader import load_dataset_patches, DenoisingDataset
from library.network import DnCNN
from library.trainer import train_ensemble_member
from library.predictor import inference_pipeline


def run_demonstration():
    print("=== Denoising Pipeline Demonstration ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Override
    # -------------------------------------------------------------------------
    print("[Step 1] Configuring environment for rapid demonstration...")

    # We override Config attributes to run a fast, minimal version of the pipeline.
    # This ensures the code completes within the time limit while exercising all logic.
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200  # Limit patch extraction to 200 patches

    # Reduce training complexity
    Config.STAGE_1_EPOCHS = 1
    Config.STAGE_2_EPOCHS = 1
    Config.ENSEMBLE_SIZE = 1  # Train only one model instead of 5
    Config.BATCH_SIZE = 16

    # Reduce model complexity for speed
    Config.NUM_FEATURES = 16
    Config.NUM_RES_BLOCKS = 2

    # Set a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_FILE = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Clean up previous demo runs if they exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set global seeds
    seed_everything(Config.SEED)
    print(f"  Working Directory: {Config.WORKING_DIR}")
    print(f"  Debug Mode: {Config.DEBUG}")
    print("  Configuration updated successfully.")

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Data Loading Logic...")

    # Load patches (this triggers extraction from images since cache is empty)
    # We use 'train_sparse' as defined in the curriculum
    patches, targets = load_dataset_patches("train_sparse", load_cached_data=False)

    print(f"  Extracted Patches Shape: {patches.shape}")
    print(f"  Extracted Targets Shape: {targets.shape}")

    # Assertions to verify data integrity
    assert patches.ndim == 4, "Patches must be 4D tensors (N, C, H, W)"
    assert targets.ndim == 4, "Targets must be 4D tensors (N, C, H, W)"
    assert patches.shape == targets.shape, "Input and Target shapes must match"
    assert (
        patches.shape[0] <= Config.DEBUG_SAMPLE_SIZE + 50
    ), "Debug sample size limit exceeded"
    assert (
        patches.max() <= 1.0 and patches.min() >= 0.0
    ), "Pixel values must be normalized to [0, 1]"

    # Verify Dataset class and Augmentations
    dataset = DenoisingDataset(patches, targets, augment=True)
    sample_noisy, sample_clean = dataset[0]

    assert torch.is_tensor(sample_noisy), "Dataset must return torch tensors"
    assert sample_noisy.shape == (
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), "Incorrect patch dimensions"
    print("  Data loading and Dataset class verified.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = DnCNN(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        num_features=Config.NUM_FEATURES,
        num_blocks=Config.NUM_RES_BLOCKS,
    ).to(device)

    # Create a dummy batch to check forward pass
    dummy_input = torch.randn(4, 1, Config.PATCH_SIZE, Config.PATCH_SIZE).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Input Shape: {dummy_input.shape}")
    print(f"  Output Shape: {output.shape}")

    # The model predicts noise, so output shape must match input shape
    assert output.shape == dummy_input.shape, "Model output shape mismatch"
    print("  Model architecture verified.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Training Loop (Member 0)...")

    # Train the first member of the ensemble
    # This function handles the curriculum (Stage 1 & 2) internally
    train_ensemble_member(member_id=0, seed=Config.SEED)

    expected_model_path = os.path.join(Config.WORKING_DIR, "model_0.pth")
    if not os.path.exists(expected_model_path):
        raise FileNotFoundError(
            f"Training failed to produce model file at {expected_model_path}"
        )

    print(f"  Model successfully saved to: {expected_model_path}")

    # -------------------------------------------------------------------------
    # 5. Inference Pipeline Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 5] Running Inference Pipeline...")

    # The inference pipeline loads test data, applies TTA, ensembles models (just 1 here),
    # and generates the submission file.
    inference_pipeline()

    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError(
            f"Inference failed to produce submission file at {Config.SUBMISSION_FILE}"
        )

    # Validate Submission Format
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"  Submission File Generated. Rows: {len(df_sub)}")
    print(f"  Columns: {list(df_sub.columns)}")

    # Check basic requirements
    assert (
        "id" in df_sub.columns and "value" in df_sub.columns
    ), "Submission missing required columns"
    assert len(df_sub) > 0, "Submission file is empty"

    # Check ID format (e.g., '110_1_1')
    sample_id = df_sub.iloc[0]["id"]
    assert len(sample_id.split("_")) >= 3, f"Invalid ID format: {sample_id}"

    # Check value range
    assert (
        df_sub["value"].min() >= 0 and df_sub["value"].max() <= 1
    ), "Pixel values out of range [0, 1]"

    print("  Inference and submission format verified.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
