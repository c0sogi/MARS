import os
import torch
import pandas as pd
import numpy as np
import time
from library.config import Config
from library.utils import set_seed, calculate_rmse
from library.model import RepCResUNetSR
from library.data_loader import get_dataloaders
from library.train import train_model
from library.inference import generate_submission


def run_demo():
    print("=== Starting Library Demonstration ===")

    # 1. Setup and Configuration Overrides for Demo Speed
    # We override specific Config attributes to make this run fast.
    # Note: We must also pass these values explicitly where functions use defaults evaluated at import time.
    Config.EPOCHS = 2
    Config.PATCHES_PER_IMAGE = 5  # Reduce from 100 to 5 for speed
    Config.BATCH_SIZE = 8
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission_test.csv")

    # Re-run setup to create new directories
    Config.setup()

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Data Loading Demonstration
    # ---------------------------------------------------------
    print("\n--- 1. Data Loading Demonstration ---")

    # Initialize dataloaders with reduced patches per image for speed
    loaders = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=2,  # Reduce workers for demo to avoid overhead
    )

    # Verify Train Loader
    train_loader = loaders["train"]
    # We manually set the dataset's patches_per_image because get_dataloaders initialized it with the default Config value
    # before we could pass it (get_dataloaders doesn't accept patches_per_image as arg, it reads Config).
    # Actually, looking at library/data_loader.py, get_dataloaders instantiates DenoisingDataset using Config.PATCHES_PER_IMAGE.
    # Since we modified Config.PATCHES_PER_IMAGE at the start of this script, and get_dataloaders imports Config,
    # it *should* see the change if the import happened at module level. However, to be safe and strictly follow
    # the "instantiate and utilize" instruction, we will rely on the modification we made to the class attribute Config.PATCHES_PER_IMAGE.
    # Let's verify the dataset length.

    # Force re-instantiation to be sure (since get_dataloaders was imported before we changed Config)
    # The provided get_dataloaders function reads Config.PATCHES_PER_IMAGE inside the function body?
    # Checking file content:
    # train_dataset = DenoisingDataset(..., patches_per_image=Config.PATCHES_PER_IMAGE, ...)
    # Yes, it reads the attribute at runtime.

    print(f"Train dataset length (patches): {len(train_loader.dataset)}")

    # Fetch a batch
    noisy_batch, clean_batch = next(iter(train_loader))

    print(f"Noisy Batch Shape: {noisy_batch.shape}")
    print(f"Clean Batch Shape: {clean_batch.shape}")

    # Assertions
    assert noisy_batch.shape == (
        Config.BATCH_SIZE,
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), "Incorrect train batch shape"
    assert clean_batch.shape == (
        Config.BATCH_SIZE,
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), "Incorrect clean batch shape"
    assert not torch.isnan(noisy_batch).any(), "NaNs found in noisy batch"

    # Verify Val Loader (Full Images)
    val_loader = loaders["val"]
    noisy_val, clean_val, val_id = next(iter(val_loader))
    print(f"Val Image Shape: {noisy_val.shape} (ID: {val_id[0]})")
    assert noisy_val.dim() == 4, "Val image should be 4D (B, C, H, W)"

    # ---------------------------------------------------------
    # 3. Model Instantiation & Verification
    # ---------------------------------------------------------
    print("\n--- 2. Model Verification ---")

    model = RepCResUNetSR().to(device)

    # Forward pass check
    model.eval()
    with torch.no_grad():
        dummy_input = noisy_batch.to(device)
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == dummy_input.shape, "Output shape mismatch"

    # Reparameterization check
    print("Testing switch_to_deploy()...")
    model.switch_to_deploy()
    # Verify it still runs
    with torch.no_grad():
        output_deploy = model(dummy_input)
    assert output_deploy.shape == dummy_input.shape, "Deploy mode output shape mismatch"
    print("Model successfully switched to deploy mode.")

    # ---------------------------------------------------------
    # 4. Training Loop Demonstration
    # ---------------------------------------------------------
    print("\n--- 3. Training Loop Demonstration ---")

    # We need a fresh model for training (not the one we just switched to deploy mode)
    # because deploy mode removes training branches.
    del model
    torch.cuda.empty_cache()

    # Train for a few epochs
    # We pass the modified Config values explicitly to override defaults
    best_rmse = train_model(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        patience=2,
    )

    print(f"Demo Training finished. Best RMSE: {best_rmse}")

    # Verify checkpoint creation
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), f"Checkpoint not found at {Config.BEST_MODEL_PATH}"
    print("Checkpoint verified.")

    # ---------------------------------------------------------
    # 5. Inference & Submission Demonstration
    # ---------------------------------------------------------
    print("\n--- 4. Inference & Submission Demonstration ---")

    # Generate submission for a subset of test images (limit=5)
    generate_submission(
        checkpoint_path=Config.BEST_MODEL_PATH,
        output_path=Config.SUBMISSION_PATH,
        test_csv_path=Config.TEST_CSV,
        limit=5,
        device=Config.DEVICE,
    )

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Rows: {len(df_sub)}")
    print(f"Columns: {list(df_sub.columns)}")

    # Check format
    assert list(df_sub.columns) == ["id", "value"], "Incorrect columns in submission"
    assert len(df_sub) > 0, "Submission file is empty"

    # Check ID format (e.g., "110_1_1")
    sample_id = df_sub.iloc[0]["id"]
    parts = sample_id.split("_")
    assert len(parts) == 3, f"Invalid ID format: {sample_id}"

    # Check value range
    min_val = df_sub["value"].min()
    max_val = df_sub["value"].max()
    print(f"Value Range: [{min_val}, {max_val}]")
    assert 0 <= min_val and max_val <= 1, "Values out of range [0, 1]"

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    run_demo()
