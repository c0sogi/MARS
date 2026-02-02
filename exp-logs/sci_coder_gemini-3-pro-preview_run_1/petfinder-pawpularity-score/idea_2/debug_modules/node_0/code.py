import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_rmse
from library.dataset import get_dataloaders, PawpularityDataset
from library.model import PawpularitySwinModel
from library.engine import train_one_epoch, validate, inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration of Pawpularity Library ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Enable debug mode to use a tiny subset of data
    Config.debug = True
    Config.debug_sample_size = 32  # Small sample size

    # Reduce training parameters
    Config.epochs = 1
    Config.batch_size = 8
    Config.num_workers = 0  # Avoid multiprocessing overhead for small script
    Config.model_name = "swin_tiny_patch4_window7_224"  # Keep default

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    print(f"Debug Mode: {Config.debug}")
    print(f"Batch Size: {Config.batch_size}")
    print(f"Device: {Config.device}")

    # -------------------------------------------------------------------------
    # 2. Utils Demonstration
    # -------------------------------------------------------------------------
    print("\n[2] Testing Utility Functions...")

    # Test Seeding
    seed_everything(Config.seed)
    print("Seed set successfully.")

    # Test RMSE Calculation
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    rmse = calculate_rmse(y_true, y_pred)

    # Manual calc: sqrt((4 + 4 + 9)/3) = sqrt(17/3) = sqrt(5.66) ~= 2.38
    expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    assert np.isclose(
        rmse, expected_rmse
    ), f"RMSE calculation mismatch: {rmse} vs {expected_rmse}"

    # Test with Tensors
    t_true = torch.tensor([10.0, 20.0, 30.0])
    t_pred = torch.tensor([12.0, 18.0, 33.0])
    rmse_tensor = calculate_rmse(t_true, t_pred)
    assert np.isclose(rmse, rmse_tensor), "Tensor RMSE calculation mismatch"

    print(f"RMSE check passed. Value: {rmse:.4f}")

    # -------------------------------------------------------------------------
    # 3. Dataset and DataLoader Demonstration
    # -------------------------------------------------------------------------
    print("\n[3] Testing Data Loading...")

    # Initialize DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders()

    print(f"Train Loader batches: {len(train_loader)}")
    print(f"Val Loader batches:   {len(val_loader)}")
    print(f"Test Loader batches:  {len(test_loader)}")

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    images = batch["image"]
    features = batch["features"]
    targets = batch["target"]
    ids = batch["id"]

    print(f"Batch Keys: {list(batch.keys())}")
    print(f"Image Shape: {images.shape}")  # Expected: [Batch_Size, 3, 224, 224]
    print(f"Features Shape: {features.shape}")  # Expected: [Batch_Size, 12]
    print(f"Target Shape: {targets.shape}")  # Expected: [Batch_Size]

    # Assertions
    assert images.shape == (
        Config.batch_size,
        3,
        Config.img_size,
        Config.img_size,
    ), "Incorrect image shape"
    assert features.shape == (Config.batch_size, 12), "Incorrect features shape"
    assert targets.shape == (Config.batch_size,), "Incorrect target shape"
    assert len(ids) == Config.batch_size, "Incorrect ID list length"

    # -------------------------------------------------------------------------
    # 4. Model Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Testing Model Architecture...")

    # Initialize model
    # We set pretrained=False to avoid downloading weights during this timed demo
    model = PawpularitySwinModel(pretrained=False)
    model.to(Config.device)

    # Perform a forward pass with the batch from step 3
    images = images.to(Config.device)
    features = features.to(Config.device)

    output = model(images, features)

    print(f"Model Output Shape: {output.shape}")  # Expected: [Batch_Size, 1]

    # Assertions
    assert output.shape == (Config.batch_size, 1), "Model output shape mismatch"
    assert not torch.isnan(output).any(), "Model produced NaN outputs"

    # -------------------------------------------------------------------------
    # 5. Engine Demonstration (Train, Val, Inference)
    # -------------------------------------------------------------------------
    print("\n[5] Testing Training and Inference Engine...")

    # Setup Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    # A. Train One Epoch
    print("Running training step...")
    train_loss = train_one_epoch(model, optimizer, train_loader, Config.device, epoch=1)
    print(f"Training Loss: {train_loss:.4f}")
    assert train_loss > 0, "Training loss should be positive"

    # B. Validate
    print("Running validation step...")
    val_loss, val_rmse = validate(model, val_loader, Config.device)
    print(f"Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}")
    assert val_rmse >= 0, "RMSE cannot be negative"

    # C. Inference
    print("Running inference step...")
    # Ensure submission dir exists
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)

    inference(model, test_loader, Config.device, output_path=Config.submission_path)

    # Verify Submission File
    assert os.path.exists(Config.submission_path), "Submission file was not created"

    submission_df = pd.read_csv(Config.submission_path)
    print(f"Submission file loaded. Rows: {len(submission_df)}")
    print(submission_df.head(3))

    # Check submission length matches test loader dataset length
    # Note: In debug mode, test dataset is also subsampled
    expected_len = len(test_loader.dataset)
    assert (
        len(submission_df) == expected_len
    ), f"Submission length ({len(submission_df)}) does not match test set length ({expected_len})"

    # Check columns
    assert (
        "Id" in submission_df.columns and "Pawpularity" in submission_df.columns
    ), "Submission file missing required columns"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
