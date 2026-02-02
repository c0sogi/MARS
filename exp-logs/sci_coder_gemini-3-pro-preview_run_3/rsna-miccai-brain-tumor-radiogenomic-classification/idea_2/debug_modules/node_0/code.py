import os
import shutil
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_train_val_datasets, get_test_dataset
from library.model import MILNet
from library.engine import train_model, generate_submission


def run_demo():
    print("Initializing Demo...")

    # 1. Setup Configuration for Speed and Reproducibility
    seed_everything(42)

    # Override Config for rapid demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 4  # Very small subset for speed
    Config.EPOCHS = 2  # Minimal epochs to verify loop
    Config.BATCH_SIZE = 2  # Small batch size
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./submission"
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Update cache paths to new working dir to avoid conflicts
    Config.CACHE_TRAIN_X = os.path.join(Config.WORKING_DIR, "cached_train_X.npy")
    Config.CACHE_TRAIN_Y = os.path.join(Config.WORKING_DIR, "cached_train_y.npy")
    Config.CACHE_VAL_X = os.path.join(Config.WORKING_DIR, "cached_val_X.npy")
    Config.CACHE_VAL_Y = os.path.join(Config.WORKING_DIR, "cached_val_y.npy")
    Config.CACHE_TEST_X = os.path.join(Config.WORKING_DIR, "cached_test_X.npy")
    Config.CACHE_TEST_IDS = os.path.join(Config.WORKING_DIR, "cached_test_ids.npy")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Configured for DEBUG mode with sample size: {Config.DEBUG_SAMPLE_SIZE}")

    # 2. Data Loading & Verification
    print("\n--- Step 1: Data Loading ---")
    # Force generation from scratch to verify processing logic
    train_dataset, val_dataset = get_train_val_datasets(load_cached_data=False)

    # Assertions for Dataset
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    assert (
        len(train_dataset) <= Config.DEBUG_SAMPLE_SIZE
    ), "Train dataset size exceeds debug limit"
    assert (
        len(val_dataset) <= Config.DEBUG_SAMPLE_SIZE
    ), "Val dataset size exceeds debug limit"

    # Check Tensor Shapes
    # Expected: (Batch, Num_Slices, Channels, H, W) -> (1, 32, 4, 256, 256)
    sample_x, sample_y = train_dataset[0]
    print(f"Sample Input Shape: {sample_x.shape}")
    print(f"Sample Target: {sample_y}")

    expected_shape = (
        Config.NUM_SLICES,
        Config.IN_CHANNELS,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    )
    assert (
        sample_x.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {sample_x.shape}"
    assert isinstance(sample_y, torch.Tensor), "Target should be a tensor"

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 3. Model Initialization & Verification
    print("\n--- Step 2: Model Verification ---")
    device = Config.DEVICE
    print(f"Using device: {device}")

    model = MILNet(in_channels=Config.IN_CHANNELS).to(device)

    # Forward pass check
    dummy_input = sample_x.unsqueeze(0).to(device)  # Add batch dimension
    print(f"Forward pass input shape: {dummy_input.shape}")

    output = model(dummy_input)
    print(f"Forward pass output shape: {output.shape}")

    assert output.shape == (1, 1), f"Expected output shape (1, 1), got {output.shape}"

    # Check gradient flow
    loss = torch.nn.BCEWithLogitsLoss()(output, torch.tensor([[1.0]]).to(device))
    loss.backward()
    print("Gradient backward pass successful.")

    # 4. Training Loop Execution
    print("\n--- Step 3: Training Loop ---")
    # This runs the engine.train_model function
    train_model(model, train_loader, val_loader, device)

    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not saved."
    print("Training loop completed successfully.")

    # 5. Inference & Submission
    print("\n--- Step 4: Inference & Submission ---")
    # Get test dataset (debug mode)
    test_dataset = get_test_dataset(load_cached_data=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    print(f"Test dataset size: {len(test_dataset)}")

    # Generate submission
    generate_submission(model, test_loader, device)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission file content head:")
    print(df_sub.head())

    assert "BraTS21ID" in df_sub.columns, "Missing BraTS21ID column"
    assert "MGMT_value" in df_sub.columns, "Missing MGMT_value column"
    assert len(df_sub) == len(test_dataset), "Submission row count mismatch"

    # Check probability range
    probs = df_sub["MGMT_value"].values
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of range [0, 1]"

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
