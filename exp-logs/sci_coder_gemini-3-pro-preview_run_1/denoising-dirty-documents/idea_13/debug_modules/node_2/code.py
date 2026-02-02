import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, rmse_score
from library.model import UNet
from library.dataset import get_train_val_datasets, get_test_dataset
from library.train import train_one_seed
from library.inference import predict_and_submit, apply_tta


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    print(">>> Setting up demo configuration...")

    # Modify Config to run a fast, minimal demonstration
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.SEEDS = [42]  # Only use one seed

    # Redirect outputs to a specific demo directory to avoid clutter/conflicts
    Config.WORKING_DIR = "./working/demo_run"
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update paths in Config to point to the new working directory
    Config.TRAIN_CACHE_PATH = os.path.join(Config.WORKING_DIR, "train_cache.npz")
    Config.VAL_CACHE_PATH = os.path.join(Config.WORKING_DIR, "val_cache.npz")
    Config.TEST_CACHE_PATH = os.path.join(Config.WORKING_DIR, "test_cache.npz")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Set global seed
    seed_everything(42)
    print("Configuration updated for speed and isolation.")

    # -------------------------------------------------------------------------
    # 2. Utility & Model Verification
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Utilities and Model...")

    # Test RMSE
    score = rmse_score(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert score == 0.0, "RMSE calculation is incorrect (expected 0.0)"
    score_bad = rmse_score(np.array([0.0]), np.array([1.0]))
    assert np.isclose(score_bad, 1.0), "RMSE calculation is incorrect (expected 1.0)"

    # Test Model Architecture
    device = Config.DEVICE
    model = UNet().to(device)
    dummy_input = torch.randn(2, 1, 320, 320).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        2,
        1,
        320,
        320,
    ), f"Model output shape mismatch. Expected (2, 1, 320, 320), got {output.shape}"
    assert (
        output.min() >= 0 and output.max() <= 1
    ), "Model output should be sigmoid activated [0, 1]"

    # Test TTA (Test Time Augmentation)
    # TTA expects (1, 1, H, W) input
    tta_input = torch.randn(1, 1, 64, 64).to(device)
    tta_output = apply_tta(model, tta_input, device)
    assert tta_output.shape == tta_input.shape, "TTA output shape mismatch"

    print("Model and Utilities verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Dataset Loading Verification
    # -------------------------------------------------------------------------
    print("\n>>> Loading Datasets...")

    # Force reload from source (load_cached_data=False) to verify processing logic
    train_ds, val_ds = get_train_val_datasets(load_cached_data=False)
    test_ds = get_test_dataset(load_cached_data=False)

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")

    assert len(train_ds) > 0, "Training dataset is empty"
    assert len(val_ds) > 0, "Validation dataset is empty"
    assert len(test_ds) > 0, "Test dataset is empty"

    # Verify __getitem__ structure
    # Train: (noisy, clean, id)
    t_noisy, t_clean, t_id = train_ds[0]
    assert t_noisy.shape == (
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Train image shape incorrect (Augmentation failed?)"
    assert t_clean.shape == (
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Train mask shape incorrect"
    assert isinstance(t_id, (str, np.str_)), "ID should be a string"

    # Test: (noisy, dummy, id) - Test transform uses reflection padding, so size varies based on input
    # We just check rank
    test_noisy, _, test_id = test_ds[0]
    assert (
        test_noisy.ndim == 3 and test_noisy.shape[0] == 1
    ), "Test image tensor format incorrect"

    print("Datasets loaded and verified.")

    # -------------------------------------------------------------------------
    # 4. Training Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Running Training Demo (1 Epoch)...")

    seed = Config.SEEDS[0]
    best_rmse = train_one_seed(
        seed=seed,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        device=device,
        num_workers=0,  # Use 0 workers for simple script execution stability
    )

    model_path = Config.get_model_path(seed)
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    print(f"Training finished. Best RMSE: {best_rmse:.4f}")

    # -------------------------------------------------------------------------
    # 5. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Running Inference Demo...")

    # This function relies on Config.SEEDS and Config.get_model_path
    # It will load the model we just trained
    predict_and_submit(load_cached_data=True)

    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Verify Submission Content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")

    assert list(df_sub.columns) == ["id", "value"], "Submission columns mismatch"
    assert not df_sub.isnull().values.any(), "Submission contains NaNs"

    # Check value range (should be [0, 1] as per requirements)
    vals = df_sub["value"].values
    assert vals.min() >= 0 and vals.max() <= 1, "Submission values out of range [0, 1]"

    # Check ID format (e.g., '110_1_1')
    example_id = df_sub.iloc[0]["id"]
    assert len(example_id.split("_")) == 3, f"Invalid ID format: {example_id}"

    print("Inference successful and submission verified.")
    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
