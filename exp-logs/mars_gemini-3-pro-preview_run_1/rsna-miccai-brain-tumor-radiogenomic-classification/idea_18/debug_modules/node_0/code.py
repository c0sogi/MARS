import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import prepare_data
from library.model import WIISNet
from library.train import run_training
from library.predict import predict_and_submit


def main():
    print("Starting WIISNet Library Demonstration...")

    # ==========================================
    # 1. Configuration Override for Demo
    # ==========================================
    # We modify the Config class attributes at runtime to create a fast, isolated demo environment.

    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    print(f"Configuring demo environment in {DEMO_DIR}...")

    # Update Paths
    Config.WORKING_DIR = DEMO_DIR
    Config.BEST_MODEL_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Update Cache Paths to avoid conflicts with existing runs
    Config.CACHE_TRAIN_IMAGES = os.path.join(DEMO_DIR, "cache_train_images.npy")
    Config.CACHE_TRAIN_LABELS = os.path.join(DEMO_DIR, "cache_train_targets.npy")
    Config.CACHE_TRAIN_IDS = os.path.join(DEMO_DIR, "cache_train_ids.npy")
    Config.CACHE_VAL_IMAGES = os.path.join(DEMO_DIR, "cache_val_images.npy")
    Config.CACHE_VAL_LABELS = os.path.join(DEMO_DIR, "cache_val_targets.npy")
    Config.CACHE_VAL_IDS = os.path.join(DEMO_DIR, "cache_val_ids.npy")
    Config.CACHE_TEST_IMAGES = os.path.join(DEMO_DIR, "cache_test_images.npy")
    Config.CACHE_TEST_IDS = os.path.join(DEMO_DIR, "cache_test_ids.npy")

    # Optimize for Speed
    Config.DEBUG = True  # Limits data to 20 subjects
    Config.DEBUG_SAMPLE_SIZE = 10  # Even smaller for this demo
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.IMG_SIZE = 128  # Smaller images for faster processing
    Config.LOAD_CACHED_DATA = False  # Force processing from scratch to demo logic

    # Set Seed
    seed_everything(Config.SEED)

    # ==========================================
    # 2. Data Preparation Demo & Validation
    # ==========================================
    print("\n[Step 2] Demonstrating Data Preparation...")

    # This will read metadata, load DICOMs (or mock if missing), and create slabs
    train_dataset, val_dataset, test_dataset = prepare_data(load_cached_data=False)

    # Assertions to verify data integrity
    print("Validating dataset properties...")

    # Check if datasets are populated (DEBUG mode ensures we have some data)
    assert len(train_dataset) > 0, "Training dataset is empty."
    assert len(val_dataset) > 0, "Validation dataset is empty."
    assert len(test_dataset) > 0, "Test dataset is empty."

    # Check sample structure
    sample_img, sample_target = train_dataset[0]

    # Expected shape: (9, 128, 128) -> 9 channels (3 modalities * 3 slices), H, W
    expected_shape = (9, Config.IMG_SIZE, Config.IMG_SIZE)
    assert (
        sample_img.shape == expected_shape
    ), f"Incorrect image shape. Expected {expected_shape}, got {sample_img.shape}"

    # Expected target: Scalar tensor
    assert isinstance(sample_target, torch.Tensor), "Target is not a tensor."
    assert sample_target.numel() == 1, "Target is not a scalar."

    print(f"Data Loaded Successfully. Train samples: {len(train_dataset)}")

    # ==========================================
    # 3. Model Architecture Demo & Validation
    # ==========================================
    print("\n[Step 3] Demonstrating Model Architecture...")

    device = get_device()
    model = WIISNet().to(device)

    # Create a dummy batch to verify forward pass
    dummy_batch = torch.randn(2, 9, Config.IMG_SIZE, Config.IMG_SIZE).to(device)

    with torch.no_grad():
        output = model(dummy_batch)

    # Expected output shape: (Batch_Size, 1) -> Binary classification logits
    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"

    print("Model instantiated and forward pass verified.")

    # ==========================================
    # 4. Training Loop Demo
    # ==========================================
    print("\n[Step 4] Demonstrating Training Loop...")

    # run_training uses the Config settings we modified (1 epoch, debug data)
    run_training()

    # Verify that the model checkpoint was saved
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), f"Model checkpoint not found at {Config.BEST_MODEL_PATH} after training."

    print("Training complete. Checkpoint saved.")

    # ==========================================
    # 5. Inference Pipeline Demo
    # ==========================================
    print("\n[Step 5] Demonstrating Inference and Submission...")

    # predict_and_submit loads the best model and generates submission.csv
    predict_and_submit()

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Verify submission format
    assert "BraTS21ID" in df_sub.columns, "Submission missing 'BraTS21ID' column."
    assert "MGMT_value" in df_sub.columns, "Submission missing 'MGMT_value' column."
    assert len(df_sub) > 0, "Submission file is empty."

    # Verify probability range
    probs = df_sub["MGMT_value"].values
    assert np.all(
        (probs >= 0) & (probs <= 1)
    ), "Predictions contain values outside [0, 1]."

    print("Inference complete. Submission file verified.")
    print("\nAll demonstration steps completed successfully!")


if __name__ == "__main__":
    main()
