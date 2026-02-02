import os
import torch
import pandas as pd
import numpy as np
from library.config import Config, set_seed
from library.data_loader import get_dataloaders
from library.model import RHTN
from library.train import train_model
from library.predict import generate_submission
from library.utils import load_checkpoint


def run_demo():
    print("============================================================")
    print("DEMO: Ship vs. Iceberg Classification Pipeline")
    print("============================================================")

    # 1. Setup and Configuration overrides for Speed
    print("\n[Step 1] Configuring environment for rapid demonstration...")
    set_seed(42)

    # Enable Debug mode to use a small subset of data (e.g., 20 samples)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20

    # Reduce training parameters for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.PATIENCE = 1

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Epochs: {Config.NUM_EPOCHS}")

    # 2. Data Loading Verification
    print("\n[Step 2] Verifying Data Loading...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=False
    )

    # Fetch one batch from training loader
    images, meta, labels = next(iter(train_loader))

    print(f"Train Batch - Images Shape: {images.shape}")
    print(f"Train Batch - Meta Shape: {meta.shape}")
    print(f"Train Batch - Labels Shape: {labels.shape}")

    # Assertions for Data Integrity
    expected_img_shape = (Config.BATCH_SIZE, 3, 75, 75)
    assert (
        images.shape == expected_img_shape
    ), f"Expected image shape {expected_img_shape}, got {images.shape}"

    assert meta.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected meta shape {(Config.BATCH_SIZE, 1)}, got {meta.shape}"

    assert labels.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected label shape {(Config.BATCH_SIZE, 1)}, got {labels.shape}"

    print("Data Loader verification passed.")

    # 3. Model Architecture Verification
    print("\n[Step 3] Verifying Model Architecture...")
    device = torch.device(Config.DEVICE)
    model = RHTN().to(device)

    # Move sample batch to device
    images_dev = images.to(device)
    meta_dev = meta.to(device)

    # Perform Forward Pass
    logits = model(images_dev, meta_dev)

    print(f"Model Output Logits Shape: {logits.shape}")

    # Assertions for Model Output
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected output shape {(Config.BATCH_SIZE, 1)}, got {logits.shape}"

    assert not torch.isnan(logits).any(), "Model output contains NaNs"
    assert not torch.isinf(logits).any(), "Model output contains Infs"

    print("Model forward pass verification passed.")

    # 4. Training Pipeline Execution
    print("\n[Step 4] Executing Training Pipeline...")
    # This function internally uses the configured DataLoaders and Model
    trained_model = train_model()

    # Verify Model Checkpoint creation
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"

    print(f"Training complete. Model saved to {Config.MODEL_SAVE_PATH}")

    # 5. Inference and Submission Generation
    print("\n[Step 5] Generating Submission...")
    generate_submission()

    # Verify Submission File creation
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Validate Submission Content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Shape: {df_sub.shape}")
    print(f"Columns: {df_sub.columns.tolist()}")

    # Assertions for Submission Format
    assert (
        "id" in df_sub.columns and "is_iceberg" in df_sub.columns
    ), "Submission file missing required columns"

    assert len(df_sub) == len(
        test_ids
    ), f"Submission row count {len(df_sub)} does not match test set size {len(test_ids)}"

    # Check probability range
    probs = df_sub["is_iceberg"].values
    assert np.all(
        (probs >= 0) & (probs <= 1)
    ), "Predictions contain probabilities outside [0, 1] range"

    print("Submission format verification passed.")

    # 6. Utility Verification (Explicit Checkpoint Loading)
    print("\n[Step 6] Verifying Utility Functions (Checkpoint Loading)...")
    # Create a fresh model instance
    new_model = RHTN()
    # Load weights
    loaded_model = load_checkpoint(new_model, Config.MODEL_SAVE_PATH, device="cpu")

    # Check if state dicts match (simple check on one layer)
    original_state = trained_model.state_dict()
    loaded_state = loaded_model.state_dict()

    layer_name = "meta_mlp.0.weight"
    if layer_name in original_state:
        diff = torch.sum(original_state[layer_name].cpu() - loaded_state[layer_name])
        assert diff == 0, "Loaded model weights do not match saved weights"

    print("Checkpoint loading verification passed.")

    print("\n============================================================")
    print("SUCCESS: All demo steps completed without errors.")
    print("============================================================")


if __name__ == "__main__":
    run_demo()
