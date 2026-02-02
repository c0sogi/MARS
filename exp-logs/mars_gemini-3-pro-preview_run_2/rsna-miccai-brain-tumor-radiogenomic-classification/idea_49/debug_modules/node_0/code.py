import os
import sys
import torch
import pandas as pd
import numpy as np
import logging

# Import from the provided library
from library.config import Config, set_seed
from library.data_loader import (
    get_dataloaders,
    MRIDataset,
    get_anchor_slices,
    get_transforms,
)
from library.model import AsymmetricEfficientNet
from library.train import run_training
from library.inference import generate_submission
from library.utils import get_logger


def main():
    # --------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides
    # --------------------------------------------------------------------------
    print(">>> [1/5] Setting up configuration...")

    # Override working directory to isolate this demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = Config.WORKING_DIR
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Create directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Optimize for speed in this demo
    Config.EPOCHS = 1
    Config.MAX_DEBUG_SAMPLES = 10  # Very small subset for speed
    Config.BATCH_SIZE = 4  # Small batch for verification
    Config.NUM_WORKERS = 2

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Initialize logger
    logger = get_logger(name="DemoScript", log_file="demo.log")
    logger.info("Configuration configured for rapid demonstration.")

    # --------------------------------------------------------------------------
    # 2. Verify Data Loading Logic
    # --------------------------------------------------------------------------
    logger.info(">>> [2/5] Verifying Data Loading Logic...")

    # Load metadata manually to test dataset class
    if not os.path.exists(Config.TRAIN_METADATA):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA}")

    df_full = pd.read_csv(Config.TRAIN_METADATA)
    # Take top 4 samples
    df_subset = df_full.head(4).copy()

    # Compute anchors for this subset
    # This tests the ROI selection logic (intensity-based slice selection)
    logger.info("Computing anchors for subset...")
    anchor_dict = get_anchor_slices(df_subset, load_cached_data=False)

    # Instantiate Dataset
    dataset = MRIDataset(
        df=df_subset,
        anchor_dict=anchor_dict,
        transform=get_transforms(mode="train"),
        mode="train",
    )

    # Check length
    assert len(dataset) == 4, f"Dataset length mismatch. Expected 4, got {len(dataset)}"

    # Check item retrieval
    img_tensor, label = dataset[0]

    # Verify Shapes
    # Expected: (C, H, W) where C = 24 (4 mods * 2 scales * 3 slices)
    expected_channels = 24
    expected_size = Config.IMG_SIZE

    logger.info(f"Sample Tensor Shape: {img_tensor.shape}")

    assert img_tensor.shape == (
        expected_channels,
        expected_size,
        expected_size,
    ), f"Image shape mismatch. Expected {(expected_channels, expected_size, expected_size)}, got {img_tensor.shape}"

    assert isinstance(label, torch.Tensor), "Label should be a torch tensor"

    logger.info("Data Loading verification passed.")

    # --------------------------------------------------------------------------
    # 3. Verify Model Architecture
    # --------------------------------------------------------------------------
    logger.info(">>> [3/5] Verifying Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AsymmetricEfficientNet().to(device)

    # Create a dummy batch
    dummy_input = img_tensor.unsqueeze(0).to(device)  # (1, 24, 224, 224)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    logger.info(f"Model Output Shape: {output.shape}")

    # Expected output: (Batch_Size, 1) - raw logits
    assert output.shape == (
        1,
        1,
    ), f"Model output shape mismatch. Expected (1, 1), got {output.shape}"

    logger.info("Model architecture verification passed.")

    # --------------------------------------------------------------------------
    # 4. Run Training Simulation
    # --------------------------------------------------------------------------
    logger.info(">>> [4/5] Running Training Simulation (Debug Mode)...")

    # We use the provided run_training function.
    # debug=True will truncate the dataset to Config.MAX_DEBUG_SAMPLES (10).
    # epochs=1 ensures it finishes quickly.

    try:
        run_training(epochs=Config.EPOCHS, debug=True)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise e

    # Verify that the model was saved
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Training completed but model file not found at {Config.MODEL_PATH}"
        )

    logger.info("Training simulation completed and model saved.")

    # --------------------------------------------------------------------------
    # 5. Run Inference and Generate Submission
    # --------------------------------------------------------------------------
    logger.info(">>> [5/5] Running Inference and Generating Submission...")

    # We use the provided generate_submission function.
    # It will load the model we just trained and predict on the test set.
    # Note: The test set has 59 samples.

    try:
        submission_df = generate_submission(load_cached_data=False)
    except Exception as e:
        logger.error(f"Inference failed with error: {e}")
        raise e

    # Verify Submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    # Check content
    assert "BraTS21ID" in submission_df.columns, "Submission missing BraTS21ID column"
    assert "MGMT_value" in submission_df.columns, "Submission missing MGMT_value column"
    assert len(submission_df) > 0, "Submission dataframe is empty"

    # Check value range (probabilities)
    preds = submission_df["MGMT_value"].values
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions out of probability range [0, 1]"

    logger.info(f"Submission generated successfully with {len(submission_df)} rows.")
    logger.info("Demo execution completed successfully.")


if __name__ == "__main__":
    main()
