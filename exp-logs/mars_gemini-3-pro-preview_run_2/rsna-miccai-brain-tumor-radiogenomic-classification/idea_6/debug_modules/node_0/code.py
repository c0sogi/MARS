import os
import pandas as pd
import torch
import numpy as np
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.model as model_lib
import library.engine as engine


def run_demo():
    print("--- Starting Library Usage Demonstration ---")

    # 1. Setup and Configuration Overrides
    # We override these to ensure the demo runs quickly.
    utils.seed_everything(config.SEED)
    config.NUM_EPOCHS = 1
    config.BATCH_SIZE = 4

    # Define paths for demo artifacts
    # We use the working directory defined in config
    print(f"Working Directory: {config.WORKING_DIR}")

    # 2. Data Loading Demonstration
    print("\n[Step 1] Loading Metadata and Creating DataLoaders...")

    # Load metadata
    train_df_full = pd.read_csv(config.TRAIN_METADATA)
    val_df_full = pd.read_csv(config.VAL_METADATA)
    test_df_full = pd.read_csv(config.TEST_METADATA)

    # Create tiny subsets for demonstration speed
    train_subset = train_df_full.head(8).copy()
    val_subset = val_df_full.head(4).copy()
    test_subset = test_df_full.head(4).copy()

    print(
        f"Subset sizes - Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}"
    )

    # Instantiate DataLoaders
    # This triggers the anchor_ratio calculation and caching mechanism
    train_loader = data_loader.get_dataloader(
        train_subset, batch_size=config.BATCH_SIZE, phase="train"
    )
    val_loader = data_loader.get_dataloader(
        val_subset, batch_size=config.BATCH_SIZE, phase="val"
    )

    # Verify DataLoader output
    images, labels = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    # Assertions for Data Integrity
    # Expected: (Batch, Channels=12, Height=224, Width=224)
    expected_channels = config.NUM_MODALITIES * config.SLICES_PER_MODALITY
    assert images.shape == (
        config.BATCH_SIZE,
        expected_channels,
        config.IMG_SIZE,
        config.IMG_SIZE,
    ), f"Incorrect image shape. Expected {(config.BATCH_SIZE, expected_channels, config.IMG_SIZE, config.IMG_SIZE)}, got {images.shape}"
    assert labels.shape == (
        config.BATCH_SIZE,
    ), f"Incorrect label shape. Expected {(config.BATCH_SIZE,)}, got {labels.shape}"

    print("Data Loading logic verified.")

    # 3. Model Instantiation and Forward Pass
    print("\n[Step 2] Initializing Model and Verifying Forward Pass...")

    model = model_lib.GroupedEfficientNet(pretrained=True)
    model.to(config.DEVICE)

    # Run a forward pass with the batch fetched earlier
    images = images.to(config.DEVICE)
    with torch.no_grad():
        logits = model(images)

    print(f"Logits Shape: {logits.shape}")

    # Assertions for Model Output
    assert logits.shape == (
        config.BATCH_SIZE,
        config.NUM_CLASSES,
    ), f"Incorrect output shape. Expected {(config.BATCH_SIZE, config.NUM_CLASSES)}, got {logits.shape}"

    print("Model architecture verified.")

    # 4. Training Loop Demonstration
    print("\n[Step 3] Running Training Loop (1 Epoch)...")

    # engine.train_model runs the loop, validates, and saves the best model
    trained_model = engine.train_model(train_loader, val_loader)

    # Verify that the model checkpoint was saved
    assert os.path.exists(
        config.MODEL_SAVE_PATH
    ), f"Model checkpoint not found at {config.MODEL_SAVE_PATH}"

    print("Training loop completed and model saved.")

    # 5. Inference and Submission
    print("\n[Step 4] Generating Submission...")

    # Generate submission using the test subset
    # This function loads the model from disk, runs inference, and saves to CSV
    engine.generate_submission(test_subset, model_path=config.MODEL_SAVE_PATH)

    # Verify Submission File
    assert os.path.exists(
        config.SUBMISSION_PATH
    ), f"Submission file not found at {config.SUBMISSION_PATH}"

    submission_df = pd.read_csv(config.SUBMISSION_PATH)
    print("Submission Head:")
    print(submission_df.head())

    # Assertions for Submission Format
    assert len(submission_df) == len(
        test_subset
    ), f"Submission row count mismatch. Expected {len(test_subset)}, got {len(submission_df)}"
    assert (
        "BraTS21ID" in submission_df.columns and "MGMT_value" in submission_df.columns
    ), "Submission file missing required columns."
    assert (
        submission_df["MGMT_value"].dtype == float
        or submission_df["MGMT_value"].dtype == np.float64
    ), "MGMT_value column should be float."

    print("Submission generation verified.")
    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
