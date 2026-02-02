import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import logging

# Import from the provided library
from library.config import Config
from library import utils, data_loader, model, train_eval


def run_demo():
    print("=== Starting MGMT Classification Pipeline Demo ===\n")

    # --------------------------------------------------------------------------
    # 1. Configuration Override for Demo
    # --------------------------------------------------------------------------
    print("[1] Configuring pipeline for rapid demonstration...")

    # Define a temporary directory for this demo run
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override global Config attributes to run a fast, small-scale test
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission_demo.csv")
    Config.BEST_MODEL_PATH = os.path.join(DEMO_DIR, "best_model_demo.pth")

    # Update cache paths to use the demo directory
    Config.CACHE_TRAIN_DATA = os.path.join(DEMO_DIR, "train_demo.npy")
    Config.CACHE_TRAIN_LABELS = os.path.join(DEMO_DIR, "train_labels.npy")
    Config.CACHE_VAL_DATA = os.path.join(DEMO_DIR, "val_demo.npy")
    Config.CACHE_VAL_LABELS = os.path.join(DEMO_DIR, "val_labels.npy")
    Config.CACHE_TEST_DATA = os.path.join(DEMO_DIR, "test_demo.npy")

    # Hyperparameters for speed
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.DEBUG_LIMIT = 6  # Process only 6 subjects (enough for train/val split)
    Config.EARLY_STOPPING_PATIENCE = 1

    # Ensure reproducibility
    utils.seed_everything(Config.SEED)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(
        f"    Epochs: {Config.NUM_EPOCHS}, Batch Size: {Config.BATCH_SIZE}, Debug Limit: {Config.DEBUG_LIMIT}"
    )
    print("    Configuration updated successfully.\n")

    # --------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # --------------------------------------------------------------------------
    print("[2] Verifying Data Pipeline...")

    # Load data (force regeneration of cache to test processing logic)
    # Note: We use load_cached_data=False to ensure we test the DICOM reading logic
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        load_cached_data=False, debug_limit=Config.DEBUG_LIMIT
    )

    # Fetch one batch from training loader
    images, labels = next(iter(train_loader))

    # Assertions
    # Expected Shape: (Batch_Size, Channels, H, W)
    # Channels = 4 modalities * 3 slices = 12
    expected_channels = 12
    expected_size = Config.IMG_SIZE

    print(f"    Train Batch Shape: {images.shape}")
    print(f"    Labels Shape: {labels.shape}")

    assert images.shape[0] <= Config.BATCH_SIZE, "Batch size exceeds configuration."
    assert (
        images.shape[1] == expected_channels
    ), f"Expected {expected_channels} channels, got {images.shape[1]}"
    assert (
        images.shape[2] == expected_size and images.shape[3] == expected_size
    ), f"Expected image size {expected_size}x{expected_size}, got {images.shape[2]}x{images.shape[3]}"
    assert (
        labels.shape[0] == images.shape[0]
    ), "Mismatch between images and labels count."

    print("    Data Pipeline verification passed.\n")

    # --------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("[3] Verifying Model Architecture...")

    # Initialize model
    net = model.AsymmetricEfficientNet()

    # Check Stem Modification
    # The first layer should be a Conv2d with in_channels=12 and groups=4
    stem = net.backbone.features[0][0]
    print(f"    Stem Layer: {stem}")

    assert isinstance(stem, torch.nn.Conv2d), "Stem is not a Conv2d layer."
    assert (
        stem.in_channels == expected_channels
    ), f"Stem input channels mismatch. Expected {expected_channels}, got {stem.in_channels}"
    assert (
        stem.groups == Config.STEM_GROUPS
    ), f"Stem groups mismatch. Expected {Config.STEM_GROUPS}, got {stem.groups}"

    # Test Forward Pass
    device = torch.device("cpu")  # Keep on CPU for simple demo check
    net.to(device)
    with torch.no_grad():
        output = net(images)

    print(f"    Forward Pass Output Shape: {output.shape}")
    assert output.shape == (
        images.shape[0],
        1,
    ), "Output shape mismatch. Expected (Batch, 1)."

    print("    Model Architecture verification passed.\n")

    # --------------------------------------------------------------------------
    # 4. Full Training Loop Simulation
    # --------------------------------------------------------------------------
    print("[4] Running Training Loop (Simulation)...")

    # Run the training routine provided in the library
    # We pass load_cached_data=True now because we just generated the cache in step 2
    train_eval.run_training(debug_limit=Config.DEBUG_LIMIT, load_cached_data=True)

    # Check if best model was saved
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), f"Best model file not found at {Config.BEST_MODEL_PATH}"
    print(f"    Training complete. Model saved to {Config.BEST_MODEL_PATH}")
    print("    Training Loop verification passed.\n")

    # --------------------------------------------------------------------------
    # 5. Inference and Submission Generation
    # --------------------------------------------------------------------------
    print("[5] Generating Submission...")

    # Run the submission generation routine
    train_eval.generate_submission(
        load_cached_data=True, debug_limit=Config.DEBUG_LIMIT
    )

    # Verify Submission File
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("    Submission File Head:")
    print(df_sub.head())

    # Check columns
    expected_cols = ["BraTS21ID", "MGMT_value"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Check values are probabilities
    if not df_sub.empty:
        assert df_sub["MGMT_value"].min() >= 0.0, "Probabilities < 0 found."
        assert df_sub["MGMT_value"].max() <= 1.0, "Probabilities > 1 found."

    print("    Submission Generation verification passed.\n")

    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Suppress verbose logging from libraries for cleaner demo output
    logging.getLogger("MGMT_Pipeline").setLevel(logging.WARNING)
    logging.getLogger("Data_Loader").setLevel(logging.WARNING)
    logging.getLogger("Model").setLevel(logging.WARNING)
    logging.getLogger("Train_Eval").setLevel(logging.INFO)  # Keep high level info

    try:
        run_demo()
    except AssertionError as e:
        print(f"\n!!! ASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
