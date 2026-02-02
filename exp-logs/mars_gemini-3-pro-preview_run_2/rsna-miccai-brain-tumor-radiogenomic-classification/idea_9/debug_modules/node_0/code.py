import os
import sys
import pandas as pd
import numpy as np
import torch
import shutil

# Import from the provided library files
from library.config import Config
from library.data import get_dataloaders, MIPDataset
from library.model import AsymmetricEfficientNet
from library.train import train
from library.inference import predict


def run_demo():
    print(
        "=== Starting Demonstration of MGMT Promoter Methylation Prediction Pipeline ===\n"
    )

    # --------------------------------------------------------------------------
    # 1. Setup & Configuration Patching
    # --------------------------------------------------------------------------
    print("--- Step 1: Configuration & Subset Creation ---")

    # Define a temporary working directory for this demo
    DEMO_DIR = "./working/demo_run"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Patch the global Config class to use this directory and run quickly
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")
    Config.BEST_MODEL_PATH = os.path.join(DEMO_DIR, "best_model.pth")

    # Reduce image size and batch size for speed
    Config.IMG_SIZE = 128
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 2

    # Create subset metadata files to avoid processing the entire dataset
    # We take a small number of samples from the existing metadata
    subset_size_train = 8
    subset_size_val = 4
    subset_size_test = 4

    print("Creating data subsets...")

    # Train Subset
    df_train = pd.read_csv("./metadata/train.csv")
    df_train_sub = df_train.head(subset_size_train).copy()
    train_sub_path = os.path.join(DEMO_DIR, "train_subset.csv")
    df_train_sub.to_csv(train_sub_path, index=False)
    Config.TRAIN_METADATA_PATH = train_sub_path
    Config.CACHE_TRAIN_PATH = os.path.join(DEMO_DIR, "train_cache.npy")

    # Val Subset
    df_val = pd.read_csv("./metadata/val.csv")
    df_val_sub = df_val.head(subset_size_val).copy()
    val_sub_path = os.path.join(DEMO_DIR, "val_subset.csv")
    df_val_sub.to_csv(val_sub_path, index=False)
    Config.VAL_METADATA_PATH = val_sub_path
    Config.CACHE_VAL_PATH = os.path.join(DEMO_DIR, "val_cache.npy")

    # Test Subset
    df_test = pd.read_csv("./metadata/test.csv")
    df_test_sub = df_test.head(subset_size_test).copy()
    test_sub_path = os.path.join(DEMO_DIR, "test_subset.csv")
    df_test_sub.to_csv(test_sub_path, index=False)
    Config.TEST_METADATA_PATH = test_sub_path
    Config.CACHE_TEST_PATH = os.path.join(DEMO_DIR, "test_cache.npy")

    print(
        f"Subsets created. Train: {len(df_train_sub)}, Val: {len(df_val_sub)}, Test: {len(df_test_sub)}"
    )

    # --------------------------------------------------------------------------
    # 2. Data Loading & Processing Verification
    # --------------------------------------------------------------------------
    print("\n--- Step 2: Verifying Data Processing & Loading ---")

    # Force processing (load_cached_data=False) to test the DICOM -> MIP pipeline
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=False
    )

    # Fetch a single batch to verify shapes
    images, labels = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    # Assertions
    # Shape should be (Batch_Size, In_Channels, H, W)
    # In_Channels = 4 modalities * 3 slabs = 12
    expected_channels = 12
    assert images.shape == (
        Config.BATCH_SIZE,
        expected_channels,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape. Expected {(Config.BATCH_SIZE, expected_channels, Config.IMG_SIZE, Config.IMG_SIZE)}, got {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect label shape. Expected {(Config.BATCH_SIZE,)}, got {labels.shape}"

    print("Data loading logic verified successfully.")

    # --------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n--- Step 3: Verifying Model Architecture ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AsymmetricEfficientNet(in_channels=expected_channels)
    model.to(device)

    # Run a forward pass with the batch fetched earlier
    with torch.no_grad():
        output = model(images.to(device))

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    # Output should be (Batch_Size, Num_Classes) -> (B, 1) usually for binary BCEWithLogits
    # Note: The model definition returns raw logits.
    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Incorrect model output shape. Expected {(Config.BATCH_SIZE, 1)}, got {output.shape}"

    print("Model architecture verified successfully.")

    # --------------------------------------------------------------------------
    # 4. Training Pipeline
    # --------------------------------------------------------------------------
    print("\n--- Step 4: Running Training Loop ---")

    # We use the training function provided in library.train
    # It will use the Config paths we patched earlier.
    # We enable caching now, as the data was processed in Step 2.
    train(epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE, load_cached_data=True)

    # Verify checkpoint creation
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Training failed to create model checkpoint at {Config.BEST_MODEL_PATH}"
        )

    print("Training completed and checkpoint saved.")

    # --------------------------------------------------------------------------
    # 5. Inference Pipeline
    # --------------------------------------------------------------------------
    print("\n--- Step 5: Running Inference Loop ---")

    # Run prediction using the trained model
    predict(load_cached_data=True)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Inference failed to create submission file at {Config.SUBMISSION_PATH}"
        )

    # Verify submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission File Head:")
    print(df_sub.head())

    assert len(df_sub) == len(
        df_test_sub
    ), f"Submission row count mismatch. Expected {len(df_test_sub)}, got {len(df_sub)}"
    assert (
        "BraTS21ID" in df_sub.columns and "MGMT_value" in df_sub.columns
    ), "Submission file missing required columns."

    print("Inference completed successfully.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    # Set fixed seed for the entire run
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        run_demo()
    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        sys.exit(1)
