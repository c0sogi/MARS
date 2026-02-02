import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import read_dicom_robust, preprocess_image
from library.data import get_dataloader
from library.model import AsymmetricEfficientNet
from library.train import run_training, predict_and_submit


def main():
    print("=== Starting Demonstration Script ===")

    # --------------------------------------------------------------------------
    # 1. Configuration Setup for Speed
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast execution...")

    # Override Config defaults to run a minimal version of the task
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 10  # Use only 10 samples for training/testing
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_EPOCHS = 1  # Only 1 epoch
    Config.NUM_WORKERS = 0  # Disable multiprocessing for small data to avoid overhead

    # Use a separate working directory for this demo to avoid overwriting existing runs
    Config.WORKING_DIR = "./working/demo_run"
    Config.CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Working Dir: {Config.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 2. Verify Utility Functions (Data Ingestion)
    # --------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Load metadata to find a real file path
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    if len(df_train) == 0:
        raise ValueError("Training metadata is empty.")

    sample_row = df_train.iloc[0]
    flair_dir = os.path.join(Config.INPUT_DIR, sample_row["path_FLAIR"])

    # Find the first valid DICOM file in the directory
    dicom_files = [f for f in os.listdir(flair_dir) if "Image-" in f]
    if not dicom_files:
        print(
            "    Warning: No DICOM files found in sample directory. Skipping image read test."
        )
    else:
        sample_file_path = os.path.join(flair_dir, dicom_files[0])

        # Test Robust Read
        img = read_dicom_robust(sample_file_path)
        if img is None:
            print("    Warning: Failed to read sample DICOM.")
        else:
            # Test Preprocessing
            processed_img = preprocess_image(
                img, target_size=(Config.IMG_SIZE, Config.IMG_SIZE)
            )

            # Assertions
            assert processed_img.shape == (
                Config.IMG_SIZE,
                Config.IMG_SIZE,
            ), f"Preprocessing shape mismatch: {processed_img.shape}"
            assert (
                processed_img.dtype == np.float32
            ), f"Preprocessing dtype mismatch: {processed_img.dtype}"
            assert (
                0.0 <= processed_img.min() and processed_img.max() <= 1.00001
            ), "Image normalization failed (values outside [0, 1])."

            print("    Success: DICOM read and preprocessed correctly.")

    # --------------------------------------------------------------------------
    # 3. Verify Data Loading Pipeline
    # --------------------------------------------------------------------------
    print("\n[3] Verifying Data Loader...")

    # Create a train loader (this will also trigger ROI cache computation for the subset)
    train_loader = get_dataloader(
        "train", debug=True, batch_size=Config.BATCH_SIZE, num_workers=0
    )

    # Fetch a single batch
    try:
        images, labels = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("DataLoader returned no data.")

    print(f"    Batch Shapes -> Images: {images.shape}, Labels: {labels.shape}")

    # Assertions
    # Expected shape: (Batch, Channels=12, H, W)
    assert images.shape == (
        Config.BATCH_SIZE,
        Config.IN_CHANNELS,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image batch shape incorrect. Expected {(Config.BATCH_SIZE, Config.IN_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE)}, got {images.shape}"

    assert labels.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Label batch shape incorrect. Expected {(Config.BATCH_SIZE, 1)}, got {labels.shape}"

    assert images.dtype == torch.float32, "Images tensor is not float32."
    print("    Success: DataLoader yields correctly shaped tensors.")

    # --------------------------------------------------------------------------
    # 4. Verify Model Architecture & Forward Pass
    # --------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    device = Config.DEVICE
    model = AsymmetricEfficientNet().to(device)

    # Move batch to device
    images = images.to(device)

    # Perform forward pass
    logits = model(images)

    # Assertions
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape incorrect. Expected {(Config.BATCH_SIZE, 1)}, got {logits.shape}"

    print("    Success: Model forward pass completed.")

    # --------------------------------------------------------------------------
    # 5. Verify Training Loop (Integration)
    # --------------------------------------------------------------------------
    print("\n[5] Executing Training Loop (1 Epoch)...")

    # run_training handles the loop, validation, and saving the best model
    # We rely on the modified Config to keep this short
    trained_model = run_training()

    # Verify checkpoint creation
    assert os.path.exists(
        Config.CHECKPOINT_PATH
    ), f"Model checkpoint not found at {Config.CHECKPOINT_PATH}"

    print("    Success: Training loop finished and model saved.")

    # --------------------------------------------------------------------------
    # 6. Verify Inference & Submission
    # --------------------------------------------------------------------------
    print("\n[6] Executing Inference and Submission Generation...")

    predict_and_submit()

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_FILE
    ), f"Submission file not found at {Config.SUBMISSION_FILE}"

    # Verify submission content format
    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"    Submission shape: {sub_df.shape}")
    print(f"    Columns: {sub_df.columns.tolist()}")

    assert (
        "BraTS21ID" in sub_df.columns and "MGMT_value" in sub_df.columns
    ), "Submission file missing required columns."

    assert len(sub_df) > 0, "Submission file is empty."

    # Check if predictions are probabilities
    preds = sub_df["MGMT_value"]
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions are not valid probabilities (must be between 0 and 1)."

    print("    Success: Submission file generated and validated.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
