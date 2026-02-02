import os
import torch
import numpy as np
import pandas as pd
import shutil
from library.config import Config, seed_everything
from library.dataset import load_data, DenoisingDataset
from library.model import UNet
from library.train import run_training
from library.predict import run_inference
from library.utils import calculate_rmse


def verify_dataset_logic():
    print("\n--- Verifying Dataset Logic ---")
    # Use a small debug size
    debug_size = 5

    # Test loading data explicitly
    # We force load_cached_data=False to verify the raw loading logic
    ids, images = load_data(
        Config.TRAIN_METADATA_PATH,
        "input_path",
        "demo_train_in",
        load_cached_data=False,
        debug_size=debug_size,
    )

    print(f"Loaded {len(ids)} images.")
    assert len(ids) == debug_size, f"Expected {debug_size} IDs, got {len(ids)}"
    assert len(images) == debug_size, f"Expected {debug_size} images, got {len(images)}"
    assert isinstance(images[0], np.ndarray), "Loaded image is not a numpy array"

    # Test Dataset Class (Training Mode)
    # Create dummy targets matching inputs
    dataset_train = DenoisingDataset(images, images, patch_size=64, train_mode=True)
    item_in, item_tar = dataset_train[0]

    assert item_in.shape == (
        1,
        64,
        64,
    ), f"Expected train patch shape (1, 64, 64), got {item_in.shape}"
    assert item_tar.shape == (
        1,
        64,
        64,
    ), f"Expected target patch shape (1, 64, 64), got {item_tar.shape}"

    # Test Dataset Class (Validation Mode - No Crop)
    dataset_val = DenoisingDataset(images, images, train_mode=False)
    item_in_val, item_tar_val = dataset_val[0]

    # Shape should match original image (C, H, W)
    h, w = images[0].shape
    assert item_in_val.shape == (
        1,
        h,
        w,
    ), f"Expected val shape (1, {h}, {w}), got {item_in_val.shape}"
    print("Dataset logic verified.")


def verify_model_logic():
    print("\n--- Verifying Model Logic ---")
    device = Config.DEVICE
    model = UNet(n_channels=1, n_classes=1).to(device)

    # Create dummy input (Batch, Channel, Height, Width)
    # Dimensions must be divisible by 16 for UNet pooling operations
    dummy_input = torch.randn(2, 1, 256, 256).to(device)

    # Forward pass
    output = model(dummy_input)

    assert (
        output.shape == dummy_input.shape
    ), f"Output shape mismatch. Expected {dummy_input.shape}, got {output.shape}"
    print("Model forward pass verified.")


def verify_metric_logic():
    print("\n--- Verifying Metric Logic ---")
    y_true = np.array([0.0, 1.0, 0.5])
    y_pred = np.array([0.0, 1.0, 0.5])
    rmse = calculate_rmse(y_true, y_pred)
    assert rmse == 0.0, "RMSE should be 0 for identical arrays"

    y_pred_off = np.array([1.0, 0.0, 1.5])  # Errors: 1, 1, 1 -> MSE=1 -> RMSE=1
    rmse_off = calculate_rmse(y_true, y_pred_off)
    assert np.isclose(rmse_off, 1.0), f"RMSE should be 1.0, got {rmse_off}"
    print("Metric logic verified.")


def run_demo_pipeline():
    print("\n--- Running Full Training & Inference Pipeline ---")

    # 1. Modify Config for Speed
    # We monkey-patch the Config class attributes to run a minimal version
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG_SAMPLE_SIZE = 10  # Use only 10 images
    Config.EARLY_STOPPING_PATIENCE = 1

    # Ensure working directory is clean for this demo run to avoid cache conflicts
    # (Optional, but good for a standalone demo)
    demo_working_dir = "./working/demo_run"
    Config.WORKING_DIR = demo_working_dir
    Config.MODEL_SAVE_PATH = os.path.join(demo_working_dir, "unet_demo.pth")
    os.makedirs(demo_working_dir, exist_ok=True)

    # 2. Run Training
    # This covers: get_dataloaders, train_one_epoch, validate, save_checkpoint
    print("Step 1: Training...")
    run_training(
        num_epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        load_cached_data=False,  # Force reload to use DEBUG_SAMPLE_SIZE
        debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    )

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."

    # 3. Run Inference
    # This covers: load_checkpoint, generate_predictions, format_submission
    print("Step 2: Inference...")
    demo_submission_path = os.path.join(demo_working_dir, "demo_submission.csv")

    run_inference(
        model_path=Config.MODEL_SAVE_PATH,
        output_path=demo_submission_path,
        load_cached_data=False,
    )

    assert os.path.exists(demo_submission_path), "Submission file was not generated."

    # 4. Verify Submission Content
    print("Step 3: Verifying Submission...")
    df = pd.read_csv(demo_submission_path)
    print(f"Submission rows: {len(df)}")
    print(f"Submission columns: {list(df.columns)}")

    assert (
        "id" in df.columns and "value" in df.columns
    ), "Submission missing required columns."
    assert len(df) > 0, "Submission file is empty."

    # Check value range
    min_val = df["value"].min()
    max_val = df["value"].max()
    print(f"Value range: [{min_val}, {max_val}]")
    assert min_val >= 0 and max_val <= 1, "Pixel values out of range [0, 1]"

    print("Pipeline finished successfully.")


if __name__ == "__main__":
    # Set seed for reproducibility
    seed_everything(42)

    # Run individual component checks
    verify_dataset_logic()
    verify_model_logic()
    verify_metric_logic()

    # Run the integrated pipeline
    run_demo_pipeline()
