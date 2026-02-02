import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import set_seed, get_logger, meters_to_latlon
from library.data_loader import load_data, GNSSWindowDataset
from library.model import SkyStateTransformer
from library.train import run_training

# Setup logger
logger = get_logger("demo")


def setup_demo_config():
    """
    Overrides Config parameters for a fast demonstration run.
    """
    logger.info("Setting up demo configuration...")

    # Use a specific working directory for the demo
    demo_dir = "./working/demo"
    os.makedirs(demo_dir, exist_ok=True)

    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Enable Debug mode to use a small subset of trips (5 trips)
    Config.DEBUG = True

    # Reduce training parameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 32
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Update cache paths to the demo directory
    Config.CACHE_TRAIN_X_SEQ = os.path.join(demo_dir, "train_X_seq.npy")
    Config.CACHE_TRAIN_X_SKY = os.path.join(demo_dir, "train_X_sky.npy")
    Config.CACHE_TRAIN_Y = os.path.join(demo_dir, "train_y.npy")
    Config.CACHE_TRAIN_META = os.path.join(demo_dir, "train_meta.parquet")

    Config.CACHE_VAL_X_SEQ = os.path.join(demo_dir, "val_X_seq.npy")
    Config.CACHE_VAL_X_SKY = os.path.join(demo_dir, "val_X_sky.npy")
    Config.CACHE_VAL_Y = os.path.join(demo_dir, "val_y.npy")
    Config.CACHE_VAL_META = os.path.join(demo_dir, "val_meta.parquet")

    Config.CACHE_TEST_X_SEQ = os.path.join(demo_dir, "test_X_seq.npy")
    Config.CACHE_TEST_X_SKY = os.path.join(demo_dir, "test_X_sky.npy")
    Config.CACHE_TEST_META = os.path.join(demo_dir, "test_meta.parquet")

    Config.CACHE_SCALER = os.path.join(demo_dir, "scaler.json")
    Config.MODEL_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    logger.info(f"Working directory set to: {Config.WORKING_DIR}")
    logger.info(f"Debug mode: {Config.DEBUG}")


def demonstrate_training():
    """
    Demonstrates the training pipeline.
    """
    logger.info("\n=== Starting Training Demonstration ===")

    # Run training
    # load_cached=False ensures we process the data from scratch using the DEBUG subset
    best_loss = run_training(
        epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, load_cached=False
    )

    logger.info(f"Training finished. Best Validation Loss: {best_loss:.4f}")

    # Verify model file exists
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError("Model file was not saved.")

    # Verify cache files exist
    if not os.path.exists(Config.CACHE_TRAIN_X_SEQ):
        raise FileNotFoundError("Training data cache not found.")

    logger.info("Training demonstration successful.")


def demonstrate_inference():
    """
    Demonstrates the inference pipeline manually on the test subset.
    We avoid the full 'generate_predictions' function to skip the time-consuming
    WLS fallback on the full dataset, focusing on verifying the model's predictions.
    """
    logger.info("\n=== Starting Inference Demonstration ===")

    # 1. Load Test Data (Subset due to DEBUG=True)
    logger.info("Loading processed test data...")
    # load_data handles loading from the cache generated during training (if test was processed)
    # or processing it now. Since we ran run_training, train/val are cached. Test might not be.
    # We force load_cached=False to ensure we process the test subset now.
    (_, _, test_data) = load_data(load_cached_data=False)
    test_X_seq, test_X_sky, test_meta = test_data

    logger.info(f"Test Subset Size: {len(test_X_seq)}")
    if len(test_X_seq) == 0:
        logger.warning(
            "No test windows generated (likely due to short trips in debug sample). Skipping inference check."
        )
        return

    # 2. Setup DataLoader
    test_dataset = GNSSWindowDataset(test_X_seq, test_X_sky, None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Load Model
    device = torch.device(Config.DEVICE)
    model = SkyStateTransformer().to(device)

    logger.info(f"Loading model weights from {Config.MODEL_PATH}")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # 4. Predict
    logger.info("Running prediction...")
    all_preds = []
    with torch.no_grad():
        for batch_seq, batch_sky in test_loader:
            batch_seq = batch_seq.to(device)
            batch_sky = batch_sky.to(device)

            # Verify input shapes
            assert (
                batch_seq.ndim == 3
            ), f"Expected 3D input for sequence, got {batch_seq.shape}"
            assert (
                batch_sky.ndim == 2
            ), f"Expected 2D input for sky, got {batch_sky.shape}"

            outputs = model(batch_seq, batch_sky)

            # Verify output shape
            assert (
                outputs.shape[1] == 2
            ), f"Expected output dim 2 (East, North), got {outputs.shape}"

            all_preds.append(outputs.cpu().numpy())

    predictions_meters = np.concatenate(all_preds, axis=0)
    logger.info(f"Predictions shape: {predictions_meters.shape}")

    # 5. Reconstruct Coordinates
    logger.info("Reconstructing Lat/Lon coordinates...")
    wls_lat = test_meta["WlsLat"].values
    wls_lon = test_meta["WlsLon"].values

    # Ensure lengths match
    assert len(wls_lat) == len(
        predictions_meters
    ), "Mismatch between metadata and predictions length."

    delta_east = predictions_meters[:, 0]
    delta_north = predictions_meters[:, 1]

    pred_lat, pred_lon = meters_to_latlon(delta_north, delta_east, wls_lat, wls_lon)

    # 6. Basic Validation of Results
    # Check for NaNs
    if np.isnan(pred_lat).any() or np.isnan(pred_lon).any():
        raise AssertionError("NaNs detected in predicted coordinates.")

    # Check bounds (rough check for valid lat/lon)
    if not ((-90 <= pred_lat).all() and (pred_lat <= 90).all()):
        raise AssertionError("Predicted Latitude out of bounds.")
    if not ((-180 <= pred_lon).all() and (pred_lon <= 180).all()):
        raise AssertionError("Predicted Longitude out of bounds.")

    # Create a small submission dataframe
    sub_df = pd.DataFrame(
        {
            "tripId": test_meta["tripId"],
            "UnixTimeMillis": test_meta["UnixTimeMillis"],
            "LatitudeDegrees": pred_lat,
            "LongitudeDegrees": pred_lon,
        }
    )

    logger.info("Sample Predictions:")
    print(sub_df.head())

    # Save this demo submission
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Demo submission saved to {Config.SUBMISSION_PATH}")

    logger.info("Inference demonstration successful.")


if __name__ == "__main__":
    # 1. Setup
    setup_demo_config()
    set_seed(Config.RANDOM_STATE)

    # 2. Train
    demonstrate_training()

    # 3. Inference
    demonstrate_inference()

    print("\nAll demonstrations completed successfully.")
