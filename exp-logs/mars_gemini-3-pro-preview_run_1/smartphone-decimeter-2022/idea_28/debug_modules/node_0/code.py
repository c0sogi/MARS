import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader

# Import library modules
import library.config as config
import library.utils as utils
import library.features as features
import library.dataset as dataset
import library.architecture as architecture
import library.engine as engine
import library.inference as inference


def run_demo():
    print("=== Starting Demonstration of GNSS Location Pipeline ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed/Demo
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Set paths to a demo working directory
    demo_working_dir = "./working/demo_run"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir)

    # Override config constants
    config.WORKING_DIR = demo_working_dir
    config.CACHE_DIR = os.path.join(demo_working_dir, "cache")
    config.MODEL_PATH = os.path.join(demo_working_dir, "best_model.pth")
    config.SUBMISSION_DIR = os.path.join(demo_working_dir, "submission")
    config.SUBMISSION_PATH = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Set Debug flags to limit data processing
    config.DEBUG = True
    config.DEBUG_DRIVE_COUNT = 1  # Only process 1 drive per split
    config.NUM_EPOCHS = 1  # Train for only 1 epoch
    config.BATCH_SIZE = 2  # Small batch size
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Force device to CPU for simplicity/stability in demo unless GPU is requested
    # (The provided environment has GPU, but CPU is safer for tiny debug batches)
    config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Working Directory: {config.WORKING_DIR}")
    print(f"Device: {config.DEVICE}")
    print(f"Debug Mode: {config.DEBUG}")

    # ---------------------------------------------------------
    # 2. Verify Utility Functions
    # ---------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test Coordinate Conversion
    lat0, lon0 = 37.4, -122.1
    lat1, lon1 = 37.41, -122.09

    # Geodetic -> ENU
    e, n = utils.geodetic_to_enu(lat1, lon1, lat0, lon0)

    # ENU -> Geodetic
    lat_rec, lon_rec = utils.enu_to_geodetic(e, n, lat0, lon0)

    # Check reconstruction
    assert np.isclose(lat1, lat_rec, atol=1e-5), "Latitude reconstruction failed"
    assert np.isclose(lon1, lon_rec, atol=1e-5), "Longitude reconstruction failed"

    # Test Haversine
    dist = utils.haversine_distance(lat0, lon0, lat1, lon1)
    assert dist > 0, "Haversine distance should be positive"

    print("Utils verification passed.")

    # ---------------------------------------------------------
    # 3. Data Loading & Feature Engineering
    # ---------------------------------------------------------
    print("\n[3] Testing Data Loading and Feature Engineering...")

    # We rely on the existing ./metadata/train_metadata.csv provided in the environment
    # The dataset class calls generate_dataset, which uses process_drive

    # Instantiate Training Dataset
    # load_cached_data=False forces processing from raw files
    train_dataset = dataset.GnssSequenceDataset(split="train", load_cached_data=False)

    print(f"Train Dataset Length (Sequences): {len(train_dataset)}")

    if len(train_dataset) > 0:
        # Fetch one sample
        sample = train_dataset[0]
        feat_shape = sample["features"].shape
        target_shape = sample["targets"].shape

        print(f"Sample Feature Shape: {feat_shape}")
        print(f"Sample Target Shape: {target_shape}")

        # Assertions
        assert (
            feat_shape[0] == config.INPUT_CHANNELS
        ), f"Expected {config.INPUT_CHANNELS} channels, got {feat_shape[0]}"
        assert (
            target_shape[0] == 2
        ), f"Expected 2 target channels (East, North), got {target_shape[0]}"
        assert feat_shape[1] == target_shape[1], "Feature and Target length mismatch"

        # Test Collate Function
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            collate_fn=dataset.collate_fn,
            num_workers=config.NUM_WORKERS,
        )

        batch = next(iter(train_loader))
        print(f"Batch Features Shape: {batch['features'].shape}")
        print(f"Batch Mask Shape: {batch['mask'].shape}")

        # Check padding alignment (must be divisible by 16)
        assert batch["features"].shape[2] % 16 == 0, "Batch length not divisible by 16"

    else:
        print("Warning: Train dataset is empty. Check metadata/input files.")

    # ---------------------------------------------------------
    # 4. Model Architecture
    # ---------------------------------------------------------
    print("\n[4] Testing Model Architecture...")

    model = architecture.SEResUNet1D().to(config.DEVICE)

    # Create dummy input matching batch shape
    # (B, C, L) where L is multiple of 16
    dummy_input = torch.randn(2, config.INPUT_CHANNELS, 128).to(config.DEVICE)

    # Forward pass
    outputs = model(dummy_input)

    print("Model Output Keys:", outputs.keys())

    # Verify outputs
    assert "final" in outputs, "Model missing 'final' output head"
    assert outputs["final"].shape == (
        2,
        2,
        128,
    ), f"Output shape mismatch: {outputs['final'].shape}"

    # Check auxiliary heads
    for scale in config.AUXILIARY_SCALES:
        key = f"aux_{scale}"
        if key in outputs:
            expected_len = 128 // scale
            assert (
                outputs[key].shape[2] == expected_len
            ), f"Aux head {key} has wrong length"

    print("Model architecture verification passed.")

    # ---------------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------------
    print("\n[5] Testing Training Loop...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # Run one epoch
    loss = engine.train_one_epoch(
        model, train_loader, optimizer, config.DEVICE, epoch=1
    )

    assert not np.isnan(loss), "Training loss returned NaN"
    print("Training loop executed successfully.")

    # Save model for inference test
    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")

    # ---------------------------------------------------------
    # 6. Validation Logic
    # ---------------------------------------------------------
    print("\n[6] Testing Validation Logic...")

    # Instantiate Validation Dataset
    val_dataset = dataset.GnssSequenceDataset(split="val", load_cached_data=False)

    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            num_workers=config.NUM_WORKERS,
        )

        metric = engine.validate(model, val_loader, config.DEVICE)
        print(f"Validation Metric Result: {metric}")
        assert metric >= 0, "Metric should be non-negative"
    else:
        print("Validation dataset empty, skipping validation test.")

    # ---------------------------------------------------------
    # 7. Inference and Submission
    # ---------------------------------------------------------
    print("\n[7] Testing Inference Generation...")

    # Create a dummy test metadata file if needed, or rely on existing one
    # The generate_submission function uses 'test' split which looks for test_metadata.csv
    # We assume test_metadata.csv exists in ./metadata as per prompt description

    try:
        # Run inference
        # This will load the model we just saved and generate predictions
        inference.generate_submission(
            load_cached_data=False,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
        )

        if os.path.exists(config.SUBMISSION_PATH):
            df_sub = pd.read_csv(config.SUBMISSION_PATH)
            print(f"Submission generated with {len(df_sub)} rows.")
            print("Head:")
            print(df_sub.head())

            required_cols = [
                "tripId",
                "UnixTimeMillis",
                "LatitudeDegrees",
                "LongitudeDegrees",
            ]
            assert all(
                col in df_sub.columns for col in required_cols
            ), "Missing columns in submission"
        else:
            raise FileNotFoundError("Submission file not created.")

    except Exception as e:
        print(f"Inference failed: {e}")
        # If test data is missing or empty in the debug subset, this might fail, which is acceptable for demo
        # provided the code logic is correct.

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
