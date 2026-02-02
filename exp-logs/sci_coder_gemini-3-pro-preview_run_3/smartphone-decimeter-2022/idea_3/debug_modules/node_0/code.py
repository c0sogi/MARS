import os
import pandas as pd
import numpy as np
import torch
import shutil

# Import library modules
from library import config
from library import utils
from library import data_factory
from library import dataset
from library import model
from library import trainer


def run_demonstration():
    print("--- Starting Library Demonstration ---")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Demo
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Define a temporary working directory for this demo
    DEMO_WORKING_DIR = "./working/demo_run"
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    # Monkey-patch the config module to use demo settings
    config.WORKING_DIR = DEMO_WORKING_DIR
    config.EPOCHS = 2  # Run only 2 epochs
    config.BATCH_SIZE = 2  # Small batch size
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    config.HIDDEN_SIZE = 32  # Smaller model for speed
    config.MAX_SEQUENCE_LENGTH = 100  # Truncate sequences for speed

    # Update paths to point to our demo metadata (to be created)
    config.TRAIN_METADATA_PATH = "./working/demo_train_meta.csv"
    config.VAL_METADATA_PATH = "./working/demo_val_meta.csv"
    config.TEST_METADATA_PATH = "./working/demo_test_meta.csv"

    # Update cache paths
    config.TRAIN_CACHE_PATH = os.path.join(DEMO_WORKING_DIR, "train_meta.parquet")
    config.VAL_CACHE_PATH = os.path.join(DEMO_WORKING_DIR, "val_meta.parquet")
    config.TEST_CACHE_PATH = os.path.join(DEMO_WORKING_DIR, "test_meta.parquet")

    print(f"Working directory set to: {config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Preparation (Sampling Metadata)
    # -------------------------------------------------------------------------
    print("\n[2] Sampling metadata for lightweight datasets...")

    # Load original metadata
    orig_train_meta = pd.read_csv("./metadata/train_metadata.csv")
    orig_val_meta = pd.read_csv("./metadata/val_metadata.csv")
    orig_test_meta = pd.read_csv("./metadata/test_metadata.csv")

    # Sample 2 trips for training
    train_trips = orig_train_meta["tripId"].unique()[:2]
    demo_train_meta = orig_train_meta[
        orig_train_meta["tripId"].isin(train_trips)
    ].copy()
    demo_train_meta.to_csv(config.TRAIN_METADATA_PATH, index=False)
    print(
        f"Created demo train metadata with {len(demo_train_meta)} rows (Trips: {len(train_trips)})"
    )

    # Sample 1 trip for validation
    val_trips = orig_val_meta["tripId"].unique()[:1]
    demo_val_meta = orig_val_meta[orig_val_meta["tripId"].isin(val_trips)].copy()
    demo_val_meta.to_csv(config.VAL_METADATA_PATH, index=False)
    print(
        f"Created demo val metadata with {len(demo_val_meta)} rows (Trips: {len(val_trips)})"
    )

    # Sample 1 trip for testing
    test_trips = orig_test_meta["tripId"].unique()[:1]
    demo_test_meta = orig_test_meta[orig_test_meta["tripId"].isin(test_trips)].copy()
    demo_test_meta.to_csv(config.TEST_METADATA_PATH, index=False)
    print(
        f"Created demo test metadata with {len(demo_test_meta)} rows (Trips: {len(test_trips)})"
    )

    # -------------------------------------------------------------------------
    # 3. Data Processing (Data Factory)
    # -------------------------------------------------------------------------
    print("\n[3] Processing raw data using data_factory...")

    # Process datasets (this will load raw CSVs, aggregate features, and save parquets)
    # We force load_cached_data=False to ensure the processing logic runs
    train_df = data_factory.process_dataset(
        config.TRAIN_METADATA_PATH, config.TRAIN_CACHE_PATH, load_cached_data=False
    )
    val_df = data_factory.process_dataset(
        config.VAL_METADATA_PATH, config.VAL_CACHE_PATH, load_cached_data=False
    )
    test_df = data_factory.process_dataset(
        config.TEST_METADATA_PATH, config.TEST_CACHE_PATH, load_cached_data=False
    )

    # Verify processing results
    assert len(train_df) > 0, "Train DataFrame is empty"
    assert len(val_df) > 0, "Val DataFrame is empty"
    assert len(test_df) > 0, "Test DataFrame is empty"

    required_cols = config.FEATURE_NAMES + ["dLat", "dLon", "lat_wls", "lon_wls"]
    for col in required_cols:
        assert col in train_df.columns, f"Missing column {col} in processed data"

    print("Data processing successful. Parquet files created.")

    # -------------------------------------------------------------------------
    # 4. Dataset and DataLoader
    # -------------------------------------------------------------------------
    print("\n[4] Initializing Datasets and DataLoaders...")

    train_loader, val_loader, test_loader, scaler = dataset.get_dataloaders(
        batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS
    )

    # Fetch one batch to verify
    batch = next(iter(train_loader))
    features = batch["features"]
    targets = batch["targets"]
    mask = batch["mask"]

    print(f"Batch Features Shape: {features.shape} (Batch, SeqLen, Features)")
    print(f"Batch Targets Shape: {targets.shape} (Batch, SeqLen, 2)")
    print(f"Batch Mask Shape: {mask.shape} (Batch, SeqLen)")

    assert features.shape[0] == config.BATCH_SIZE
    assert features.shape[2] == config.INPUT_SIZE
    assert targets.shape[2] == config.OUTPUT_SIZE
    assert mask.shape == features.shape[:2]

    print("DataLoader verification successful.")

    # -------------------------------------------------------------------------
    # 5. Model Initialization
    # -------------------------------------------------------------------------
    print("\n[5] Initializing ResidualBiLSTM Model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.ResidualBiLSTM(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        output_size=config.OUTPUT_SIZE,
    ).to(device)

    # Test forward pass
    features = features.to(device)
    lengths = batch[
        "lengths"
    ]  # Keep lengths on CPU for pack_padded_sequence if needed, or move if model handles it

    outputs = net(features, lengths)
    print(f"Model Output Shape: {outputs.shape}")

    assert outputs.shape == targets.shape, "Model output shape mismatch"
    print("Model initialization and forward pass successful.")

    # -------------------------------------------------------------------------
    # 6. Training Loop
    # -------------------------------------------------------------------------
    print("\n[6] Running Training Loop...")

    # Run training using the trainer module
    best_score = trainer.run_training(
        train_loader,
        val_loader,
        epochs=config.EPOCHS,
        learning_rate=1e-3,
        device_name="cuda" if torch.cuda.is_available() else "cpu",
    )

    model_path = os.path.join(config.WORKING_DIR, "model_best.pth")
    assert os.path.exists(model_path), "Best model file was not saved."
    print(f"Training complete. Best Val Score: {best_score}")

    # -------------------------------------------------------------------------
    # 7. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n[7] Generating Submission...")

    submission_output_path = os.path.join(DEMO_WORKING_DIR, "demo_submission.csv")
    trainer.generate_submission(
        test_loader,
        model_path,
        submission_output_path,
        device_name="cuda" if torch.cuda.is_available() else "cpu",
    )

    assert os.path.exists(submission_output_path), "Submission file not created."
    sub_df = pd.read_csv(submission_output_path)
    print(f"Submission generated with {len(sub_df)} rows.")

    # Verify submission format
    expected_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch."

    # -------------------------------------------------------------------------
    # 8. Utility Functions Verification
    # -------------------------------------------------------------------------
    print("\n[8] Verifying Utility Functions...")

    # Test Haversine
    lat1, lon1 = 0.0, 0.0
    lat2, lon2 = 1.0, 0.0
    dist = utils.haversine_loss(
        np.array([lat1]), np.array([lon1]), np.array([lat2]), np.array([lon2])
    )
    # 1 degree latitude is approx 111km
    print(f"Haversine distance (0,0) to (1,0): {dist[0]:.2f} meters")
    assert 110000 < dist[0] < 112000, "Haversine calculation incorrect"

    # Test ECEF to Geodetic
    # Approximate Earth radius on X axis
    x, y, z = 6378137.0, 0.0, 0.0
    lat, lon, alt = utils.ecef_to_geodetic(np.array([x]), np.array([y]), np.array([z]))
    print(
        f"ECEF ({x}, {y}, {z}) -> Lat: {lat[0]:.4f}, Lon: {lon[0]:.4f}, Alt: {alt[0]:.4f}"
    )
    assert (
        abs(lat[0]) < 1e-5 and abs(lon[0]) < 1e-5
    ), "ECEF to Geodetic conversion incorrect"

    # Test Calc Score
    # Create dummy predictions and ground truth
    df_pred = pd.DataFrame(
        {
            "tripId": ["trip1", "trip1"],
            "UnixTimeMillis": [1000, 2000],
            "LatitudeDegrees": [0.0, 0.0],
            "LongitudeDegrees": [0.0, 0.0],
        }
    )
    df_gt = pd.DataFrame(
        {
            "tripId": ["trip1", "trip1"],
            "UnixTimeMillis": [1000, 2000],
            "LatitudeDegrees": [0.0001, 0.0002],  # Small offset
            "LongitudeDegrees": [0.0, 0.0],
        }
    )

    score = utils.calc_score(df_pred, df_gt)
    print(f"Calculated Score for dummy data: {score:.4f}")
    assert score > 0, "Score calculation failed (should be > 0)"

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    run_demonstration()
