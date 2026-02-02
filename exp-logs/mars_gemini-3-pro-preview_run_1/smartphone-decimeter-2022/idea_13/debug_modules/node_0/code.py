import os
import shutil
import numpy as np
import torch
import pandas as pd
from library.config import Config
from library.utils import set_seed, WGS84Converter, haversine_distance
from library.data_processing import GNSSPreprocessor
from library.dataset import GNSSSequenceDataset
from library.model import MultiScaleResUNet1D
from library.loss import DeepSupervisionMAELoss
from library.train import train_pipeline


def test_utils():
    print("\n--- Testing Utils ---")
    converter = WGS84Converter()

    # Test 1: Zero distance
    lat, lon = 37.0, -122.0
    east, north = converter.deg_to_meters(lat, lon, lat, lon)
    assert np.isclose(east, 0.0) and np.isclose(
        north, 0.0
    ), "Zero distance conversion failed"

    # Test 2: Reversibility
    # Move ~111km north (1 degree lat) and ~88km east (1 degree lon at 37 lat)
    target_lat = lat + 1.0
    target_lon = lon + 1.0

    d_east, d_north = converter.deg_to_meters(target_lat, target_lon, lat, lon)
    rec_lat, rec_lon = converter.meters_to_deg(d_east, d_north, lat, lon)

    assert np.isclose(
        rec_lat, target_lat
    ), f"Lat reconstruction failed: {rec_lat} vs {target_lat}"
    assert np.isclose(
        rec_lon, target_lon
    ), f"Lon reconstruction failed: {rec_lon} vs {target_lon}"

    # Test 3: Haversine
    dist = haversine_distance(lat, lon, target_lat, lon)  # 1 deg lat difference
    # 1 deg lat is approx 111km
    assert 110000 < dist < 112000, f"Haversine distance unexpected: {dist}"
    print("Utils tests passed.")


def test_data_processing():
    print("\n--- Testing Data Processing (Debug Mode) ---")
    preprocessor = GNSSPreprocessor()

    # Force processing from scratch by disabling cache loading for this test
    # We use debug=True to only process a few rows from metadata
    train_df = preprocessor.process_data(
        Config.TRAIN_METADATA_PATH, "train_debug", load_cached_data=False, debug=True
    )

    assert not train_df.empty, "Processed dataframe is empty"
    assert "d_east" in train_df.columns, "Target column d_east missing"
    assert "d_north" in train_df.columns, "Target column d_north missing"
    assert "Cn0DbHz_mean" in train_df.columns, "Feature Cn0DbHz_mean missing"

    print(f"Processed DataFrame Shape: {train_df.shape}")
    print("Data Processing tests passed.")
    return train_df


def test_dataset_and_loader(df):
    print("\n--- Testing Dataset and DataLoader ---")
    feature_cols = Config.FEATURE_NAMES
    target_cols = ["d_east", "d_north"]

    dataset = GNSSSequenceDataset(
        df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        window_size=32,  # Small window for test
        stride=32,
        mode="train",
    )

    assert len(dataset) > 0, "Dataset is empty"

    item = dataset[0]
    features = item["features"]
    targets = item["targets"]
    mask = item["mask"]

    # Check shapes: (Channels, Time)
    assert features.shape == (
        len(feature_cols),
        32,
    ), f"Feature shape mismatch: {features.shape}"
    assert targets.shape == (2, 32), f"Target shape mismatch: {targets.shape}"
    assert mask.shape == (32,), f"Mask shape mismatch: {mask.shape}"

    print("Dataset tests passed.")


def test_model_and_loss():
    print("\n--- Testing Model and Loss ---")
    device = torch.device("cpu")  # Test on CPU for simplicity
    model = MultiScaleResUNet1D().to(device)
    criterion = DeepSupervisionMAELoss()

    # Dummy batch: (Batch, Features, Time)
    B, C, T = 2, Config.NUM_FEATURES, 64
    dummy_input = torch.randn(B, C, T).to(device)
    dummy_target = torch.randn(B, 2, T).to(device)
    dummy_mask = torch.ones(B, T).to(device)

    # Training mode (Deep Supervision)
    model.train()
    outputs = model(dummy_input)
    assert isinstance(outputs, list), "Model should return list in training mode"
    assert len(outputs) == 3, f"Expected 3 outputs (final + 2 aux), got {len(outputs)}"
    assert outputs[0].shape == (B, 2, T), f"Output shape mismatch: {outputs[0].shape}"

    loss = criterion(outputs, dummy_target, dummy_mask)
    assert loss.item() > 0, "Loss should be positive"

    # Eval mode
    model.eval()
    output = model(dummy_input)
    assert torch.is_tensor(output), "Model should return tensor in eval mode"
    assert output.shape == (B, 2, T)

    print("Model and Loss tests passed.")


def run_demonstration():
    # 1. Override Config for Speed
    print("Configuring demonstration parameters...")
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Speed optimizations
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.TRAIN_WINDOW_SIZE = 128  # Smaller window for speed
    Config.DEBUG_SAMPLE_SIZE = 50  # Only process 50 metadata rows

    # 2. Run Unit Tests
    test_utils()
    train_df = test_data_processing()
    test_dataset_and_loader(train_df)
    test_model_and_loss()

    # 3. Run Full Pipeline (Integration Test)
    print("\n--- Running Full Training Pipeline (Debug Mode) ---")
    # This will load data, train for 1 epoch, and run inference
    # Note: We use debug=True to ensure it uses the small subset logic in library functions
    train_pipeline(debug=True, epochs=1, patience=1)

    # 4. Verify Output
    submission_file = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if os.path.exists(submission_file):
        df_sub = pd.read_csv(submission_file)
        print(f"\nSubmission generated successfully with {len(df_sub)} rows.")
        print(df_sub.head())
    else:
        print("\nWarning: Submission file was not generated.")


if __name__ == "__main__":
    try:
        run_demonstration()
        print("\nDemonstration completed successfully.")
    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        raise e
