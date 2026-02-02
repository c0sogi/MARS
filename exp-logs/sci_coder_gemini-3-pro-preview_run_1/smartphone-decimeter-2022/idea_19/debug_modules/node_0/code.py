import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.geo_utils import wgs84_to_ecef, ecef_to_wgs84, wgs84_to_enu
from library.data_processor import DataProcessor
from library.dataset import GNSSHeatmapDataset
from library.model import SkyResUNet
from library.trainer import Trainer
from library.inference import generate_submission

# Suppress warnings and progress bars for cleaner output
warnings.filterwarnings("ignore")
os.environ["TQDM_DISABLE"] = "1"


def test_geo_utils():
    print("\n[1] Testing Geo Utils...")

    # Test 1: WGS84 <-> ECEF Roundtrip
    lat, lon, alt = 37.4219999, -122.0840575, 10.0
    x, y, z = wgs84_to_ecef(lat, lon, alt)
    lat_out, lon_out, alt_out = ecef_to_wgs84(x, y, z)

    assert np.isclose(lat, lat_out, atol=1e-5), f"Lat mismatch: {lat} vs {lat_out}"
    assert np.isclose(lon, lon_out, atol=1e-5), f"Lon mismatch: {lon} vs {lon_out}"
    assert np.isclose(alt, alt_out, atol=1e-3), f"Alt mismatch: {alt} vs {alt_out}"
    print("  - WGS84 <-> ECEF roundtrip passed.")

    # Test 2: WGS84 -> ENU (Relative to self should be 0)
    e, n, u = wgs84_to_enu(lat, lon, lat, lon, alt)
    assert (
        np.isclose(e, 0, atol=1e-3)
        and np.isclose(n, 0, atol=1e-3)
        and np.isclose(u, 0, atol=1e-3)
    ), f"ENU self-reference failed: {e}, {n}, {u}"
    print("  - ENU self-reference passed.")
    print("  Geo Utils verified.")


def setup_demo_config():
    print("\n[2] Setting up Demo Configuration...")

    # Define a working directory for the demo
    demo_dir = "./working/demo_run"
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config parameters for speed
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.MODEL_DIR = os.path.join(demo_dir, "models")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")

    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.WINDOW_SIZE = 32  # Reduce window size for speed
    Config.STRIDE = 32

    print(f"  - Working Directory: {Config.WORKING_DIR}")
    print(f"  - Epochs: {Config.EPOCHS}")
    print(f"  - Batch Size: {Config.BATCH_SIZE}")

    return demo_dir


def create_mini_metadata(demo_dir):
    print("\n[3] Creating Mini Metadata...")

    # Load original metadata
    train_meta_orig = pd.read_csv("./metadata/train_metadata.csv")
    test_meta_orig = pd.read_csv("./metadata/test_metadata.csv")

    # Select a small subset of drives
    # We pick one drive for train, one for val (from train_meta), one for test
    unique_drives = train_meta_orig["drive_id"].unique()
    if len(unique_drives) >= 2:
        train_drive = unique_drives[0]
        val_drive = unique_drives[1]
    else:
        train_drive = unique_drives[0]
        val_drive = unique_drives[0]

    test_drives = test_meta_orig["drive_id"].unique()
    test_drive = test_drives[0] if len(test_drives) > 0 else None

    print(f"  - Selected Train Drive: {train_drive}")
    print(f"  - Selected Val Drive:   {val_drive}")
    print(f"  - Selected Test Drive:  {test_drive}")

    # Filter DataFrames
    mini_train = train_meta_orig[train_meta_orig["drive_id"] == train_drive].head(
        200
    )  # Limit rows
    mini_val = train_meta_orig[train_meta_orig["drive_id"] == val_drive].head(100)

    if test_drive:
        mini_test = test_meta_orig[test_meta_orig["drive_id"] == test_drive].head(100)
    else:
        mini_test = test_meta_orig.head(100)

    # Save to demo directory
    train_path = os.path.join(demo_dir, "mini_train_meta.csv")
    val_path = os.path.join(demo_dir, "mini_val_meta.csv")
    test_path = os.path.join(demo_dir, "mini_test_meta.csv")

    mini_train.to_csv(train_path, index=False)
    mini_val.to_csv(val_path, index=False)
    mini_test.to_csv(test_path, index=False)

    # Override Config paths
    Config.TRAIN_METADATA_PATH = train_path
    Config.VAL_METADATA_PATH = val_path
    Config.TEST_METADATA_PATH = test_path

    print("  - Mini metadata files created and Config updated.")


def test_data_processing():
    print("\n[4] Testing Data Processor & Dataset...")

    # Initialize Dataset (this triggers DataProcessor)
    # load_cached_data=False ensures we actually run the processing logic
    ds = GNSSHeatmapDataset(split="train", load_cached_data=False)

    print(f"  - Dataset length (windows): {len(ds)}")

    if len(ds) > 0:
        item = ds[0]
        feats = item["features"]
        targets = item["targets"]
        mask = item["mask"]

        print(
            f"  - Feature shape: {feats.shape} (Expected: {Config.INPUT_CHANNELS}, {Config.WINDOW_SIZE}, {Config.AZIMUTH_BINS})"
        )
        print(f"  - Target shape: {targets.shape} (Expected: {Config.WINDOW_SIZE}, 2)")
        print(f"  - Mask shape: {mask.shape} (Expected: {Config.WINDOW_SIZE})")

        # Validation
        assert feats.shape == (
            Config.INPUT_CHANNELS,
            Config.WINDOW_SIZE,
            Config.AZIMUTH_BINS,
        ), "Feature shape mismatch"
        assert targets.shape == (Config.WINDOW_SIZE, 2), "Target shape mismatch"
        assert mask.shape == (Config.WINDOW_SIZE,), "Mask shape mismatch"

        print("  Data processing verified.")
    else:
        print("  Warning: Dataset is empty. Check input data availability.")


def test_model_architecture():
    print("\n[5] Testing Model Architecture...")

    model = SkyResUNet()
    # Create dummy input: (Batch, Channels, Time, Azimuth)
    dummy_input = torch.randn(
        2, Config.INPUT_CHANNELS, Config.WINDOW_SIZE, Config.AZIMUTH_BINS
    )

    # Test Training Mode (Deep Supervision)
    model.train()
    outputs = model(dummy_input)
    assert isinstance(outputs, dict), "Training output should be a dict"
    assert "main" in outputs, "Missing main head output"
    print(f"  - Training output keys: {list(outputs.keys())}")
    print(
        f"  - Main output shape: {outputs['main'].shape} (Expected: 2, {Config.WINDOW_SIZE}, 2)"
    )

    # Test Eval Mode
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    assert torch.is_tensor(output), "Eval output should be a tensor"
    assert output.shape == (
        2,
        Config.WINDOW_SIZE,
        2,
    ), f"Eval output shape mismatch: {output.shape}"

    print("  Model architecture verified.")


def run_training_demo():
    print("\n[6] Running Training Demo...")

    # Load datasets
    train_ds = GNSSHeatmapDataset(split="train", load_cached_data=True)
    val_ds = GNSSHeatmapDataset(split="val", load_cached_data=False)

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("  Skipping training due to empty datasets.")
        return

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    trainer = Trainer()
    trainer.fit(train_loader, val_loader)

    # Verify model saved
    if os.path.exists(trainer.model_path):
        print(f"  - Model checkpoint saved at {trainer.model_path}")
    else:
        print("  - Error: Model checkpoint not found.")


def run_inference_demo():
    print("\n[7] Running Inference Demo...")

    # Create a dummy sample_submission.csv in the input folder is not possible (read-only).
    # The inference code reads from Config.INPUT_DIR/sample_submission.csv.
    # The merge logic uses a left join on the sample submission.
    # Since we reduced the test metadata, the inference will only produce predictions for a few rows.
    # The final output will be the full sample submission size, with most rows filled by baseline/NaN
    # (though the code fills with predictions where available).

    # We need to ensure the Trainer can load the model we just trained
    if not os.path.exists(os.path.join(Config.MODEL_DIR, "best_model.pth")):
        print("  Skipping inference: No trained model found.")
        return

    try:
        generate_submission(load_cached_data=False)

        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        if os.path.exists(sub_path):
            df_sub = pd.read_csv(sub_path)
            print(f"  - Submission generated with {len(df_sub)} rows.")
            print("  Inference pipeline verified.")
        else:
            print("  - Error: Submission file not generated.")

    except Exception as e:
        print(f"  - Inference failed with error: {e}")
        # This might happen if sample_submission.csv doesn't match our mini test set logic perfectly,
        # but for a demo, we catch and report.


if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Test Utilities
    test_geo_utils()

    # 2. Setup Config
    demo_dir = setup_demo_config()

    # 3. Create Data Subsets
    create_mini_metadata(demo_dir)

    # 4. Test Data Processing
    test_data_processing()

    # 5. Test Model
    test_model_architecture()

    # 6. Run Training
    run_training_demo()

    # 7. Run Inference
    run_inference_demo()

    print("\nDemo completed successfully.")
