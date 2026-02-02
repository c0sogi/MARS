import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library
import library.config as config_module
from library.utils import WGS84, seed_everything
from library.preprocessing import GNSSPreprocessor
from library.dataset import GnssSequenceDataset
from library.model import PhaseAwareAttentionResUNet
from library.loss import DecimatedMAELoss
from library.trainer import Trainer
import library.inference as inference_module


def run_demo():
    print("=== Starting Phase-Aware ResUNet Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Define demo paths
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

    print(f"Working directory: {DEMO_DIR}")

    # Monkey-patch the Config class to use demo paths and smaller parameters
    # This affects all instances created hereafter
    config_module.Config.WORKING_DIR = DEMO_DIR
    config_module.Config.CACHE_DIR = CACHE_DIR
    config_module.Config.SUBMISSION_DIR = SUBMISSION_DIR
    config_module.Config.SUBMISSION_PATH = os.path.join(
        SUBMISSION_DIR, "submission.csv"
    )

    config_module.Config.TRAIN_METADATA_PATH = os.path.join(
        DEMO_DIR, "mini_train_meta.csv"
    )
    config_module.Config.VAL_METADATA_PATH = os.path.join(DEMO_DIR, "mini_val_meta.csv")
    config_module.Config.TEST_METADATA_PATH = os.path.join(
        DEMO_DIR, "mini_test_meta.csv"
    )

    # Speed up training for demo
    config_module.Config.NUM_EPOCHS = 1
    config_module.Config.TRAIN_SEQUENCE_LENGTH = 64  # Shorter sequences
    config_module.Config.BATCH_SIZE = 4
    config_module.Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Initialize Config instance (creates dirs)
    cfg = config_module.Config()
    seed_everything(cfg.RANDOM_SEED)

    # -------------------------------------------------------------------------
    # 2. Create Mini Metadata (Data Subset)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Creating mini metadata files...")

    # Load original metadata
    orig_train_meta_path = "./metadata/train_metadata.csv"
    orig_test_meta_path = "./metadata/test_metadata.csv"

    if not os.path.exists(orig_train_meta_path) or not os.path.exists(
        orig_test_meta_path
    ):
        raise FileNotFoundError("Original metadata files not found in ./metadata")

    df_train_full = pd.read_csv(orig_train_meta_path)
    df_test_full = pd.read_csv(orig_test_meta_path)

    # Sample 1 drive for train, 1 for val (from train set), 1 for test
    # We pick drives that exist
    train_drives = df_train_full["drive_id"].unique()
    test_drives = df_test_full["drive_id"].unique()

    # Select first available drive for consistency
    train_drive = train_drives[0]
    val_drive = train_drives[
        0
    ]  # Use same drive for val to ensure data exists, split logic handles overlap usually but here we force split
    test_drive = test_drives[0]

    print(f"  Selected Train Drive: {train_drive}")
    print(f"  Selected Test Drive:  {test_drive}")

    # Filter data
    # Take first 500 points for train, next 200 for val (simulating split)
    df_train_mini = (
        df_train_full[df_train_full["drive_id"] == train_drive].head(500).copy()
    )
    df_val_mini = (
        df_train_full[df_train_full["drive_id"] == train_drive].iloc[500:700].copy()
    )

    # For test, take first 200 points
    df_test_mini = df_test_full[df_test_full["drive_id"] == test_drive].head(200).copy()

    # Save mini metadata
    df_train_mini.to_csv(cfg.TRAIN_METADATA_PATH, index=False)
    df_val_mini.to_csv(cfg.VAL_METADATA_PATH, index=False)
    df_test_mini.to_csv(cfg.TEST_METADATA_PATH, index=False)

    print(f"  Saved mini_train_meta.csv ({len(df_train_mini)} rows)")
    print(f"  Saved mini_val_meta.csv ({len(df_val_mini)} rows)")
    print(f"  Saved mini_test_meta.csv ({len(df_test_mini)} rows)")

    # -------------------------------------------------------------------------
    # 3. Verify Utils
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying WGS84 Utils...")
    # Known point: Googleplex
    lat, lon, alt = 37.4220, -122.0841, 10.0
    x, y, z = WGS84.geodetic_to_ecef(lat, lon, alt)
    lat_rec, lon_rec, alt_rec = WGS84.ecef_to_geodetic(x, y, z)

    print(f"  Original: ({lat}, {lon}, {alt})")
    print(f"  Recovered: ({lat_rec:.4f}, {lon_rec:.4f}, {alt_rec:.4f})")

    assert np.isclose(lat, lat_rec, atol=1e-5), "Latitude conversion failed"
    assert np.isclose(lon, lon_rec, atol=1e-5), "Longitude conversion failed"

    # Test metric conversion
    dn, de = WGS84.latlon_to_meters(0.0001, 0.0001, lat)  # Small offset
    dlat, dlon = WGS84.meters_to_latlon(dn, de, lat)
    assert np.isclose(dlat, 0.0001, atol=1e-8), "Metric conversion failed"
    print("  WGS84 Utils Verified.")

    # -------------------------------------------------------------------------
    # 4. Preprocessing
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Preprocessing...")
    preprocessor = GNSSPreprocessor()

    # Generate Train
    train_df = preprocessor.generate_dataset(split="train", load_cached_data=False)
    print(f"  Processed Train Data: {train_df.shape}")
    assert not train_df.empty, "Train dataframe is empty"
    assert "DeltaNorthMeters" in train_df.columns, "Target column missing"

    # Generate Val
    val_df = preprocessor.generate_dataset(split="val", load_cached_data=False)
    print(f"  Processed Val Data: {val_df.shape}")

    # Generate Test
    test_df = preprocessor.generate_dataset(split="test", load_cached_data=False)
    print(f"  Processed Test Data: {test_df.shape}")
    assert not test_df.empty, "Test dataframe is empty"

    # -------------------------------------------------------------------------
    # 5. Dataset & DataLoader
    # -------------------------------------------------------------------------
    print("\n[Step 5] Initializing Dataset...")
    train_dataset = GnssSequenceDataset(train_df, split="train", config=cfg)

    print(f"  Dataset size (sequences): {len(train_dataset)}")
    sample = train_dataset[0]

    feat_shape = sample["features"].shape
    target_shape = sample["targets"].shape

    print(f"  Sample Feature Shape: {feat_shape}")
    print(f"  Sample Target Shape: {target_shape}")

    # Expected: [Channels, Length]
    assert (
        feat_shape[0] == cfg.INPUT_CHANNELS
    ), f"Expected {cfg.INPUT_CHANNELS} input channels"
    assert (
        target_shape[0] == cfg.OUTPUT_CHANNELS
    ), f"Expected {cfg.OUTPUT_CHANNELS} output channels"
    assert (
        feat_shape[1] == cfg.TRAIN_SEQUENCE_LENGTH
    ), "Sequence length mismatch (padded)"

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    # -------------------------------------------------------------------------
    # 6. Model & Loss
    # -------------------------------------------------------------------------
    print("\n[Step 6] Initializing Model and Loss...")
    model = PhaseAwareAttentionResUNet(config=cfg)
    criterion = DecimatedMAELoss(config=cfg)

    # Dummy forward pass
    dummy_input = torch.randn(
        2, cfg.INPUT_CHANNELS, cfg.TRAIN_SEQUENCE_LENGTH
    )  # Batch=2
    dummy_target = torch.randn(2, cfg.OUTPUT_CHANNELS, cfg.TRAIN_SEQUENCE_LENGTH)
    dummy_mask = torch.ones(2, cfg.TRAIN_SEQUENCE_LENGTH, dtype=torch.bool)

    model.train()
    outputs = model(dummy_input)

    # Check deep supervision output structure
    if cfg.USE_DEEP_SUPERVISION:
        assert isinstance(
            outputs, tuple
        ), "Model should return tuple in training mode with deep supervision"
        final_out, aux_outs = outputs
        print(f"  Final Output Shape: {final_out.shape}")
        print(f"  Num Aux Outputs: {len(aux_outs)}")
    else:
        final_out = outputs

    loss, metrics = criterion(outputs, dummy_target, dummy_mask)
    print(f"  Computed Loss: {loss.item():.4f}")
    print(f"  Loss Metrics: {metrics.keys()}")

    # -------------------------------------------------------------------------
    # 7. Training Loop
    # -------------------------------------------------------------------------
    print("\n[Step 7] Running Trainer...")
    trainer = Trainer(config=cfg)

    # We pass the already processed data via cache (it was saved in step 4)
    # The trainer will reload it from cache
    trained_model = trainer.train_model(load_cached_data=True, epochs=1)

    best_model_path = os.path.join(cfg.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint not saved"
    print("  Training finished successfully.")

    # -------------------------------------------------------------------------
    # 8. Inference
    # -------------------------------------------------------------------------
    print("\n[Step 8] Running Inference...")

    # We need to ensure sample_submission.csv exists for the merge step in inference
    # Since we are using a subset, we should create a dummy sample_submission matching our mini test set
    dummy_sample_sub = pd.DataFrame(
        {
            "tripId": [
                f"{r.drive_id}-{r.phone_name}" for _, r in df_test_mini.iterrows()
            ],
            "UnixTimeMillis": df_test_mini["UnixTimeMillis"],
            "LatitudeDegrees": 0.0,
            "LongitudeDegrees": 0.0,
        }
    )
    dummy_sub_path = os.path.join(cfg.INPUT_DIR, "sample_submission.csv")

    # NOTE: The input dir is read-only. We cannot write sample_submission.csv there.
    # The inference code checks config.INPUT_DIR for sample_submission.
    # We must patch config.INPUT_DIR to point to our demo dir where we can write a dummy sample sub.
    # However, the preprocessor uses INPUT_DIR to find raw files (test/...).
    # So we can't simply change INPUT_DIR globally without breaking raw file loading.

    # Workaround: The inference code handles missing sample_submission by saving raw predictions.
    # "Warning: sample_submission.csv not found. Saving generated predictions directly."
    # This is acceptable for the demo.

    # But wait, we want to verify it works.
    # Let's temporarily mock the path in the inference function call or let it fallback.
    # The `generate_submission` method in `inference.py` uses `config.INPUT_DIR`.
    # Let's just let it fallback to saving raw predictions, which is fine.

    trainer.generate_submission(model=trained_model, load_cached_data=True)

    assert os.path.exists(cfg.SUBMISSION_PATH), "Submission file not generated"

    sub_df = pd.read_csv(cfg.SUBMISSION_PATH)
    print(f"  Submission generated with {len(sub_df)} rows.")
    print("  Head:")
    print(sub_df.head())

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
