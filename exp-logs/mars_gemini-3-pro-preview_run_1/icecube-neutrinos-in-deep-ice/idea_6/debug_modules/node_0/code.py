import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import (
    set_seed,
    spherical_to_cartesian,
    cartesian_to_spherical,
    angular_dist_score,
)
from library.data_processing import IceCubeDataset, collate_fn
from library.model_architecture import DynGTNet
from library.training_engine import Trainer
from library.inference_engine import generate_submission


def create_subset_metadata(output_dir):
    """
    Creates small subsets of the original metadata files for demonstration purposes.
    """
    print("Creating metadata subsets for rapid demonstration...")
    os.makedirs(output_dir, exist_ok=True)

    # Define paths
    orig_train = "./metadata/train_metadata.parquet"
    orig_val = "./metadata/val_metadata.parquet"
    orig_test = "./metadata/test_metadata.parquet"

    # Read and slice (taking 200 train, 50 val, 50 test events)
    # This ensures the script runs very fast
    df_train = pd.read_parquet(orig_train).head(200)
    df_val = pd.read_parquet(orig_val).head(50)
    df_test = pd.read_parquet(orig_test).head(50)

    # Save to new paths
    new_train_path = os.path.join(output_dir, "train_subset.parquet")
    new_val_path = os.path.join(output_dir, "val_subset.parquet")
    new_test_path = os.path.join(output_dir, "test_subset.parquet")

    df_train.to_parquet(new_train_path, index=False)
    df_val.to_parquet(new_val_path, index=False)
    df_test.to_parquet(new_test_path, index=False)

    print(f"Subsets saved to {output_dir}")
    return new_train_path, new_val_path, new_test_path


def verify_utils():
    """
    Verifies the mathematical utility functions.
    """
    print("\n=== Verifying Utility Functions ===")

    # Test 1: Coordinate Conversion (Z-axis)
    # Zenith=0 implies pointing up (0, 0, 1)
    az, zen = 0.0, 0.0
    x, y, z = spherical_to_cartesian(az, zen)
    assert (
        np.isclose(x, 0.0) and np.isclose(y, 0.0) and np.isclose(z, 1.0)
    ), f"Failed Z-axis conversion. Got ({x}, {y}, {z})"

    # Test 2: Inverse Conversion
    az_rec, zen_rec = cartesian_to_spherical(x, y, z)
    assert np.isclose(az_rec, 0.0) and np.isclose(
        zen_rec, 0.0
    ), f"Failed Inverse conversion. Got az={az_rec}, zen={zen_rec}"

    # Test 3: X-axis
    # Azimuth=0, Zenith=pi/2 -> (1, 0, 0)
    az, zen = 0.0, np.pi / 2
    x, y, z = spherical_to_cartesian(az, zen)
    assert (
        np.isclose(x, 1.0) and np.isclose(y, 0.0) and np.isclose(z, 0.0, atol=1e-6)
    ), f"Failed X-axis conversion. Got ({x}, {y}, {z})"

    # Test 4: Angular Distance Score
    # Distance between (0,0) and (0,0) should be 0
    score = angular_dist_score([[0, 0]], [[0, 0]])
    assert np.isclose(score, 0.0), f"Self distance should be 0, got {score}"

    # Distance between Z-up (zen=0) and Z-down (zen=pi) should be pi
    score = angular_dist_score([[0, 0]], [[0, np.pi]])
    assert np.isclose(score, np.pi), f"Opposite distance should be pi, got {score}"

    print("Utils verification passed.")


def main():
    # 1. Setup
    set_seed(42)

    # Create a working directory for this demo
    demo_dir = "./working/demo_execution"
    meta_subset_dir = os.path.join(demo_dir, "metadata")

    # 2. Prepare Data Subsets
    train_path, val_path, test_path = create_subset_metadata(meta_subset_dir)

    # 3. Override Config for Speed and Demo Isolation
    print("\n=== Configuring Environment ===")
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.MODEL_DIR = os.path.join(demo_dir, "model_checkpoints")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")

    # Point to subsets
    Config.TRAIN_META_PATH = train_path
    Config.VAL_META_PATH = val_path
    Config.TEST_META_PATH = test_path

    # Reduce Model Complexity for Demo Speed
    Config.HIDDEN_CHANNELS = 32
    Config.NUM_HEADS = 4
    Config.NUM_LAYERS = 2
    Config.K_KNN = 5

    # Reduce Data Sampling Complexity
    Config.N_PULSES = 64
    Config.K_CAUSAL = 16
    Config.M_SIGNAL = 32

    # Reduce Training Loop
    Config.BATCH_SIZE = 16
    Config.MAX_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Initialize directories
    Config.setup()
    print("Configuration updated for demo execution.")

    # 4. Verify Utils
    verify_utils()

    # 5. Verify Data Processing
    print("\n=== Verifying Data Processing ===")
    # Instantiate dataset
    train_ds = IceCubeDataset(mode="train")
    print(f"Train dataset size: {len(train_ds)}")

    # Fetch single item
    sample = train_ds[0]
    x, y, eid = sample["x"], sample["y"], sample["event_id"]

    print(f"Sample shapes - Input: {x.shape}, Target: {y.shape}")

    # Assertions
    assert x.shape == (
        Config.N_PULSES,
        Config.IN_CHANNELS,
    ), f"Input shape mismatch. Expected ({Config.N_PULSES}, {Config.IN_CHANNELS}), got {x.shape}"
    assert y.shape == (2,), f"Target shape mismatch. Expected (2,), got {y.shape}"
    assert isinstance(eid, (int, np.integer)), "Event ID should be integer"

    # Test DataLoader
    loader = DataLoader(train_ds, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    assert batch["x"].shape == (4, Config.N_PULSES, Config.IN_CHANNELS)
    assert batch["y"].shape == (4, 2)
    print("Data processing verification passed.")

    # 6. Verify Model Architecture
    print("\n=== Verifying Model Architecture ===")
    model = DynGTNet()
    model.eval()

    with torch.no_grad():
        # Pass the batch from previous step
        out = model(batch["x"])

    print(f"Model output shape: {out.shape}")
    assert out.shape == (
        4,
        3,
    ), f"Model output mismatch. Expected (4, 3), got {out.shape}"
    print("Model architecture verification passed.")

    # 7. Verify Training Loop
    print("\n=== Verifying Training Engine ===")
    # Initialize datasets for trainer
    val_ds = IceCubeDataset(mode="val")
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn)

    trainer = Trainer(model, loader, val_loader, Config)

    # Run training
    print("Running training for 1 epoch...")
    trainer.fit()

    # Check if model checkpoint was created
    best_model_path = os.path.join(Config.MODEL_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not saved."
    print("Training engine verification passed.")

    # 8. Verify Inference Engine
    print("\n=== Verifying Inference Engine ===")
    # We can use the generate_submission function directly
    # It internally re-loads the test dataset and the best model

    # Ensure test metadata is set up correctly (done in step 3)

    submission_df = generate_submission(batch_size=Config.BATCH_SIZE, num_workers=0)

    # Assertions on submission
    assert not submission_df.empty, "Submission DataFrame is empty."
    assert list(submission_df.columns) == [
        "event_id",
        "azimuth",
        "zenith",
    ], f"Submission columns mismatch. Got {submission_df.columns}"
    assert (
        len(submission_df) == 50
    ), f"Expected 50 predictions (from subset), got {len(submission_df)}"

    # Check output file
    sub_csv_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(sub_csv_path), "Submission CSV file not found."

    print("Inference engine verification passed.")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    main()
