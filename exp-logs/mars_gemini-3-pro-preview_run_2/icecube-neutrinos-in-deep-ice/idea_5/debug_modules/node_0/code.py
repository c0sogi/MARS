import os
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config, seed_everything
from library.utils import (
    spherical_to_cartesian,
    cartesian_to_spherical,
    angular_dist_score,
)
from library.data_loader import get_dataloaders, IceCubeDataset
from library.model import DualStreamNetwork, predict_submission
from library.trainer import IceCubeTrainer
from torch.utils.data import DataLoader


def main():
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    print("============================================================")
    print("   IceCube Direction Prediction - Pipeline Demonstration    ")
    print("============================================================")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    print("\n[1] Setting up Configuration...")

    # Override Config parameters for a fast demonstration run
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Enable Debug mode to use a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 500  # Use only 500 events for train/val

    # Reduce training parameters for speed
    Config.BATCH_SIZE = 32
    Config.NUM_EPOCHS = 2
    Config.WARMUP_EPOCHS = 0  # No warmup for short run
    Config.NUM_WORKERS = 2  # Minimal workers
    Config.SEQ_LEN = 128  # Slightly shorter sequence for speed

    # Ensure directories exist
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Mode: {Config.DEBUG} (Subset Size: {Config.DEBUG_SUBSET_SIZE})")
    print(f"    Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test 1: Coordinate Conversion (Spherical -> Cartesian)
    # Azimuth=0, Zenith=pi/2 -> x=1, y=0, z=0
    az_t = torch.tensor([0.0])
    zen_t = torch.tensor([np.pi / 2])
    x, y, z = spherical_to_cartesian(az_t, zen_t)

    assert torch.allclose(x, torch.tensor([1.0]), atol=1e-5), "x coord mismatch"
    assert torch.allclose(y, torch.tensor([0.0]), atol=1e-5), "y coord mismatch"
    assert torch.allclose(z, torch.tensor([0.0]), atol=1e-5), "z coord mismatch"
    print("    Spherical -> Cartesian conversion: OK")

    # Test 2: Coordinate Conversion (Cartesian -> Spherical)
    # x=0, y=1, z=0 -> Azimuth=pi/2, Zenith=pi/2
    x_t, y_t, z_t = torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([0.0])
    az_res, zen_res = cartesian_to_spherical(x_t, y_t, z_t)

    assert torch.allclose(
        az_res, torch.tensor([np.pi / 2]), atol=1e-5
    ), "Azimuth mismatch"
    assert torch.allclose(
        zen_res, torch.tensor([np.pi / 2]), atol=1e-5
    ), "Zenith mismatch"
    print("    Cartesian -> Spherical conversion: OK")

    # Test 3: Angular Distance Score
    # Identical vectors -> 0 error
    score_perfect = angular_dist_score(
        np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0])
    )
    assert np.isclose(score_perfect, 0.0, atol=1e-5), "Perfect score mismatch"

    # Opposite vectors (Zenith 0 vs Pi) -> Pi error
    score_opp = angular_dist_score(
        np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([np.pi])
    )
    assert np.isclose(score_opp, np.pi, atol=1e-5), "Opposite score mismatch"
    print("    Angular Distance Metric: OK")

    # -------------------------------------------------------------------------
    # 3. Verify Data Loading
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Data Loading...")

    # Initialize Loaders (train and val)
    train_loader, val_loader = get_dataloaders(Config)

    print(f"    Train Loader Size: {len(train_loader)} batches")
    print(f"    Val Loader Size: {len(val_loader)} batches")

    # Fetch a single batch to verify structure
    try:
        seq_x, geom_x, targets, event_ids = next(iter(train_loader))
        print("    Batch fetched successfully.")
        print(
            f"    - Seq Features Shape: {seq_x.shape} (Expected: [{Config.BATCH_SIZE}, {Config.N_CHANNELS}, {Config.SEQ_LEN}])"
        )
        print(
            f"    - Geom Features Shape: {geom_x.shape} (Expected: [{Config.BATCH_SIZE}, {Config.NUM_GEOM_FEATURES}])"
        )
        print(
            f"    - Targets Shape: {targets.shape} (Expected: [{Config.BATCH_SIZE}, 2])"
        )

        # Assertions
        assert seq_x.shape == (Config.BATCH_SIZE, Config.N_CHANNELS, Config.SEQ_LEN)
        assert geom_x.shape == (Config.BATCH_SIZE, Config.NUM_GEOM_FEATURES)
        assert targets.shape == (Config.BATCH_SIZE, 2)

    except Exception as e:
        print(f"    Error fetching batch: {e}")
        raise e

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    model = DualStreamNetwork().to(Config.DEVICE)

    # Move batch to device
    seq_x = seq_x.to(Config.DEVICE)
    geom_x = geom_x.to(Config.DEVICE)

    # Forward pass
    with torch.no_grad():
        preds = model(seq_x, geom_x)

    print(f"    Prediction Shape: {preds.shape} (Expected: [{Config.BATCH_SIZE}, 3])")

    # Assertions
    assert preds.shape == (Config.BATCH_SIZE, 3)
    # Check if output is not NaN
    assert not torch.isnan(preds).any(), "Model output contains NaNs"
    print("    Model forward pass: OK")

    # -------------------------------------------------------------------------
    # 5. Verify Training Pipeline
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Training Pipeline...")

    trainer = IceCubeTrainer(Config, train_loader, val_loader)

    print("    Starting training loop (2 Epochs)...")
    best_model_path = trainer.fit()

    # Verify model file creation
    if os.path.exists(best_model_path):
        print(f"    Training completed. Model saved at: {best_model_path}")
    else:
        raise FileNotFoundError(f"Best model file not found at {best_model_path}")

    # -------------------------------------------------------------------------
    # 6. Verify Inference Pipeline
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Inference Pipeline...")

    # Create a test loader manually to limit size (get_test_dataloader defaults to full set)
    test_meta_path = os.path.join(Config.METADATA_DIR, "test_metadata.parquet")
    test_dataset = IceCubeDataset(
        metadata_path=test_meta_path,
        mode="test",
        limit_size=100,  # Only predict 100 events
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"    Test Dataset Size: {len(test_dataset)}")

    # Run prediction
    predict_submission(Config, test_loader, best_model_path)

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"    Submission file generated at: {Config.SUBMISSION_PATH}")
        print(f"    Submission Shape: {sub_df.shape}")
        print(f"    Columns: {list(sub_df.columns)}")

        # Assertions
        assert len(sub_df) == 100, f"Expected 100 predictions, got {len(sub_df)}"
        assert list(sub_df.columns) == ["event_id", "azimuth", "zenith"]
        assert not sub_df.isnull().values.any(), "Submission contains NaNs"
        print("    Submission content: OK")
    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    print("\n============================================================")
    print("   Demonstration Completed Successfully!                    ")
    print("============================================================")


if __name__ == "__main__":
    main()
