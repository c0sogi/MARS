import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import library components
from library.config import Config
from library.dataset import NuScenesDataset
from library.model import PillarUNet3D
from library.loss import CenterPointLoss
from library.train import train_model
from library.inference import generate_submission
from library.utils import (
    get_transformation_matrix,
    get_quaternion_from_yaw,
    transform_points,
)


def run_demonstration():
    print("=== Starting 3D Object Detection Library Demo ===")

    # ==============================================================================
    # 1. Configuration Setup & Overrides
    # ==============================================================================
    print("\n[1] Setting up Configuration...")

    # Initialize directories
    Config.setup()

    # Override Config for speed and demonstration purposes
    Config.NUM_EPOCHS = 1
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 4  # Use a tiny subset for instant execution
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for this small demo
    Config.MAX_PILLARS = 1000  # Reduce memory footprint

    # Set seeds for reproducibility in this script
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    print("Configuration configured for fast demonstration.")

    # ==============================================================================
    # 2. Dataset Verification
    # ==============================================================================
    print("\n[2] Verifying Dataset Logic...")

    # Instantiate dataset in training mode
    dataset = NuScenesDataset(is_train=True, load_cached_data=False)

    # Verify dataset length matches debug limit
    assert (
        len(dataset) == Config.DEBUG_SAMPLE_SIZE
    ), f"Dataset length {len(dataset)} does not match debug limit {Config.DEBUG_SAMPLE_SIZE}"

    # Fetch a single sample
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")

    # Verify essential keys exist
    required_keys = [
        "voxels",
        "coordinates",
        "num_points",
        "hm",
        "target_reg",
        "ind",
        "mask",
        "cat",
    ]
    for key in required_keys:
        assert key in sample, f"Missing key '{key}' in dataset sample."

    # Verify Voxel shapes
    # voxels: (Num_Pillars, Max_Points, Features)
    assert sample["voxels"].ndim == 3
    assert sample["voxels"].shape[1] == Config.MAX_POINTS_PER_PILLAR
    assert sample["voxels"].shape[2] == Config.NUM_POINT_FEATURES

    # Test Collate Function
    print("Testing collate_fn...")
    batch_list = [dataset[0], dataset[1]]
    batch = NuScenesDataset.collate_fn(batch_list)

    # Verify batch structure
    assert "batch_size" in batch
    assert batch["batch_size"] == 2
    # Coordinates should have 4 columns: (batch_idx, z, y, x)
    assert batch["coordinates"].shape[1] == 4

    print("Dataset verification passed.")

    # ==============================================================================
    # 3. Model Verification
    # ==============================================================================
    print("\n[3] Verifying Model Architecture...")

    device = Config.DEVICE
    model = PillarUNet3D().to(device)

    # Move batch to device
    batch_gpu = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_gpu[k] = v.to(device)
        else:
            batch_gpu[k] = v

    # Perform Forward Pass
    print("Running forward pass...")
    with torch.no_grad():
        preds = model(batch_gpu)

    # Verify Output
    assert "hm" in preds
    assert "reg" in preds

    # Check output shapes
    # Heatmap: (B, Num_Classes, H, W)
    # Regression: (B, 8, H, W)
    B = batch["batch_size"]
    C = Config.NUM_CLASSES
    H = Config.GRID_SIZE[1]
    W = Config.GRID_SIZE[0]

    assert preds["hm"].shape == (
        B,
        C,
        H,
        W,
    ), f"Expected hm shape {(B, C, H, W)}, got {preds['hm'].shape}"
    assert preds["reg"].shape == (
        B,
        8,
        H,
        W,
    ), f"Expected reg shape {(B, 8, H, W)}, got {preds['reg'].shape}"

    print("Model verification passed.")

    # ==============================================================================
    # 4. Loss Verification
    # ==============================================================================
    print("\n[4] Verifying Loss Calculation...")

    loss_fn = CenterPointLoss()

    # Compute loss
    loss, stats = loss_fn(preds, batch_gpu)

    # Verify loss is scalar and valid
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert not torch.isnan(loss).any(), "Loss contains NaN values"

    print(f"Calculated Loss: {loss.item():.4f}")
    print(f"Loss Stats: {stats}")
    print("Loss verification passed.")

    # ==============================================================================
    # 5. Training Loop Integration Test
    # ==============================================================================
    print("\n[5] Running Training Loop (Integration Test)...")

    # Run the provided training function
    # This will train for 1 epoch on 4 samples
    train_model(
        num_epochs=1,
        debug_limit=Config.DEBUG_SAMPLE_SIZE,
        load_cached_data=False,
        patience=1,
    )

    # Verify that checkpoints were created
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "latest_model.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"

    print("Training loop completed successfully.")

    # ==============================================================================
    # 6. Inference Integration Test
    # ==============================================================================
    print("\n[6] Running Inference (Integration Test)...")

    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Run inference using the model we just "trained"
    generate_submission(checkpoint_path, output_path=submission_path)

    # Verify submission file
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    assert "Id" in df_sub.columns
    assert "PredictionString" in df_sub.columns
    # Check that we have rows corresponding to the test set (or debug subset of it)
    print(f"Submission generated with {len(df_sub)} rows.")

    print("Inference verification passed.")

    # ==============================================================================
    # 7. Utilities Verification
    # ==============================================================================
    print("\n[7] Verifying Utility Functions...")

    # Test Transformation Matrix Construction
    # Rotate 90 degrees around Z axis, translate by (10, 0, 0)
    yaw = np.pi / 2
    rot_quat = get_quaternion_from_yaw(yaw)
    translation = [10.0, 0.0, 0.0]

    mat = get_transformation_matrix(translation, rot_quat)

    # Test Point Transformation
    # Point at (1, 0, 0)
    # Rotated 90 deg -> (0, 1, 0)
    # Translated (10, 0, 0) -> (10, 1, 0)
    points = np.array([[1.0, 0.0, 0.0]])
    transformed = transform_points(points, mat)

    expected = np.array([[10.0, 1.0, 0.0]])
    assert np.allclose(
        transformed, expected, atol=1e-5
    ), f"Transformation failed. Expected {expected}, got {transformed}"

    print("Utility verification passed.")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
