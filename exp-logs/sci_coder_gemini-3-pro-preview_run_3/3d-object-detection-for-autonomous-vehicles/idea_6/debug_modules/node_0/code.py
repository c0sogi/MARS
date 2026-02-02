import os
import torch
import numpy as np
import pandas as pd
import shutil
import warnings

# Import library components
from library.config import Config
from library.dataset import NuScenesLidarDataset
from library.model import TemporalPointPillars
from library.loss import DetectionLoss
from library.trainer import Trainer
from library.inference import InferenceEngine


def run_demo():
    print("=== Starting 3D Object Detection Library Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Setup
    # ---------------------------------------------------------
    print("\n[1] Setting up Configuration...")

    # Define a specific working directory for this demo
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config parameters for speed and demonstration purposes
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.GT_DATABASE_DIR = os.path.join(demo_dir, "gt_database")
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "model_checkpoint.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")
    Config.LOG_FILE = os.path.join(demo_dir, "train.log")

    # Reduce computational load
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.MAX_PILLARS_TRAIN = 1000  # Reduce max pillars for speed
    Config.MAX_PILLARS_TEST = 1000
    Config.AUG_USE_GT_SAMPLING = False  # Disable GT database generation (slow)
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set seeds for reproducibility
    Config.setup()
    print(f"Working Directory: {Config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Dataset & DataLoader Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Dataset and DataLoader...")

    # Initialize dataset with a small subset
    subset_size = 8
    dataset = NuScenesLidarDataset(
        mode="train", subset_size=subset_size, load_cached_data=False
    )

    print(f"Dataset initialized with {len(dataset)} samples.")

    # Fetch a single item
    sample = dataset[0]

    # Verify item structure
    required_keys = [
        "voxels",
        "coordinates",
        "num_points",
        "targets",
        "sample_token",
        "metadata",
    ]
    for key in required_keys:
        assert key in sample, f"Missing key in dataset sample: {key}"

    print("Single sample keys verified.")

    # Verify shapes
    voxels = sample["voxels"]
    coords = sample["coordinates"]

    # Voxels: (M, MaxPoints, Features)
    assert voxels.ndim == 3, f"Voxels should be 3D, got {voxels.ndim}"
    assert (
        voxels.shape[1] == Config.MAX_POINTS_PER_PILLAR
    ), "Incorrect max points per pillar"
    assert voxels.shape[2] == Config.NUM_POINT_FEATURES, "Incorrect feature dimension"

    # Coords: (M, 3) -> [z, y, x]
    assert coords.ndim == 2 and coords.shape[1] == 3, "Incorrect coordinates shape"

    # Test Collate Function via DataLoader
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, collate_fn=NuScenesLidarDataset.collate_fn
    )
    batch = next(iter(loader))

    print("Batch collation successful.")
    print(f"Batch Voxels Shape: {batch['voxels'].shape}")
    print(
        f"Batch Coordinates Shape: {batch['coordinates'].shape}"
    )  # Should be (TotalM, 4) [b, z, y, x]

    assert (
        batch["coordinates"].shape[1] == 4
    ), "Batch coordinates should have 4 columns (batch_idx, z, y, x)"

    # ---------------------------------------------------------
    # 3. Model & Loss Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Model and Loss...")

    device = torch.device(Config.DEVICE)
    model = TemporalPointPillars().to(device)
    criterion = DetectionLoss(Config).to(device)

    # Move batch to device
    voxels_dev = batch["voxels"].to(device)
    num_points_dev = batch["num_points"].to(device)
    coords_dev = batch["coordinates"].to(device)

    targets_dev = batch["targets"]
    for k, v in targets_dev.items():
        if isinstance(v, torch.Tensor):
            targets_dev[k] = v.to(device)

    # Forward Pass
    model.train()
    preds = model(voxels_dev, num_points_dev, coords_dev)

    # Verify Output Keys
    expected_heads = ["heatmap", "offset", "height", "dim", "rot"]
    for head in expected_heads:
        assert head in preds, f"Model output missing head: {head}"

    # Verify Heatmap Shape: (B, NumClasses, H, W)
    hm_shape = preds["heatmap"].shape
    grid_size = Config.get_grid_size()  # [W, H, D]
    expected_shape = (2, Config.NUM_CLASSES, grid_size[1], grid_size[0])  # B, C, H, W

    assert (
        hm_shape == expected_shape
    ), f"Heatmap shape mismatch. Expected {expected_shape}, got {hm_shape}"
    print(f"Model forward pass successful. Heatmap shape: {hm_shape}")

    # Calculate Loss
    loss, loss_dict = criterion(preds, targets_dev)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    print(f"Loss calculation successful. Total Loss: {loss.item():.4f}")
    print(f"Loss Components: {list(loss_dict.keys())}")

    # ---------------------------------------------------------
    # 4. Training Loop (Trainer)
    # ---------------------------------------------------------
    print("\n[4] Running Trainer (1 Epoch, Subset)...")

    # Initialize Trainer with a small subset
    trainer = Trainer(load_cached_data=False, subset_size=16)

    # Run training
    trainer.fit()

    # Verify checkpoint creation
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not created."
    print("Training complete. Checkpoint saved.")

    # ---------------------------------------------------------
    # 5. Inference (InferenceEngine)
    # ---------------------------------------------------------
    print("\n[5] Running Inference...")

    # Initialize Inference Engine with a small subset
    inference_engine = InferenceEngine(
        checkpoint_path=Config.MODEL_SAVE_PATH, subset_size=10
    )

    # Run inference
    inference_engine.run_inference()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(df)} rows.")
    print("First 2 rows of submission:")
    print(df.head(2))

    # Verify columns
    assert (
        "Id" in df.columns and "PredictionString" in df.columns
    ), "Submission columns mismatch"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Ensure input directory exists (sanity check for environment)
    if not os.path.exists("./input"):
        raise FileNotFoundError(
            "Input directory './input' not found. Please run in the correct environment."
        )

    run_demo()
