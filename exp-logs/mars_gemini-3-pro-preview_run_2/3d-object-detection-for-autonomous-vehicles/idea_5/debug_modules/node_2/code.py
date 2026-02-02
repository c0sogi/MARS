import os
import torch
import numpy as np
import shutil
import warnings
import importlib

# Import library components
import library.dataset

importlib.reload(library.dataset)

from library.config import TrainConfig, DataConfig, VoxelConfig, ModelConfig, set_seeds
from library.utils import create_voxel_grid, BoxUtils
from library.model import CenterPointPillars
from library.loss import CenterPointLoss
from library.dataset import LidarDataset, collate_fn
from library.trainer import Trainer
from library.inference import InferenceRunner

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demonstration():
    print("=== Starting 3D Object Detection Library Demonstration ===\n")

    # 1. Setup and Configuration Adjustments for Demo
    # We modify the TrainConfig to ensure the demo runs quickly.
    print("[1] Configuring environment for rapid demonstration...")
    TrainConfig.epochs = 1
    TrainConfig.debug_subset_size = 4  # Use only 4 samples for training/inference
    TrainConfig.batch_size = 2
    TrainConfig.log_interval = 1

    # Ensure working directories exist
    os.makedirs(DataConfig.work_dir, exist_ok=True)
    os.makedirs(DataConfig.cache_dir, exist_ok=True)

    # Set seeds for reproducibility
    set_seeds(TrainConfig.seed)
    device = torch.device(TrainConfig.device)
    print(f"    Device: {device}")
    print(f"    Debug Subset Size: {TrainConfig.debug_subset_size}")
    print("    Configuration complete.\n")

    # 2. Verify Voxelization Logic
    print("[2] Verifying Voxelization (PointPillars Preprocessing)...")
    voxel_config = VoxelConfig()

    # Generate synthetic point cloud: (N, 4) -> [x, y, z, intensity]
    # Create points within the valid range
    num_points = 500
    points = np.random.uniform(-50, 50, (num_points, 4)).astype(np.float32)
    points[:, 2] = np.random.uniform(-2, 2, (num_points,))  # Z range

    pillar_features, pillar_coords, pillar_num_points = create_voxel_grid(
        points, voxel_config
    )

    # Assertions
    assert (
        pillar_features.ndim == 3
    ), "Pillar features should be 3D (M, max_points, features)"
    assert (
        pillar_features.shape[2] == voxel_config.num_point_features
    ), f"Feature dim should be {voxel_config.num_point_features}"
    assert (
        pillar_coords.ndim == 2 and pillar_coords.shape[1] == 3
    ), "Coords should be (M, 3)"

    print(f"    Generated {pillar_features.shape[0]} pillars from {num_points} points.")
    print("    Voxelization logic verified.\n")

    # 3. Verify Model Architecture
    print("[3] Verifying Model Architecture (Forward Pass)...")
    model = CenterPointPillars().to(device)
    model.eval()

    # Create a dummy batch based on the voxelization output
    # Coords need batch index prepended: (batch_idx, z, y, x)
    batch_size = 2

    # Replicate data for batch size of 2
    b_pillar_features = (
        torch.from_numpy(pillar_features).repeat(batch_size, 1, 1).to(device)
    )

    # Add batch indices (0 and 1)
    coords_0 = torch.from_numpy(pillar_coords)
    coords_0 = torch.cat(
        [torch.zeros((coords_0.shape[0], 1), dtype=torch.int32), coords_0], dim=1
    )

    coords_1 = torch.from_numpy(pillar_coords)
    coords_1 = torch.cat(
        [torch.ones((coords_1.shape[0], 1), dtype=torch.int32), coords_1], dim=1
    )

    b_pillar_coords = torch.cat([coords_0, coords_1], dim=0).to(device)
    b_pillar_features = torch.cat(
        [torch.from_numpy(pillar_features).to(device)] * 2, dim=0
    )

    batched_inputs = {
        "pillar_features": b_pillar_features,
        "pillar_coords": b_pillar_coords,
        "batch_size": batch_size,
    }

    with torch.no_grad():
        preds = model(batched_inputs)

    # Verify Output Shapes
    # Grid size: [512, 512] for standard config
    grid_w, grid_h = voxel_config.grid_size[0], voxel_config.grid_size[1]

    print("    Checking output head shapes:")
    for head_name, tensor in preds.items():
        print(f"      - {head_name}: {tensor.shape}")
        assert tensor.shape[0] == batch_size, f"Batch size mismatch in {head_name}"
        assert (
            tensor.shape[2] == grid_h and tensor.shape[3] == grid_w
        ), f"Spatial dim mismatch in {head_name}"

    print("    Model architecture verified.\n")

    # 4. Verify Loss Function
    print("[4] Verifying Loss Function...")
    criterion = CenterPointLoss().to(device)

    # Create dummy targets matching prediction shapes
    targets = {}
    for head, channels in ModelConfig.heads.items():
        targets[head] = torch.zeros((batch_size, channels, grid_h, grid_w)).to(device)

    # Add mask for regression
    targets["mask_reg"] = torch.zeros((batch_size, 1, grid_h, grid_w)).to(device)
    # Set a few pixels to 1 to simulate objects
    targets["mask_reg"][:, :, 100, 100] = 1.0
    targets["hm"][:, 0, 100, 100] = 1.0  # Object of class 0 at (100, 100)

    loss_dict = criterion(preds, targets)

    print(f"    Total Loss: {loss_dict['loss'].item():.4f}")
    assert not torch.isnan(loss_dict["loss"]), "Loss is NaN"
    print("    Loss function verified.\n")

    # 5. Verify Dataset Loading
    print("[5] Verifying Dataset Loading...")
    # Initialize dataset with debug subset
    dataset = LidarDataset(
        metadata_path=DataConfig.train_metadata_path,
        split="train",
        enable_augmentation=False,
        subset_size=2,
    )

    sample = dataset[0]
    required_keys = ["pillar_features", "pillar_coords", "targets", "token", "matrix"]
    for k in required_keys:
        assert k in sample, f"Missing key {k} in dataset sample"

    print(f"    Loaded sample token: {sample['token']}")
    print(f"    Number of pillars: {sample['pillar_features'].shape[0]}")
    print("    Dataset verified.\n")

    # 6. Run Training Loop
    print("[6] Running Training Loop (Demo)...")
    # Initialize Trainer with debug subset
    trainer = Trainer(debug_subset_size=TrainConfig.debug_subset_size)

    try:
        trainer.train()
        print("    Training execution successful.")
    except Exception as e:
        print(f"    Training failed: {e}")
        raise e

    # Verify checkpoints exist
    assert os.path.exists(
        TrainConfig.latest_model_path
    ), "Latest model checkpoint not found"
    print("    Checkpoints verified.\n")

    # 7. Run Inference
    print("[7] Running Inference (Demo)...")
    inference_runner = InferenceRunner()

    try:
        inference_runner.predict_and_format(subset_size=TrainConfig.debug_subset_size)
        print("    Inference execution successful.")
    except Exception as e:
        print(f"    Inference failed: {e}")
        raise e

    # Verify submission file
    assert os.path.exists(TrainConfig.submission_path), "Submission file not found"
    print(f"    Submission saved to: {TrainConfig.submission_path}")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demonstration()
