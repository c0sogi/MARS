import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import box_encode, box_decode, iou_bev, iou3d, points_in_boxes_gpu
from library.dataset import LidarDataset, collate_fn
from library.model_blocks import TwoStagePointPillars
from library.detector import PointPillarsDetector
from library.trainer import Trainer


def set_seeds(seed=42):
    """Sets fixed random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_for_demo():
    """
    Overrides default configuration for a fast demonstration run.
    """
    print("Configuring settings for fast demonstration...")
    # Use a separate working directory for the demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.GT_DATABASE_DIR = os.path.join(Config.WORKING_DIR, "gt_database")
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.GT_DATABASE_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Speed optimizations
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.SUBSET_SIZE = 4  # Very small subset for speed
    Config.DEBUG = True

    # Disable GT Augmentation to skip time-consuming database generation
    Config.USE_GT_AUGMENTATION = False

    # Reduce voxel grid complexity for speed if needed (optional, keeping defaults for correctness)
    # Config.MAX_VOXELS_TRAIN = 1000

    Config.print_config()


def test_utils():
    """
    Validates utility functions: Box Encoding/Decoding and IoU.
    """
    print("\n--- Testing Utility Functions ---")

    # 1. Test Box Encoding/Decoding
    # Format: [x, y, z, w, l, h, yaw]
    anchors = torch.tensor(
        [[10.0, 10.0, -1.0, 2.0, 4.0, 1.5, 0.0]], dtype=torch.float32
    )
    gt_boxes = torch.tensor(
        [[10.5, 10.5, -1.0, 2.0, 4.0, 1.5, 0.1]], dtype=torch.float32
    )

    # Encode
    encoded = box_encode(gt_boxes, anchors)
    # Decode
    decoded = box_decode(encoded, anchors)

    # Assert reconstruction is accurate
    diff = torch.abs(gt_boxes - decoded).max()
    print(f"Box Encode/Decode Max Diff: {diff.item():.6f}")
    assert diff < 1e-5, "Box encoding/decoding failed to reconstruct original boxes."

    # 2. Test IoU BEV
    # Create two overlapping boxes
    box_a = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]], dtype=torch.float32)
    box_b = torch.tensor([[1.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]], dtype=torch.float32)

    # Area A = 4, Area B = 4. Intersection is 1x2=2. Union = 4+4-2=6. IoU = 2/6 = 0.333...
    iou_val = iou_bev(box_a, box_b)
    print(f"Calculated BEV IoU: {iou_val.item():.4f}")
    assert torch.abs(iou_val - (1.0 / 3.0)) < 1e-4, "BEV IoU calculation incorrect."

    # 3. Test 3D IoU
    # Same boxes, perfectly aligned in Z and Height. 3D IoU should equal BEV IoU.
    iou3d_val = iou3d(box_a, box_b)
    print(f"Calculated 3D IoU: {iou3d_val.item():.4f}")
    assert torch.abs(iou3d_val - (1.0 / 3.0)) < 1e-4, "3D IoU calculation incorrect."

    print("Utils verification passed.")


def test_dataset_and_loader():
    """
    Validates Dataset loading and DataLoader collation.
    """
    print("\n--- Testing Dataset and DataLoader ---")

    # Initialize Dataset
    ds = LidarDataset(
        split="train", subset_size=Config.SUBSET_SIZE, load_cached_data=False
    )
    print(f"Dataset length: {len(ds)}")
    assert len(ds) == Config.SUBSET_SIZE, "Dataset subset size mismatch."

    # Fetch one sample
    sample = ds[0]
    print("Sample keys:", sample.keys())

    # Verify Voxel Shapes
    # voxels: (M, 32, 4)
    assert (
        sample["voxels"].ndim == 3
        and sample["voxels"].shape[1] == 32
        and sample["voxels"].shape[2] == 4
    ), f"Unexpected voxel shape: {sample['voxels'].shape}"

    # Verify Coordinates
    # coordinates: (M, 3) -> (z, y, x)
    assert (
        sample["coordinates"].ndim == 2 and sample["coordinates"].shape[1] == 3
    ), f"Unexpected coordinates shape: {sample['coordinates'].shape}"

    # Test DataLoader Collation
    loader = torch.utils.data.DataLoader(
        ds, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn, shuffle=False
    )

    batch = next(iter(loader))
    print(f"Batch Size: {len(batch['sample_tokens'])}")

    # Check collated coordinates: (M_total, 4) -> (batch_idx, z, y, x)
    assert (
        batch["coordinates"].shape[1] == 4
    ), "Collated coordinates should have 4 columns (batch_idx + 3 coords)."

    print("Dataset and DataLoader verification passed.")
    return batch


def test_model_forward(batch):
    """
    Validates the model architecture forward pass.
    """
    print("\n--- Testing Model Forward Pass ---")

    device = torch.device(Config.DEVICE)
    model = TwoStagePointPillars().to(device)
    model.eval()

    # Prepare inputs
    voxels = batch["voxels"].to(device)
    num_points = batch["num_points"].to(device)
    coordinates = batch["coordinates"].to(device)
    batch_size = len(batch["sample_tokens"])

    # Forward Stage 1
    with torch.no_grad():
        heatmap, regression, backbone_feats = model(
            voxels, num_points, coordinates, batch_size=batch_size
        )

    print(f"Heatmap Shape: {heatmap.shape}")
    print(f"Regression Shape: {regression.shape}")
    print(f"Backbone Feats Shape: {backbone_feats.shape}")

    # Verify Shapes
    # Heatmap: (B, Num_Classes, H, W)
    # Grid size is defined in Config.GRID_SIZE. Backbone stride is 1 (based on model analysis).
    expected_h = (
        Config.GRID_SIZE[1] // Config.FEATURE_MAP_STRIDE
    )  # Usually stride is effectively 1 or 2 depending on blocks
    # Actually, looking at Backbone:
    # Input to backbone is (B, C, H, W) from scatter.
    # Block1 (stride 1) -> Block2 (stride 2) -> Block3 (stride 2).
    # Deconv1 (stride 1) from Block1.
    # Deconv2 (stride 2) from Block2.
    # Deconv3 (stride 4) from Block3.
    # All upsample back to Block1 size.
    # So output size should match Grid Size (800x800).

    assert (
        heatmap.shape[2] == Config.GRID_SIZE[1]
        and heatmap.shape[3] == Config.GRID_SIZE[0]
    ), f"Heatmap spatial dim mismatch. Expected {Config.GRID_SIZE[1]}x{Config.GRID_SIZE[0]}, got {heatmap.shape[2]}x{heatmap.shape[3]}"

    assert regression.shape[1] == 8, "Regression output should have 8 channels."

    # Test Stage 2 (RoI Head)
    # Generate dummy proposals
    # Proposals: (B, K, 7)
    K = 5
    dummy_proposals = torch.zeros((batch_size, K, 7), device=device)
    # Fill with some valid ranges
    dummy_proposals[:, :, 0] = 10.0  # x
    dummy_proposals[:, :, 1] = 0.0  # y
    dummy_proposals[:, :, 2] = -1.0  # z
    dummy_proposals[:, :, 3] = 2.0  # w
    dummy_proposals[:, :, 4] = 4.0  # l
    dummy_proposals[:, :, 5] = 1.5  # h

    with torch.no_grad():
        residuals, iou_pred = model.forward_stage2(backbone_feats, dummy_proposals)

    print(f"Stage 2 Residuals Shape: {residuals.shape}")
    print(f"Stage 2 IoU Pred Shape: {iou_pred.shape}")

    assert residuals.shape == (batch_size, K, 7), "Stage 2 residuals shape mismatch."
    assert iou_pred.shape == (batch_size, K), "Stage 2 IoU prediction shape mismatch."

    print("Model forward pass verification passed.")


def run_full_pipeline():
    """
    Runs the Trainer to demonstrate the full training and inference loop.
    """
    print("\n--- Running Full Trainer Pipeline ---")

    # Initialize Trainer with debug settings
    trainer = Trainer(debug=True, subset_size=Config.SUBSET_SIZE, epochs=Config.EPOCHS)

    # Run pipeline
    trainer.run()

    # Verify submission file generation
    if os.path.exists(Config.SUBMISSION_PATH):
        df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission generated with {len(df)} rows.")
        print(df.head())
        assert len(df) > 0, "Submission file is empty."
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("Full pipeline execution successful.")


if __name__ == "__main__":
    # 1. Setup
    set_seeds(42)
    configure_for_demo()

    # 2. Verify Utils
    test_utils()

    # 3. Verify Data
    batch = test_dataset_and_loader()

    # 4. Verify Model
    test_model_forward(batch)

    # 5. Run Trainer
    run_full_pipeline()

    print("\nAll demonstrations completed successfully.")
