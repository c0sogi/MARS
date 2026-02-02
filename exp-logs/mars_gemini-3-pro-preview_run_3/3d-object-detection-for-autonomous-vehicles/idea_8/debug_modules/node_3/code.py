import os
import sys
import pandas as pd
import torch
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.dataset import LidarDataset
from library.detector import TwoStagePointPillars
from library.trainer import train_one_epoch
from library.inference import generate_submission


def run_demo():
    print("=== Starting 3D Object Detection Demo ===")

    # ------------------------------------------------------------------
    # 1. Configuration & Data Prep (Optimize for Speed)
    # ------------------------------------------------------------------
    print("\n[Step 1] Preparing Subset Data and Overriding Config...")

    # Define working paths
    demo_dir = "./working/demo_run"
    os.makedirs(demo_dir, exist_ok=True)

    cache_dir = os.path.join(demo_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Create subset metadata for Train (8 samples)
    # We use the existing metadata file provided in the environment
    orig_train_meta = pd.read_csv("./metadata/train_metadata.csv")
    subset_train_meta = orig_train_meta.head(8).copy()
    subset_train_path = os.path.join(demo_dir, "train_subset.csv")
    subset_train_meta.to_csv(subset_train_path, index=False)

    # Create subset metadata for Test (4 samples)
    orig_test_meta = pd.read_csv("./metadata/test_metadata.csv")
    subset_test_meta = orig_test_meta.head(4).copy()
    subset_test_path = os.path.join(demo_dir, "test_subset.csv")
    subset_test_meta.to_csv(subset_test_path, index=False)

    # Override Config global attributes
    Config.WORK_DIR = demo_dir
    Config.CACHE_DIR = cache_dir
    Config.TRAIN_METADATA = subset_train_path
    Config.VAL_METADATA = subset_train_path  # Use train subset for val to save time
    Config.TEST_METADATA = subset_test_path
    Config.CHECKPOINT_PATH = os.path.join(demo_dir, "model_checkpoint.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Reduce compute parameters
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.MAX_VOXELS_TRAIN = 2000  # Reduced from 16000
    Config.MAX_VOXELS_TEST = 2000
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    print("Config updated. Metadata subsets created.")

    # ------------------------------------------------------------------
    # 2. Dataset Verification
    # ------------------------------------------------------------------
    print("\n[Step 2] Verifying Dataset and Voxelization...")

    # Initialize dataset (will compute transforms for the 8 subset samples)
    train_dataset = LidarDataset(split="train", load_cached_data=False)
    print(f"Dataset initialized with {len(train_dataset)} samples.")

    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=LidarDataset.collate_fn,
        num_workers=0,
    )

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify Batch Structure
    required_keys = [
        "voxels",
        "coordinates",
        "num_points",
        "gt_boxes",
        "sample_tokens",
        "trans_matrices",
    ]
    for key in required_keys:
        assert key in batch, f"Missing key in batch: {key}"

    # Verify Tensor Shapes
    # voxels: (M, 32, 9)
    voxels = batch["voxels"]
    coords = batch["coordinates"]

    print(f"Voxels shape: {voxels.shape}")
    print(f"Coordinates shape: {coords.shape}")

    assert voxels.dim() == 3 and voxels.shape[1] == 32 and voxels.shape[2] == 9
    assert coords.dim() == 2 and coords.shape[1] == 4  # (batch_idx, z, y, x)

    # Move batch to device
    device = Config.DEVICE
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
        elif isinstance(value, list):
            batch[key] = [
                v.to(device) if isinstance(v, torch.Tensor) else v for v in value
            ]

    # ------------------------------------------------------------------
    # 3. Model Logic Verification
    # ------------------------------------------------------------------
    print("\n[Step 3] Verifying Model Forward Pass...")

    model = TwoStagePointPillars()
    model.to(device)

    # Test Training Forward Pass
    model.train()
    loss_dict = model(batch, mode="train")

    print("Training Loss Output:", loss_dict)
    assert "total_loss" in loss_dict
    assert "loss_hm" in loss_dict
    assert "loss_box" in loss_dict
    assert "loss_refine" in loss_dict
    assert loss_dict["total_loss"].item() > 0

    # Test Inference Forward Pass
    model.eval()
    with torch.no_grad():
        boxes_list, scores_list, labels_list = model(batch, mode="test")

    print(f"Inference output count: {len(boxes_list)}")
    assert len(boxes_list) == Config.BATCH_SIZE

    # Check structure of predictions
    if len(boxes_list[0]) > 0:
        assert boxes_list[0].shape[1] == 7  # (x, y, z, w, l, h, yaw)
        assert scores_list[0].dim() == 1
        assert labels_list[0].dim() == 1

    # ------------------------------------------------------------------
    # 4. Training Loop Simulation
    # ------------------------------------------------------------------
    print("\n[Step 4] Running Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, total_steps=len(train_loader)
    )

    # Run one epoch
    avg_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, 0, 1)
    print(f"Epoch finished with average loss: {avg_loss:.4f}")

    # Save checkpoint for inference step
    torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
    print(f"Model saved to {Config.CHECKPOINT_PATH}")

    # ------------------------------------------------------------------
    # 5. Inference & Submission
    # ------------------------------------------------------------------
    print("\n[Step 5] Generating Submission on Test Subset...")

    # generate_submission internally loads Config.TEST_METADATA (which we overrode)
    # and Config.CHECKPOINT_PATH (which we just saved)
    generate_submission(
        checkpoint_path=Config.CHECKPOINT_PATH,
        output_path=Config.SUBMISSION_PATH,
        batch_size=Config.BATCH_SIZE,
    )

    # Verify submission file content
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file created with {len(sub_df)} rows.")
        assert len(sub_df) == 4  # Matches our subset size
        assert "Id" in sub_df.columns
        assert "PredictionString" in sub_df.columns
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
