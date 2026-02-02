import os
import shutil
import pandas as pd
import torch
import numpy as np
import warnings

# Import library components
from library.config import Config
from library.dataset import LyftDataset, collate_fn
from library.model import PointPillars
from library.loss import PointPillarsLoss, AnchorGenerator
from library.engine import Trainer
from library.submission import SubmissionGenerator
from library.utils import setup_logger


def run_demo():
    # ==============================================================================
    # 1. SETUP & CONFIGURATION OVERRIDES
    # ==============================================================================
    print(">>> Setting up demonstration environment...")

    # Define a separate working directory for this demo to avoid conflicts
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths and parameters for speed
    Config.WORKING_DIR = DEMO_DIR
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "model_checkpoint.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Create Metadata Subsets (Top 10 samples for train, 4 for val/test)
    # This ensures the code runs in seconds rather than hours
    train_full = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_full = pd.read_csv(Config.VAL_METADATA_PATH)
    test_full = pd.read_csv(Config.TEST_METADATA_PATH)

    train_sub_path = os.path.join(DEMO_DIR, "train_subset.csv")
    val_sub_path = os.path.join(DEMO_DIR, "val_subset.csv")
    test_sub_path = os.path.join(DEMO_DIR, "test_subset.csv")

    train_full.head(10).to_csv(train_sub_path, index=False)
    val_full.head(4).to_csv(val_sub_path, index=False)
    test_full.head(4).to_csv(test_sub_path, index=False)

    # Apply overrides to Config class
    Config.TRAIN_METADATA_PATH = train_sub_path
    Config.VAL_METADATA_PATH = val_sub_path
    Config.TEST_METADATA_PATH = test_sub_path
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Use main process to avoid multiprocessing overhead in demo
    Config.MAX_PILLARS = 1000  # Reduce max pillars to save memory/time for demo

    # Set random seeds for reproducibility
    Config.set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # ==============================================================================
    # 2. DATASET & VOXELIZATION VERIFICATION
    # ==============================================================================
    print("\n>>> Verifying Dataset and Voxelization...")

    # Initialize dataset (this will also generate the GT Database for the subset)
    # We force load_cached_data=False initially to ensure it generates for our subset
    ds = LyftDataset(Config.TRAIN_METADATA_PATH, mode="train", load_cached_data=False)

    # Fetch a single sample
    sample = ds[0]

    # Verify Voxelizer output shapes
    # pillars: (MaxP, MaxPts, 9)
    assert sample["pillars"].shape[1] == Config.MAX_POINTS_PER_PILLAR
    assert sample["pillars"].shape[2] == 9
    # coords: (MaxP, 2)
    assert sample["pillar_coords"].shape[1] == 2

    print("Dataset verification passed. Sample shapes correct.")

    # ==============================================================================
    # 3. MODEL & FORWARD PASS VERIFICATION
    # ==============================================================================
    print("\n>>> Verifying Model Forward Pass...")

    model = PointPillars().to(device)
    model.eval()

    # Create a batch
    batch_data = collate_fn([ds[0], ds[1]])

    pillars = batch_data["pillars"].to(device)
    coords = batch_data["pillar_coords"].to(device)
    num_points = batch_data["num_points"].to(device)

    with torch.no_grad():
        cls_preds, box_preds, dir_preds = model(pillars, coords, num_points)

    # Verify Output Shapes
    # Batch size
    B = 2
    # Grid size (H/2, W/2) -> 256x256
    H, W = Config.GRID_SIZE[1] // 2, Config.GRID_SIZE[0] // 2
    # Anchors per location
    num_anchors = model.anchors_per_loc

    # Expected output: (B, H*W*NumAnchors, Channels)
    expected_spatial = H * W * num_anchors

    assert cls_preds.shape == (B, expected_spatial, Config.NUM_CLASSES)
    assert box_preds.shape == (B, expected_spatial, 7)
    assert dir_preds.shape == (B, expected_spatial, 2)

    print(f"Model verification passed. Output shape: {cls_preds.shape}")

    # ==============================================================================
    # 4. LOSS CALCULATION VERIFICATION
    # ==============================================================================
    print("\n>>> Verifying Loss Calculation...")

    anchor_generator = AnchorGenerator()
    criterion = PointPillarsLoss(anchor_generator).to(device)

    loss_dict = criterion(
        cls_preds,
        box_preds,
        dir_preds,
        batch_data["gt_boxes"],
        batch_data["gt_classes"],
    )

    # Verify loss components
    assert "cls_loss" in loss_dict
    assert "loc_loss" in loss_dict
    assert "dir_loss" in loss_dict

    total_loss = loss_dict["cls_loss"] + loss_dict["loc_loss"] + loss_dict["dir_loss"]
    assert not torch.isnan(total_loss), "Loss returned NaN"
    assert total_loss.item() >= 0, "Loss must be non-negative"

    print(f"Loss verification passed. Total Loss: {total_loss.item():.4f}")

    # ==============================================================================
    # 5. TRAINING LOOP DEMONSTRATION
    # ==============================================================================
    print("\n>>> Starting Training Loop (1 Epoch)...")

    # Initialize Trainer
    # Note: Trainer internally initializes a new model and optimizer
    trainer = Trainer(
        model_save_path=Config.MODEL_SAVE_PATH,
        load_cached_data=True,  # Can use cache now as it was generated in step 2
        device=device,
    )

    # Run training
    trainer.fit(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    print("Training complete. Checkpoint saved.")

    # ==============================================================================
    # 6. INFERENCE & SUBMISSION DEMONSTRATION
    # ==============================================================================
    print("\n>>> Generating Submission...")

    # Initialize Generator
    sub_gen = SubmissionGenerator(model_path=Config.MODEL_SAVE_PATH, device=device)

    # Run generation
    sub_gen.generate(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(sub_df)} rows.")
    print("First row prediction:", sub_df.iloc[0]["PredictionString"])

    print("\n>>> Demo Completed Successfully!")


if __name__ == "__main__":
    run_demo()
