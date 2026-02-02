import os
import shutil
import torch
import numpy as np
import pandas as pd
import warnings
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.dataset import NuScenesDataset
from library.modules import IoUAwareCenterPoint
from library.loss import IoUAwareLoss
from library.runner import Trainer
from library.utils import iou3d_global, box3d_to_corners
from library.inference import predict_and_format

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def setup_demo_config():
    """
    Overrides default configuration for a fast demonstration run.
    """
    print(">>> Setting up Demo Configuration...")

    # Set paths to a temporary working directory
    Config.WORK_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache")
    Config.LOG_DIR = os.path.join(Config.WORK_DIR, "logs")
    Config.CKPT_DIR = os.path.join(Config.WORK_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORK_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Create directories
    for d in [
        Config.WORK_DIR,
        Config.CACHE_DIR,
        Config.LOG_DIR,
        Config.CKPT_DIR,
        Config.SUBMISSION_DIR,
    ]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # Reduce compute requirements for demo
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.MAX_EPOCHS = 1
    Config.TOP_K = 5  # Reduce top K for easier inspection

    # Set seed for reproducibility
    Config.set_seed()
    print("Configuration updated.")


def verify_utils():
    """
    Verifies geometric utility functions.
    """
    print("\n>>> Verifying Utilities...")

    # 1. Test box3d_to_corners
    # Box format: x, y, z, w, l, h, yaw
    box = torch.tensor([[0, 0, 0, 2, 4, 2, 0]], dtype=torch.float32)
    corners = box3d_to_corners(box)

    # Expected corners for a 2x4x2 box at origin with 0 yaw
    # x ranges [-1, 1], y ranges [-2, 2], z ranges [-1, 1]
    assert corners.shape == (1, 8, 3), f"Expected shape (1, 8, 3), got {corners.shape}"

    # 2. Test IoU Calculation
    # Case A: Identical boxes -> IoU should be 1.0
    box_a = torch.tensor([[10, 10, 0, 2, 4, 2, 0]], dtype=torch.float32)
    box_b = torch.tensor([[10, 10, 0, 2, 4, 2, 0]], dtype=torch.float32)
    iou_same = iou3d_global(box_a, box_b)
    assert torch.isclose(
        iou_same[0, 0], torch.tensor(1.0)
    ), f"Identical boxes IoU should be 1.0, got {iou_same[0,0]}"

    # Case B: Disjoint boxes -> IoU should be 0.0
    box_c = torch.tensor([[20, 20, 0, 2, 4, 2, 0]], dtype=torch.float32)
    iou_diff = iou3d_global(box_a, box_c)
    assert torch.isclose(
        iou_diff[0, 0], torch.tensor(0.0)
    ), f"Disjoint boxes IoU should be 0.0, got {iou_diff[0,0]}"

    print("Utilities verified successfully.")


def verify_dataset():
    """
    Verifies dataset loading, augmentation, and target generation.
    """
    print("\n>>> Verifying Dataset...")

    # Load a tiny subset of training data
    ds = NuScenesDataset(
        "train", enable_augmentation=True, has_targets=True, max_samples=4
    )
    print(f"Dataset loaded with {len(ds)} samples.")

    # Fetch one sample
    sample = ds[0]

    # Check Points
    points = sample["points"]
    assert (
        points.dim() == 2 and points.shape[1] == 4
    ), f"Points shape mismatch: {points.shape}"

    # Check Targets
    targets = sample["targets"]
    assert "hm" in targets, "Missing heatmap in targets"
    assert "reg" in targets, "Missing regression in targets"

    # Check Heatmap Shape: (Num_Classes, H, W)
    # Grid size calculation: Range 102.4m / Voxel 0.1m / Downsample 4 = 256
    expected_dim = int(
        (Config.POINT_CLOUD_RANGE[3] - Config.POINT_CLOUD_RANGE[0])
        / Config.VOXEL_SIZE[0]
        / Config.DOWN_RATIO
    )
    hm_shape = targets["hm"].shape
    assert hm_shape == (
        Config.NUM_CLASSES,
        expected_dim,
        expected_dim,
    ), f"Heatmap shape mismatch. Expected {(Config.NUM_CLASSES, expected_dim, expected_dim)}, got {hm_shape}"

    # Check GT Boxes
    gt_boxes = sample["gt_boxes"]
    if len(gt_boxes) > 0:
        assert (
            gt_boxes.shape[1] == 8
        ), "GT Boxes should have 8 columns (x,y,z,w,l,h,yaw,cls)"

    print("Dataset verified successfully.")
    return ds


def verify_model_and_loss(dataset):
    """
    Verifies model forward pass and loss calculation.
    """
    print("\n>>> Verifying Model and Loss...")

    loader = DataLoader(
        dataset, batch_size=Config.BATCH_SIZE, collate_fn=NuScenesDataset.collate_fn
    )
    batch = next(iter(loader))

    # Move to device
    device = Config.DEVICE
    points = [p.to(device) for p in batch["points"]]
    targets = {k: v.to(device) for k, v in batch["targets"].items()}

    # Initialize Model
    model = IoUAwareCenterPoint().to(device)

    # Forward Pass
    preds = model({"points": points})

    # Check Output Shapes
    # Heatmap
    assert "hm" in preds
    assert preds["hm"].shape[1] == Config.NUM_CLASSES

    # Regression Heads
    assert "reg" in preds
    assert preds["reg"].shape[1] == 2

    # IoU Head
    assert "iou" in preds
    assert preds["iou"].shape[1] == 1

    print("Model forward pass successful.")

    # Loss Calculation
    loss_fn = IoUAwareLoss()
    loss, loss_dict = loss_fn(preds, targets)

    print(f"Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss > 0, "Loss should be positive"

    print("Loss calculation verified.")
    return model


def run_training_demo(dataset):
    """
    Demonstrates the Trainer class.
    """
    print("\n>>> Running Training Demo...")

    # Split dataset for demo (train on same for simplicity)
    train_loader = DataLoader(
        dataset, batch_size=Config.BATCH_SIZE, collate_fn=NuScenesDataset.collate_fn
    )
    val_loader = DataLoader(
        dataset, batch_size=Config.BATCH_SIZE, collate_fn=NuScenesDataset.collate_fn
    )

    trainer = Trainer()

    # Overwrite patience to avoid early stopping logic interference in demo
    trainer.patience = 10

    # Run fit
    trainer.fit(train_loader, val_loader, epochs=1)

    # Check if checkpoint exists
    ckpt_path = os.path.join(Config.CKPT_DIR, "best_model.pth")
    assert os.path.exists(ckpt_path), "Checkpoint file was not created."

    print("Training demo completed.")
    return trainer.model


def run_inference_demo(trained_model):
    """
    Demonstrates inference and submission generation.
    """
    print("\n>>> Running Inference Demo...")

    # Load small test set
    test_ds = NuScenesDataset(
        "test", enable_augmentation=False, has_targets=False, max_samples=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, collate_fn=NuScenesDataset.collate_fn
    )

    # Run prediction
    predict_and_format(
        trained_model, test_loader, Config.SUBMISSION_PATH, device=Config.DEVICE
    )

    # Verify Output
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "Id" in df.columns and "PredictionString" in df.columns
    ), "Submission CSV missing required columns."
    assert len(df) == 4, f"Expected 4 predictions, got {len(df)}"

    print("Inference demo completed.")
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")
    print("Sample Output:")
    print(df.head())


if __name__ == "__main__":
    # 1. Setup
    setup_demo_config()

    # 2. Verify Utilities
    verify_utils()

    # 3. Verify Dataset
    ds = verify_dataset()

    # 4. Verify Model & Loss
    model = verify_model_and_loss(ds)

    # 5. Run Training Loop
    # We use the Trainer class which re-initializes the model,
    # so we pass the dataset to it.
    trained_model = run_training_demo(ds)

    # 6. Run Inference
    run_inference_demo(trained_model)

    print("\n>>> All demonstrations completed successfully.")
