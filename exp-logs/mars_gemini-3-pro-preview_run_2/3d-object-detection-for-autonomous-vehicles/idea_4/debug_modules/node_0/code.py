import os
import sys
import torch
import numpy as np
import warnings
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import (
    quaternion_to_matrix,
    project_3d_to_2d,
    gaussian_radius,
    draw_gaussian,
    global_to_camera,
    camera_to_global,
)
from library.dataset import Mono3DDataset
from library.model import MonoCenterNet
from library.loss import Mono3DLoss
from library.train import train_model
from library.inference import decode_detections, generate_submission


def main():
    print("=== Starting 3D Object Detection Library Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Setup
    # ---------------------------------------------------------
    print("\n[1] Setting up Configuration...")
    Config.setup()
    Config.set_seed(42)

    # Override defaults for speed and demo purposes
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo
    Config.DEBUG = True  # Enable debug mode to use data subsets
    Config.DATA_SUBSET_RATIO = 0.01  # Use 1% of data for quick execution

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # ---------------------------------------------------------
    # 2. Verify Utilities
    # ---------------------------------------------------------
    print("\n[2] Verifying Geometric Utilities...")

    # Test: Quaternion to Matrix (Identity)
    q_identity = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z
    R_identity = quaternion_to_matrix(q_identity)
    assert np.allclose(
        R_identity, np.eye(3)
    ), "Quaternion to Matrix failed for identity."

    # Test: Project 3D to 2D
    # Simple Pinhole Camera: fx=fy=100, cx=cy=50
    K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=np.float32)
    point_3d = np.array([10.0, 10.0, 20.0])  # x, y, z
    # Expected u = (x * fx / z) + cx = (10 * 100 / 20) + 50 = 50 + 50 = 100
    # Expected v = (y * fy / z) + cy = (10 * 100 / 20) + 50 = 50 + 50 = 100
    pts_2d, depths = project_3d_to_2d(point_3d, K)

    assert np.allclose(
        pts_2d[0], [100.0, 100.0]
    ), f"3D to 2D Projection failed. Got {pts_2d[0]}"
    assert np.isclose(depths[0], 20.0), "Depth calculation failed."
    print("    Utilities verified successfully.")

    # ---------------------------------------------------------
    # 3. Dataset Loading
    # ---------------------------------------------------------
    print("\n[3] Loading Dataset (Train Split)...")
    # We disable loading from cache to demonstrate raw processing,
    # but in practice, caching speeds up subsequent runs.
    dataset = Mono3DDataset(
        split="train",
        load_cached_data=False,
        debug=Config.DEBUG,
        subset_ratio=Config.DATA_SUBSET_RATIO,
    )

    assert len(dataset) > 0, "Dataset is empty."
    print(f"    Loaded {len(dataset)} samples.")

    # Fetch one sample to verify structure
    img_tensor, targets, info = dataset[0]

    print(f"    Image Tensor Shape: {img_tensor.shape}")
    assert img_tensor.shape == (
        3,
        Config.INPUT_HEIGHT,
        Config.INPUT_WIDTH,
    ), "Image tensor shape mismatch."

    # Verify Targets
    expected_hm_shape = (Config.NUM_CLASSES, Config.OUTPUT_HEIGHT, Config.OUTPUT_WIDTH)
    assert (
        targets["hm"].shape == expected_hm_shape
    ), f"Heatmap shape mismatch. Expected {expected_hm_shape}, got {targets['hm'].shape}"

    print("    Dataset sample verified.")

    # ---------------------------------------------------------
    # 4. Model Forward Pass & Loss
    # ---------------------------------------------------------
    print("\n[4] Running Model Forward Pass...")
    device = Config.DEVICE
    model = MonoCenterNet().to(device)
    criterion = Mono3DLoss()

    # Prepare batch (unsqueeze to add batch dimension)
    batch_imgs = img_tensor.unsqueeze(0).to(device)
    batch_targets = {k: v.unsqueeze(0).to(device) for k, v in targets.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(batch_imgs)

    # Verify Output Keys
    required_keys = ["hm", "depth", "dim", "rot", "offset"]
    for k in required_keys:
        assert k in outputs, f"Model output missing key: {k}"

    print("    Model forward pass successful.")

    # Calculate Loss
    loss, stats = criterion(outputs, batch_targets)
    print(f"    Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN."

    # ---------------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------------
    print("\n[5] Executing Training Loop (1 Epoch)...")
    # This function handles the Trainer instantiation and fitting
    train_model(
        debug=Config.DEBUG,
        load_cached_data=True,  # Use cache if available (we just created it in step 3 implicitly via dataset init if we saved it)
        num_epochs=1,
        batch_size=Config.BATCH_SIZE,
    )

    # Verify Checkpoint Creation
    latest_ckpt = os.path.join(Config.CHECKPOINT_DIR, "latest_model.pth")
    assert os.path.exists(latest_ckpt), f"Checkpoint not found at {latest_ckpt}"
    print(f"    Training complete. Checkpoint saved to {latest_ckpt}")

    # ---------------------------------------------------------
    # 6. Inference & Submission
    # ---------------------------------------------------------
    print("\n[6] Running Inference and Generating Submission...")

    # Use the checkpoint we just trained
    generate_submission(
        checkpoint_path=latest_ckpt,
        split="test",
        debug=Config.DEBUG,
        load_cached_data=False,  # Force fresh load for test set
    )

    submission_file = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(
        submission_file
    ), f"Submission file not found at {submission_file}"

    # Verify Submission Format
    df_sub = pd.read_csv(submission_file)
    print(f"    Submission generated with {len(df_sub)} rows.")
    assert (
        "Id" in df_sub.columns and "PredictionString" in df_sub.columns
    ), "Submission file missing required columns."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
