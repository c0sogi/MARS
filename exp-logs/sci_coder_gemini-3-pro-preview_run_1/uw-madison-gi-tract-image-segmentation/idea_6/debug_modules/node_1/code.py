import os
import shutil
import numpy as np
import torch
import pandas as pd
import random

# Import library components
from library.config import Config
from library.utils import rle_encode, rle_decode, compute_metrics, robust_normalize
from library.postprocessing import keep_largest_component_3d
from library.data import create_dataloaders
from library.model import ResNet3DUNet
from library.losses import BCEDiceLoss
from library.trainer import Trainer


def set_seed(seed=42):
    """Sets fixed random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def test_utilities():
    print("\n=== Testing Utilities ===")

    # 1. Test RLE Encoding/Decoding
    shape = (100, 100)
    mask = np.zeros(shape, dtype=np.uint8)
    # Create a square
    mask[20:40, 20:40] = 1

    encoded = rle_encode(mask)
    decoded = rle_decode(encoded, shape)

    assert isinstance(encoded, str), "RLE encode should return a string"
    assert np.array_equal(mask, decoded), "RLE decoded mask does not match original"
    print("RLE Encode/Decode: PASSED")

    # 2. Test Robust Normalization
    img = np.random.rand(50, 50).astype(np.float32) * 1000
    # Add outliers
    img[0, 0] = 10000
    img[0, 1] = -100
    norm_img = robust_normalize(img, lower=1.0, upper=99.0)

    assert (
        norm_img.min() >= 0.0 and norm_img.max() <= 1.0
    ), "Normalized image out of [0, 1] range"
    assert norm_img.shape == img.shape, "Normalized image shape mismatch"
    print("Robust Normalization: PASSED")

    # 3. Test Metrics
    pred = np.zeros((10, 10), dtype=np.uint8)
    gt = np.zeros((10, 10), dtype=np.uint8)
    pred[2:5, 2:5] = 1
    gt[2:5, 2:5] = 1

    metrics = compute_metrics(pred, gt)
    assert np.isclose(metrics["dice"], 1.0), "Perfect overlap should have Dice 1.0"
    assert np.isclose(
        metrics["hausdorff"], 0.0
    ), "Perfect overlap should have Hausdorff 0.0"

    # Test mismatch
    gt_offset = np.zeros((10, 10), dtype=np.uint8)
    gt_offset[3:6, 3:6] = 1
    metrics_mismatch = compute_metrics(pred, gt_offset)
    assert metrics_mismatch["dice"] < 1.0, "Imperfect overlap should have Dice < 1.0"
    print("Metrics Computation: PASSED")


def test_postprocessing():
    print("\n=== Testing Post-processing ===")

    # Create a 3D volume with two components
    vol = np.zeros((10, 50, 50), dtype=np.uint8)

    # Large component
    vol[2:8, 10:40, 10:40] = 1  # 6 * 30 * 30 = 5400 voxels

    # Small component (noise)
    vol[0:2, 1:5, 1:5] = 1  # 2 * 4 * 4 = 32 voxels

    # Process
    # Set min_size larger than noise but smaller than main object
    clean_vol = keep_largest_component_3d(vol, min_size=100)

    # Check that small component is gone
    assert clean_vol[0:2, 1:5, 1:5].sum() == 0, "Noise component was not removed"
    # Check that large component remains
    assert clean_vol[2:8, 10:40, 10:40].sum() == 5400, "Large component was damaged"

    print("Keep Largest Component 3D: PASSED")


def test_model_and_loss():
    print("\n=== Testing Model and Loss ===")

    device = Config.DEVICE

    # Instantiate Model
    model = ResNet3DUNet(in_channels=1, out_channels=3).to(device)

    # Create dummy input: (Batch, Channel, Depth, Height, Width)
    # Using small spatial size for speed in this unit test
    dummy_input = torch.randn(2, 1, 16, 64, 64).to(device)

    # Forward pass
    output = model(dummy_input)

    # Check Output Shape: (B, Num_Classes, D, H, W)
    expected_shape = (2, 3, 16, 64, 64)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Got {output.shape}, expected {expected_shape}"

    # Instantiate Loss
    criterion = BCEDiceLoss()

    # Create dummy target
    dummy_target = torch.randint(0, 2, expected_shape).float().to(device)

    # Calculate Loss
    loss = criterion(output, dummy_target)

    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"

    print("Model Forward Pass & Loss: PASSED")


def run_full_pipeline():
    print("\n=== Running Full Pipeline Demonstration ===")

    # 1. Setup Configuration for Demo
    # We override Config to run fast
    Config.DEBUG = True  # Limits dataset to 2 cases
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2

    # Override paths to a demo directory
    Config.WORKING_DIR = "./working/demo_run"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.PREDICTION_DIR = os.path.join(Config.WORKING_DIR, "predictions")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")

    # Clean up previous demo run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    Config.setup()

    # 2. Data Loading
    print("Creating DataLoaders...")
    # load_cached_data=False forces processing from scratch to test that logic
    train_loader, val_loader, test_loader = create_dataloaders(load_cached_data=False)

    # Verify Train Loader Batch
    batch = next(iter(train_loader))
    imgs = batch["image"]
    masks = batch["mask"]

    print(f"Train Batch Image Shape: {imgs.shape}")
    print(f"Train Batch Mask Shape: {masks.shape}")

    assert imgs.ndim == 5, "Image batch should be 5D (B, C, D, H, W)"
    assert masks.ndim == 5, "Mask batch should be 5D (B, C, D, H, W)"
    assert imgs.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"

    # 3. Trainer Execution
    print("Initializing Trainer...")
    trainer = Trainer(train_loader, val_loader, test_loader)

    print("Starting Training Loop (1 Epoch)...")
    trainer.fit()

    # Verify Checkpoint
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Best model checkpoint was not created"
    print("Checkpoint verified.")

    # 4. Inference & Submission
    print("Running Inference and Generating Submission...")
    trainer.predict_and_submit()

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    # Verify Submission Content
    df_sub = pd.read_csv(submission_path)
    print(f"Submission file generated with {len(df_sub)} rows.")
    assert (
        "id" in df_sub.columns
        and "class" in df_sub.columns
        and "predicted" in df_sub.columns
    )

    print("Full Pipeline: PASSED")


if __name__ == "__main__":
    try:
        set_seed(42)

        # Run Unit Tests
        test_utilities()
        test_postprocessing()
        test_model_and_loss()

        # Run Integration Test
        run_full_pipeline()

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nFAILED: {e}")
        raise e
