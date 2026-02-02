import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import DataLoader

# Import from provided library
from library.config import Config, seed_everything
from library.utils import (
    gaussian_radius,
    draw_gaussian,
    get_affine_transform,
    affine_transform,
    preprocess_metadata,
)
from library.dataset import KuzushijiDataset
from library.model import DKN
from library.loss import DKNLoss
from library.engine import Engine


def test_utils():
    """
    Verifies mathematical and geometric utility functions.
    """
    print("\n=== Testing Utilities ===")

    # 1. Test Gaussian Radius Calculation
    # For a 100x100 box, radius should be positive
    radius = gaussian_radius((100, 100), min_overlap=0.7)
    assert radius > 0, f"Gaussian radius must be positive, got {radius}"
    print(f"Gaussian radius for 100x100: {radius:.4f}")

    # 2. Test Gaussian Drawing
    heatmap = np.zeros((100, 100), dtype=np.float32)
    center = (50, 50)
    draw_gaussian(heatmap, center, radius=10)

    # Peak should be at center
    assert heatmap[50, 50] == 1.0, "Gaussian peak should be 1.0"
    # Edges should be near zero
    assert heatmap[0, 0] == 0.0, "Gaussian tail should be 0.0"
    print("Gaussian drawing verified.")

    # 3. Test Affine Transform
    # Center (50, 50), Scale 100, Output 100x100 -> Identity-like transform
    trans = get_affine_transform(
        center=np.array([50, 50]), scale=100, rot=0, output_size=[100, 100]
    )
    assert trans.shape == (2, 3), "Affine transform matrix must be 2x3"

    # Test point transformation
    pt = np.array([50, 50])
    new_pt = affine_transform(pt, trans)
    # Center should map to center of output (50, 50)
    assert np.allclose(
        new_pt, [50, 50], atol=1.0
    ), f"Affine transform failed. Got {new_pt}"
    print("Affine transform verified.")


def test_dataset():
    """
    Verifies dataset loading, preprocessing, and tensor shapes.
    """
    print("\n=== Testing Dataset ===")

    # Initialize Dataset (Force reload cache to test parsing logic)
    # Note: We use the 'train' split which expects labels.
    ds = KuzushijiDataset(split="train", load_cached_data=False)

    assert len(ds) > 0, "Dataset is empty."
    print(f"Dataset size: {len(ds)}")

    # Fetch one sample
    sample = ds[0]

    # Check keys
    expected_keys = ["input", "image_id", "hm", "cls_target", "reg_target", "reg_mask"]
    for k in expected_keys:
        assert k in sample, f"Missing key in dataset sample: {k}"

    # Check Input Image Shape (3, H, W) -> Config.IMG_SIZE is 1024
    img = sample["input"]
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect input shape: {img.shape}"

    # Check Target Shapes (Downsampled by 4 -> 256)
    out_size = Config.IMG_SIZE // 4
    assert sample["hm"].shape == (
        1,
        out_size,
        out_size,
    ), f"Incorrect heatmap shape: {sample['hm'].shape}"
    assert sample["reg_target"].shape == (
        4,
        out_size,
        out_size,
    ), f"Incorrect regression target shape: {sample['reg_target'].shape}"

    print("Dataset shapes and keys verified.")
    return sample


def test_model_and_loss(sample_batch):
    """
    Verifies model forward pass and loss computation.
    """
    print("\n=== Testing Model & Loss ===")

    device = torch.device(Config.DEVICE)
    model = DKN(num_classes=Config.NUM_CLASSES).to(device)
    criterion = DKNLoss()

    # Prepare batch (Add batch dimension)
    batch = {}
    for k, v in sample_batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)
        else:
            batch[k] = [v]  # List for strings like image_id

    # Forward Pass
    model.eval()
    with torch.no_grad():
        outputs = model(batch["input"])

    # Check Output Structure
    assert "hm" in outputs
    assert "cls" in outputs
    assert "reg" in outputs

    # Check Output Shapes
    # Heatmap: (B, 1, H/4, W/4)
    assert outputs["hm"].shape == batch["hm"].shape, "Heatmap output shape mismatch"
    # Classification: (B, NumClasses, H/4, W/4)
    assert (
        outputs["cls"].shape[1] == Config.NUM_CLASSES
    ), "Class output channels mismatch"

    print("Model forward pass successful.")

    # Loss Calculation
    # We need gradients for loss usually, but here we just check calculation
    loss, stats = criterion(outputs, batch)

    assert not torch.isnan(loss), "Loss returned NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    print(f"Loss computed successfully: {loss.item():.4f}")
    print(f"Loss components: {stats}")


def test_engine_pipeline():
    """
    Verifies the full training and inference pipeline using the Engine class.
    Uses debug=True to run only a few batches.
    """
    print("\n=== Testing Engine (Train & Predict) ===")

    engine = Engine()

    # 1. Train (Fit)
    # debug=True limits the loop to 10 batches per epoch
    print("Running training loop (1 Epoch, Debug Mode)...")
    engine.fit(epochs=1, debug=True)

    # Check if model was saved
    assert os.path.exists(engine.model_save_path), "Model file was not saved."
    print("Model training simulation complete.")

    # 2. Inference (Predict)
    print("Running inference...")
    engine.predict()

    # Check submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Validate submission format
    df = pd.read_csv(Config.SUBMISSION_PATH)
    assert "image_id" in df.columns
    assert "labels" in df.columns
    assert len(df) > 0, "Submission file is empty."

    print(f"Submission generated with {len(df)} rows.")
    print("Engine pipeline verified.")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # 1. Test Utils
    test_utils()

    # 2. Test Dataset
    sample = test_dataset()

    # 3. Test Model & Loss
    test_model_and_loss(sample)

    # 4. Test Full Engine
    test_engine_pipeline()

    print("\nAll tests passed successfully!")
