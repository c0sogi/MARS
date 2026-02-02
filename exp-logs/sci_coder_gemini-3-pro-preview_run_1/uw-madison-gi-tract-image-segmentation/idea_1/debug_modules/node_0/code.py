import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.utils import set_seed, dice_coefficient, hausdorff_distance
from library.dataset import UWMadisonDataset
from library.model import UNetResNet18, BCEDiceLoss
from library.train import run_training
from library.inference import run_inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def test_metrics():
    print("\n=== Testing Metrics ===")
    # Create dummy masks: 10x10
    # Case 1: Perfect overlap
    y_true = np.zeros((10, 10), dtype=np.uint8)
    y_true[2:5, 2:5] = 1
    y_pred = y_true.copy()

    dice = dice_coefficient(y_true, y_pred)
    hd = hausdorff_distance(y_true, y_pred)

    print(
        f"Perfect Match -> Dice: {dice:.4f} (Expected ~1.0), HD: {hd:.4f} (Expected 0.0)"
    )
    assert np.isclose(dice, 1.0, atol=1e-4), "Dice should be 1.0 for perfect match"
    assert np.isclose(hd, 0.0, atol=1e-4), "Hausdorff should be 0.0 for perfect match"

    # Case 2: No overlap
    y_pred_empty = np.zeros_like(y_true)
    dice_empty = dice_coefficient(y_true, y_pred_empty)
    # HD is 1.0 if one is empty (normalized logic in utils.py)
    hd_empty = hausdorff_distance(y_true, y_pred_empty)

    print(f"No Overlap -> Dice: {dice_empty:.4f} (Expected ~0.0), HD: {hd_empty:.4f}")
    assert np.isclose(dice_empty, 0.0, atol=1e-4), "Dice should be 0.0 for no overlap"


def test_dataset_loading():
    print("\n=== Testing Dataset Loading ===")
    # Use a small fraction to speed up loading
    fraction = 0.01
    img_size = 256

    dataset = UWMadisonDataset(mode="train", fraction=fraction, img_size=img_size)
    print(f"Loaded Train Dataset with fraction={fraction}: {len(dataset)} samples")

    assert len(dataset) > 0, "Dataset should not be empty"

    # Fetch one sample
    img, mask = dataset[0]

    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Mask Shape: {mask.shape}")

    # Assertions
    # Image: (C, H, W) -> ResNet expects 3 channels usually, but our dataset logic
    # might output (1, H, W) or (3, H, W) depending on albumentations/logic.
    # Checking library/dataset.py:
    # if img_aug.ndim == 2: img_aug = img_aug[np.newaxis, ...] -> (1, H, W)
    # else: transpose -> (C, H, W)
    # The input images are grayscale pngs usually read as 2D or 3D.
    # Let's verify the actual output.
    assert img.ndim == 3, "Image tensor should be 3D (C, H, W)"
    assert mask.ndim == 3, "Mask tensor should be 3D (C, H, W)"
    assert mask.shape[0] == 3, "Mask should have 3 channels (one per class)"
    assert (
        img.shape[1] == img_size and img.shape[2] == img_size
    ), f"Image should be resized to {img_size}x{img_size}"

    # Check normalization
    assert img.min() >= 0.0 and img.max() <= 1.0, "Image should be normalized to [0, 1]"


def test_model_architecture():
    print("\n=== Testing Model Architecture ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetResNet18(num_classes=3).to(device)

    # Create dummy input batch: (Batch=2, Channel=1, H=256, W=256)
    # Note: The modified ResNet in library/model.py expects 1 input channel.
    dummy_input = torch.randn(2, 1, 256, 256).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Input Shape: {dummy_input.shape}")
    print(f"Model Output Shape: {output.shape}")

    assert output.shape == (2, 3, 256, 256), "Output shape mismatch"
    assert (
        output.min() >= 0.0 and output.max() <= 1.0
    ), "Output should be sigmoid activated [0, 1]"


def test_loss_function():
    print("\n=== Testing Loss Function ===")
    criterion = BCEDiceLoss(bce_weight=0.5)

    # Dummy predictions (logits applied sigmoid already in model, but loss expects probabilities)
    pred = torch.tensor([[[[0.9, 0.1], [0.1, 0.9]]]]).float()  # Shape (1, 1, 2, 2)
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]).float()

    loss = criterion(pred, target)
    print(f"Calculated Loss: {loss.item():.4f}")

    assert loss.item() >= 0, "Loss should be non-negative"
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"


def run_full_training_cycle():
    print("\n=== Running Training Cycle (Demonstration) ===")
    # Run for 1 epoch with very small data fraction to verify pipeline works
    run_training(
        epochs=1,
        batch_size=4,
        fraction=0.01,  # Use 1% of data for speed
        lr=1e-3,
        patience=1,
        img_size=256,
    )

    # Verify checkpoint creation
    checkpoint_path = "./working/idea_1/best_model.pth"
    if os.path.exists(checkpoint_path):
        print(f"Success: Checkpoint created at {checkpoint_path}")
    else:
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path} after training."
        )


def run_inference_cycle():
    print("\n=== Running Inference Cycle (Demonstration) ===")
    # Run inference using the checkpoint generated above
    # Note: run_inference uses the 'test' mode dataset which loads ./metadata/test.csv
    # We cannot filter the test set size via arguments in run_inference,
    # but the test set is moderate size (6800), should finish quickly with batch processing.

    run_inference(
        batch_size=32,
        img_size=256,
        checkpoint_path="./working/idea_1/best_model.pth",
        submission_dir="./submission",
    )

    submission_path = "./submission/submission.csv"
    if os.path.exists(submission_path):
        df = pd.read_csv(submission_path)
        print(f"Success: Submission file created at {submission_path}")
        print(f"Submission Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        expected_cols = ["id", "class", "predicted"]
        assert (
            list(df.columns) == expected_cols
        ), f"Expected columns {expected_cols}, got {list(df.columns)}"
        assert len(df) > 0, "Submission file is empty"
    else:
        raise FileNotFoundError(f"Submission file not found at {submission_path}")


if __name__ == "__main__":
    set_seed(42)

    # 1. Verify Metrics
    test_metrics()

    # 2. Verify Dataset
    test_dataset_loading()

    # 3. Verify Model
    test_model_architecture()

    # 4. Verify Loss
    test_loss_function()

    # 5. Run Training Loop
    run_full_training_cycle()

    # 6. Run Inference Loop
    run_inference_cycle()

    print("\nAll demonstrations completed successfully.")
