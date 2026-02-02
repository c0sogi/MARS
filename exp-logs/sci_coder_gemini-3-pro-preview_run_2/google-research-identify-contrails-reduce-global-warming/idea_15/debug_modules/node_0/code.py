import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import CFG
from library.utils import set_seed, rle_encode, dice_coef_batch
from library.dataset import ContrailDataset, get_transforms
from library.model import ConvNeXtUNet
from library.losses import HybridLoss
from library.train import train_model
from library.predict import predict_and_submit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def verify_utils():
    """Verifies utility functions (RLE encoding and Dice metric)."""
    print("\n[1/5] Verifying Utilities...")

    # Test RLE Encoding
    # Create a simple 2x2 mask
    # [[0, 1],
    #  [0, 1]]
    # Flattened (Column-Major/Fortran): [0, 0, 1, 1]
    # Indices (1-based): 1, 2, 3, 4
    # Expected RLE: Start at 3, Length 2 -> "3 2"
    mask = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    rle = rle_encode(mask)
    assert rle == "3 2", f"RLE Encoding failed. Expected '3 2', got '{rle}'"

    # Test Empty RLE
    empty_mask = np.zeros((2, 2), dtype=np.uint8)
    assert rle_encode(empty_mask) == "-", "RLE Encoding for empty mask failed."

    # Test Dice Coefficient
    # Perfect overlap
    y_true = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    y_pred = torch.tensor(
        [[[[0.0, 1.0], [0.0, 1.0]]]]
    )  # Logits not needed for this specific util func test if we pass thresholded?
    # Note: dice_coef_batch applies threshold internally to y_pred.
    # If we pass 1.0, it is > 0.5.
    dice = dice_coef_batch(y_pred, y_true, threshold=0.5)
    assert np.isclose(
        dice, 1.0
    ), f"Dice score for perfect match should be 1.0, got {dice}"

    # No overlap
    y_pred_zero = torch.zeros_like(y_pred)
    dice_zero = dice_coef_batch(y_pred_zero, y_true, threshold=0.5)
    assert np.isclose(
        dice_zero, 0.0
    ), f"Dice score for no match should be 0.0, got {dice_zero}"

    print("Utils verified successfully.")


def verify_dataset():
    """Verifies dataset loading and shapes."""
    print("\n[2/5] Verifying Dataset...")

    # Initialize dataset in debug mode
    ds = ContrailDataset(
        metadata_path=CFG.train_metadata_path,
        split="train",
        transform=get_transforms("train"),
        debug=True,
        load_cached_data=False,  # Disable cache to test raw loading logic
    )

    assert len(ds) > 0, "Dataset is empty."

    # Fetch one sample
    sample = ds[0]
    image = sample["image"]
    mask = sample["mask"]
    record_id = sample["record_id"]

    # Check shapes
    # Image: (C=6, H=256, W=256)
    assert image.shape == (6, 256, 256), f"Incorrect image shape: {image.shape}"
    # Mask: (C=1, H=256, W=256)
    assert mask.shape == (1, 256, 256), f"Incorrect mask shape: {mask.shape}"

    # Check value ranges (Ash composite normalized roughly to 0-1, Diff can be neg/pos)
    # Just ensure it's not empty or NaN
    assert not torch.isnan(image).any(), "Image contains NaNs."
    assert not torch.isnan(mask).any(), "Mask contains NaNs."

    print(f"Dataset verified. Sample ID: {record_id}, Image Shape: {image.shape}")


def verify_model_and_loss():
    """Verifies model architecture and loss calculation."""
    print("\n[3/5] Verifying Model and Loss...")

    # Instantiate Model
    model = ConvNeXtUNet(
        backbone_name="convnext_tiny",
        in_channels=6,
        num_classes=1,
        pretrained=False,  # Speed up initialization
    )
    model.to(CFG.device)
    model.eval()

    # Create dummy input batch (B=2, C=6, H=256, W=256)
    dummy_input = torch.randn(2, 6, 256, 256).to(CFG.device)
    dummy_target = torch.randint(0, 2, (2, 1, 256, 256)).float().to(CFG.device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        2,
        1,
        256,
        256,
    ), f"Model output shape mismatch. Got {output.shape}"

    # Loss Calculation
    criterion = HybridLoss()
    # HybridLoss expects logits (raw output)
    loss = criterion(output, dummy_target)

    assert loss.dim() == 0, "Loss should be a scalar."
    assert loss.item() >= 0, "Loss should be non-negative."

    print("Model and Loss verified successfully.")


def run_demo_training():
    """Runs a short training loop."""
    print("\n[4/5] Running Demo Training...")

    # Override CFG for speed
    CFG.epochs = 1
    CFG.debug = True
    CFG.debug_sample_size = 16  # Small subset
    CFG.batch_size = 4

    # Setup output directories for demo
    CFG.working_dir = "./working/demo_run"
    CFG.output_dir = CFG.working_dir
    CFG.best_model_path = os.path.join(CFG.working_dir, "best_model.pth")
    CFG.submission_dir = os.path.join(CFG.working_dir, "submission")
    CFG.submission_path = os.path.join(CFG.submission_dir, "submission.csv")

    # Clean up previous demo run if exists
    if os.path.exists(CFG.best_model_path):
        os.remove(CFG.best_model_path)

    # Run training
    # This function inside library/train.py uses the global CFG we just modified
    train_model(debug=True)

    assert os.path.exists(CFG.best_model_path), "Training failed to save best_model.pth"
    print("Demo training completed.")


def run_demo_inference():
    """Runs inference using the trained model."""
    print("\n[5/5] Running Demo Inference...")

    # Ensure we are using the same paths
    CFG.working_dir = "./working/demo_run"
    CFG.best_model_path = os.path.join(CFG.working_dir, "best_model.pth")

    # Run inference
    predict_and_submit(debug=True)

    assert os.path.exists(
        CFG.submission_path
    ), "Inference failed to generate submission.csv"

    # Validate submission format
    df = pd.read_csv(CFG.submission_path)
    required_cols = ["record_id", "encoded_pixels"]
    assert all(
        col in df.columns for col in required_cols
    ), f"Submission missing columns. Found {df.columns}"
    assert len(df) > 0, "Submission file is empty."

    print("Demo inference completed.")
    print(f"Submission saved to: {CFG.submission_path}")
    print(f"Head of submission:\n{df.head()}")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    try:
        # 1. Verify Utils
        verify_utils()

        # 2. Verify Dataset
        verify_dataset()

        # 3. Verify Model & Loss
        verify_model_and_loss()

        # 4. Run Training Demo
        run_demo_training()

        # 5. Run Inference Demo
        run_demo_inference()

        print("\nAll demonstrations passed successfully!")

    except AssertionError as e:
        print(f"\n[FAIL] Assertion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Unexpected Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
