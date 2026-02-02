import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library components
from library.config import get_config, Config
from library.utils import (
    rle_encode,
    rle_decode,
    pad_image,
    unpad_image,
    do_kaggle_metric,
)
from library.dataset import get_dataloaders
from library.model import SaltNet, predict
from library.loss import BCELovaszLoss
from library.train import train_model


def set_seed(seed=42):
    """Sets fixed seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_utilities():
    """Verifies utility functions for RLE, padding, and metrics."""
    print("--- Verifying Utilities ---")

    # 1. RLE Encode/Decode Round-trip
    original_mask = np.zeros((101, 101), dtype=np.uint8)
    original_mask[20:30, 20:30] = 1  # 10x10 square of salt

    rle_str = rle_encode(original_mask)
    decoded_mask = rle_decode(rle_str, shape=(101, 101))

    assert np.array_equal(original_mask, decoded_mask), "RLE round-trip failed."
    print("RLE Encode/Decode: OK")

    # 2. Padding/Unpadding Round-trip
    # Input size is 101, model requires 128 (powers of 2)
    img = np.random.randint(0, 255, (101, 101), dtype=np.uint8)
    padded_img = pad_image(img, target_size=128)
    assert padded_img.shape == (128, 128), f"Padding shape mismatch: {padded_img.shape}"

    unpadded_img = unpad_image(padded_img, original_size=101)
    assert unpadded_img.shape == (
        101,
        101,
    ), f"Unpadding shape mismatch: {unpadded_img.shape}"
    assert np.array_equal(img, unpadded_img), "Padding/Unpadding content mismatch."
    print("Image Padding: OK")

    # 3. Metric Calculation
    # Case A: Perfect Match
    y_true = np.zeros((2, 101, 101), dtype=np.uint8)
    y_true[0, 50:60, 50:60] = 1
    y_pred_perfect = y_true.copy()
    score_perfect = do_kaggle_metric(y_pred_perfect, y_true)
    assert (
        score_perfect == 1.0
    ), f"Metric should be 1.0 for perfect match, got {score_perfect}"

    # Case B: Complete Mismatch (excluding empty-empty case)
    y_pred_bad = np.zeros_like(y_true)
    # Image 0: True has salt, Pred has none -> Score 0
    # Image 1: True has no salt, Pred has none -> Score 1 (Empty-Empty match is perfect IoU)
    # Mean Score -> 0.5
    score_bad = do_kaggle_metric(y_pred_bad, y_true)
    assert score_bad == 0.5, f"Metric should be 0.5 for mixed case, got {score_bad}"
    print("Metric Calculation: OK")


def test_data_loading(config):
    """Verifies DataLoader functionality and data shapes."""
    print("\n--- Verifying Data Loading ---")

    # Force no cache to test raw loading logic
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=False
    )

    # Fetch a single batch
    images, masks, depths, ids = next(iter(train_loader))

    # Verify Shapes
    # Configured batch size is 4 for this demo
    B = config.BATCH_SIZE
    assert images.shape == (B, 1, 128, 128), f"Image shape mismatch: {images.shape}"
    assert masks.shape == (B, 1, 128, 128), f"Mask shape mismatch: {masks.shape}"
    assert depths.shape == (B, 1), f"Depth shape mismatch: {depths.shape}"
    assert len(ids) == B, "IDs length mismatch"

    # Verify Data Types and Ranges
    # Images are normalized (approx mean 0, std 1), not 0-255
    assert images.dtype == torch.float32, "Images should be float32"
    assert masks.dtype == torch.float32, "Masks should be float32 for BCE loss"
    assert torch.equal(masks, masks.round()), "Masks should be binary (0.0 or 1.0)"

    print(f"Data Loading: OK (Batch Size: {B})")


def test_model_logic(config):
    """Verifies Model architecture and Loss function."""
    print("\n--- Verifying Model & Loss ---")

    device = config.DEVICE
    model = SaltNet().to(device)
    criterion = BCELovaszLoss()

    # Dummy inputs
    dummy_img = torch.randn(2, 1, 128, 128).to(device)
    dummy_depth = torch.randn(2, 1).to(device)
    dummy_mask = torch.randint(0, 2, (2, 1, 128, 128)).float().to(device)

    # 1. Forward Pass
    logits = model(dummy_img, dummy_depth)
    assert logits.shape == (2, 1, 128, 128), f"Output shape mismatch: {logits.shape}"
    print("Model Forward Pass: OK")

    # 2. Loss Calculation
    loss = criterion(logits, dummy_mask)
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    # 3. Backward Pass
    loss.backward()
    # Check if gradients are populated
    param = list(model.parameters())[0]
    assert param.grad is not None, "Gradients not computed"
    print("Loss & Backward Pass: OK")


def run_pipeline_demo(config):
    """Runs a shortened training and inference pipeline."""
    print("\n--- Running Pipeline Demo ---")

    # Clean working directory to ensure fresh run
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)

    # 1. Train
    # This uses library.train.train_model
    print("Starting Training (1 Epoch)...")
    train_model(config)

    # Check for checkpoint
    if not os.path.exists(config.CHECKPOINT_PATH):
        # If model didn't improve (possible with 1 epoch and random init), manually save for inference test
        print(
            "Note: Checkpoint not saved by training loop (Metric didn't improve). Saving manually for inference test."
        )
        os.makedirs(os.path.dirname(config.CHECKPOINT_PATH), exist_ok=True)
        model = SaltNet()
        torch.save(model.state_dict(), config.CHECKPOINT_PATH)
    else:
        print("Checkpoint saved successfully.")

    # 2. Inference
    # This uses library.model.predict
    print("Starting Inference...")
    predict(config)

    # Check for submission
    if os.path.exists(config.SUBMISSION_PATH):
        df = pd.read_csv(config.SUBMISSION_PATH)
        print(f"Submission generated at: {config.SUBMISSION_PATH}")
        print(f"Submission rows: {len(df)}")
        assert len(df) > 0, "Submission file is empty"
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)

    # Configure for speed:
    # DEBUG=True enables small dataset subset (50 samples)
    # Epochs=1, Batch Size=4 ensures very fast iteration
    config = get_config(debug=True, epochs=1, batch_size=4)

    # Further restrict sample size for extreme speed in demo
    config.DEBUG_SAMPLE_SIZE = 20

    # 1. Verify Utilities
    test_utilities()

    # 2. Verify Data Loading
    test_data_loading(config)

    # 3. Verify Model & Loss
    test_model_logic(config)

    # 4. Run Pipeline (Train + Predict)
    run_pipeline_demo(config)

    print("\nAll demonstrations completed successfully.")
