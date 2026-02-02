import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, rle_encode, dice_coef
from library.data import get_loaders
from library.model import ContrailUNet
from library.training import train_model, DiceBCELoss

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Contrail Identification Pipeline Demo ===\n")

    # --- 1. Configure for Fast Demonstration ---
    print("[1] Configuring environment for rapid execution...")
    # Modify Config class attributes directly to control the execution flow
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.DEBUG_SAMPLE_SIZE = 20  # Use a tiny subset of data
    Config.TOP_K_CHECKPOINTS = 2
    Config.NUM_WORKERS = 2  # Reduce workers to avoid overhead on small data

    # Ensure working directory exists (handled by Config import, but good to double check)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")
    print("    Configuration updated.\n")

    # --- 2. Verify Utility Functions ---
    print("[2] Verifying Utility Functions...")
    set_seed(Config.SEED)

    # Test RLE Encoding
    # Create a simple 4x4 mask:
    # 0 1 0 0
    # 0 1 0 0
    # 0 0 0 0
    # 0 0 0 0
    # Flattened (Column-major/Fortran): 0,0,0,0 (col1), 1,1,0,0 (col2), ...
    # Indices (1-based): 5, 6 are 1s.
    dummy_mask = np.zeros((4, 4), dtype=np.uint8)
    dummy_mask[0:2, 1] = 1
    rle_str = rle_encode(dummy_mask)
    # Expected: start at 5, length 2 -> "5 2"
    assert rle_str == "5 2", f"RLE Encoding failed. Expected '5 2', got '{rle_str}'"
    print("    RLE Encoding: OK")

    # Test Dice Coefficient
    # Perfect match
    y_true = torch.tensor([1.0, 1.0, 0.0, 0.0])
    y_pred = torch.tensor([1.0, 1.0, 0.0, 0.0])
    score = dice_coef(y_pred, y_true)
    assert torch.isclose(score, torch.tensor(1.0)), f"Dice (Perfect) failed: {score}"

    # No overlap
    y_pred_bad = torch.tensor([0.0, 0.0, 1.0, 1.0])
    score_bad = dice_coef(y_pred_bad, y_true, smooth=0.0)
    assert torch.isclose(
        score_bad, torch.tensor(0.0)
    ), f"Dice (Bad) failed: {score_bad}"
    print("    Dice Coefficient: OK\n")

    # --- 3. Verify Data Loading ---
    print("[3] Verifying Data Pipeline...")
    train_loader, val_loader, test_loader = get_loaders(
        debug=True, batch_size=Config.BATCH_SIZE
    )

    # Fetch one batch
    images, masks = next(iter(train_loader))

    # Check shapes
    # Image: (B, 6, 256, 256) -> 6 channels (3 Ash t, 3 Ash diff)
    expected_img_shape = (Config.BATCH_SIZE, 6, 256, 256)
    expected_mask_shape = (Config.BATCH_SIZE, 1, 256, 256)

    assert (
        images.shape == expected_img_shape
    ), f"Image shape mismatch. Got {images.shape}"
    assert masks.shape == expected_mask_shape, f"Mask shape mismatch. Got {masks.shape}"

    # Check value ranges (Ash composite should be roughly 0-1 due to normalization)
    print(
        f"    Batch loaded. Image stats: min={images.min():.4f}, max={images.max():.4f}"
    )
    print("    Data Loading: OK\n")

    # --- 4. Verify Model Architecture ---
    print("[4] Verifying Model Architecture...")
    model = ContrailUNet(
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=None,  # Random init for speed/no download dependency check
        in_channels=Config.N_CHANNELS,
        classes=1,
    )
    model.to(Config.DEVICE)

    # Forward pass
    images = images.to(Config.DEVICE)
    with torch.no_grad():
        logits = model(images)

    assert (
        logits.shape == expected_mask_shape
    ), f"Output shape mismatch. Got {logits.shape}"
    print("    Forward pass successful.")
    print("    Model Architecture: OK\n")

    # --- 5. Run Training Loop ---
    print("[5] Executing Training Loop (Debug Mode)...")
    # train_model handles the loop, validation, and saving best model
    # We use the library function directly
    train_model(debug=True)

    # Verify output file exists
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Training finished but {Config.BEST_MODEL_PATH} was not found."
        )

    print(f"    Model saved successfully to {Config.BEST_MODEL_PATH}")
    print("    Training Loop: OK\n")

    # --- 6. Inference and Submission Generation ---
    print("[6] Verifying Inference and Submission Format...")

    # Load the trained model
    best_state = torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
    model.load_state_dict(best_state)
    model.eval()

    submission_rows = []

    # Process a small subset of test data
    # We iterate manually to demonstrate inference logic
    print("    Running inference on test batch...")
    with torch.no_grad():
        for batch_idx, (images, record_ids) in enumerate(test_loader):
            if batch_idx >= 1:
                break  # Just do one batch

            images = images.to(Config.DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Thresholding
            preds = (probs > 0.5).float().cpu().numpy()

            for i in range(len(record_ids)):
                # Extract single mask: (1, H, W) -> (H, W)
                mask_np = preds[i, 0, :, :]

                # Encode
                rle = rle_encode(mask_np)

                # If empty, use '-'
                if rle == "":
                    rle = "-"

                submission_rows.append(
                    {"record_id": record_ids[i], "encoded_pixels": rle}
                )

    # Create DataFrame
    sub_df = pd.DataFrame(submission_rows)
    print(f"    Generated {len(sub_df)} predictions.")
    print("    Sample Prediction:")
    print(sub_df.head(2))

    # Check format
    assert "record_id" in sub_df.columns
    assert "encoded_pixels" in sub_df.columns

    print("    Inference & Submission: OK\n")

    print("=== Demo Complete: All components verified successfully ===")


if __name__ == "__main__":
    main()
