import os
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Import components from the provided library
from library.config import Config
from library.utils import set_seed, rle_encode, dice_coefficient, GlobalDiceTracker
from library.dataset import ContrailDataset, get_transforms
from library.model import CascadedUNet
from library.loss import DeepSupervisionLoss
from library.train import fit
from library.predict import generate_submission


def main():
    # ==========================================
    # 0. Setup and Configuration Override
    # ==========================================
    print(">>> Setting up demonstration environment...")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set seed for reproducibility
    set_seed(42)

    # Override Config paths to use a specific demo directory in ./working
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Update Config attributes dynamically
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.BEST_MODEL_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set compute parameters for a quick run
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2
    # We will use debug=True in fit/predict to limit dataset size

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Model Path: {Config.BEST_MODEL_PATH}")

    # ==========================================
    # 1. Verify Utilities
    # ==========================================
    print("\n>>> Verifying Utility Functions...")

    # Test RLE Encoding
    # Create a simple 3x3 mask
    # Pixels numbered top-to-bottom, left-to-right
    # (0,0) -> 1, (1,0) -> 2, (2,0) -> 3
    # (0,1) -> 4, (1,1) -> 5, (2,1) -> 6
    # (0,2) -> 7, (1,2) -> 8, (2,2) -> 9
    mask_test = np.zeros((3, 3), dtype=np.uint8)
    mask_test[0, 0] = 1  # Pixel 1
    mask_test[1, 0] = 1  # Pixel 2
    mask_test[0, 1] = 1  # Pixel 4

    # Expected RLE: "1 2 4 1" (Start at 1, length 2; Start at 4, length 1)
    rle_result = rle_encode(mask_test)
    assert (
        rle_result == "1 2 4 1"
    ), f"RLE Encoding failed. Expected '1 2 4 1', got '{rle_result}'"
    print("    RLE Encoding: OK")

    # Test Dice Coefficient
    y_true = torch.tensor([1.0, 1.0, 0.0])
    y_pred = torch.tensor([1.0, 0.0, 0.0])
    # Intersection = 1, Union = 1 + 2 = 3. Dice = 2*1 / 3 = 0.666...
    dice = dice_coefficient(y_pred, y_true).item()
    assert abs(dice - 0.666666) < 1e-4, f"Dice Coefficient failed. Got {dice}"
    print("    Dice Coefficient: OK")

    # ==========================================
    # 2. Verify Dataset and Transforms
    # ==========================================
    print("\n>>> Verifying Dataset and Transforms...")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Initialize Dataset (Train mode)
    ds = ContrailDataset(
        metadata_df=train_df.head(10),  # Use small subset
        split="train",
        transform=get_transforms("train"),
    )

    # Fetch one sample
    image, mask = ds[0]

    # Check shapes
    # Image: (C, H, W) where C=6 (3 Ash + 3 Diff)
    # Mask: (1, H, W)
    assert image.shape == (6, 256, 256), f"Image shape mismatch. Got {image.shape}"
    assert mask.shape == (1, 256, 256), f"Mask shape mismatch. Got {mask.shape}"

    # Check value ranges (should be normalized approx 0-1 or standardized)
    # Based on normalization logic, values are clipped to [0, 1]
    assert image.min() >= 0.0 and image.max() <= 1.0, "Image values out of range [0, 1]"
    assert mask.min() >= 0.0 and mask.max() <= 1.0, "Mask values out of range [0, 1]"

    print(f"    Sample Loaded: Image {image.shape}, Mask {mask.shape}")
    print("    Dataset Verification: OK")

    # ==========================================
    # 3. Verify Model Architecture
    # ==========================================
    print("\n>>> Verifying Model Architecture...")

    device = Config.DEVICE
    model = CascadedUNet().to(device)

    # Create dummy input batch (B, C, H, W)
    dummy_input = torch.randn(2, 6, 256, 256).to(device)

    # Forward pass
    stage1_out, stage2_out = model(dummy_input)

    # Check output shapes (B, 1, H, W)
    assert stage1_out.shape == (
        2,
        1,
        256,
        256,
    ), f"Stage 1 output shape incorrect: {stage1_out.shape}"
    assert stage2_out.shape == (
        2,
        1,
        256,
        256,
    ), f"Stage 2 output shape incorrect: {stage2_out.shape}"

    print("    Forward Pass: OK")

    # ==========================================
    # 4. Verify Loss Function
    # ==========================================
    print("\n>>> Verifying Loss Function...")

    criterion = DeepSupervisionLoss()
    dummy_target = torch.randint(0, 2, (2, 1, 256, 256)).float().to(device)

    loss, metrics = criterion((stage1_out, stage2_out), dummy_target)

    assert isinstance(loss, torch.Tensor), "Loss is not a tensor"
    assert (
        "loss_stage1" in metrics and "loss_stage2" in metrics
    ), "Metrics missing stage info"
    assert not torch.isnan(loss), "Loss returned NaN"

    print(f"    Computed Loss: {loss.item():.4f}")
    print("    Loss Function: OK")

    # ==========================================
    # 5. Run Training Loop (Mini-Epoch)
    # ==========================================
    print("\n>>> Running Training Loop (Debug Mode)...")

    # Run fit with debug=True to use a small subset (Config.DEBUG_SAMPLE_SIZE)
    # and only 1 epoch for speed.
    fit(epochs=1, batch_size=Config.BATCH_SIZE, learning_rate=1e-3, debug=True)

    # Verify model checkpoint was created
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not saved."
    print(f"    Checkpoint saved at: {Config.BEST_MODEL_PATH}")
    print("    Training Loop: OK")

    # ==========================================
    # 6. Run Inference
    # ==========================================
    print("\n>>> Running Inference (Debug Mode)...")

    # Run inference with debug=True
    generate_submission(debug=True)

    # Verify submission file
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not generated."

    # Check submission content format
    sub_df = pd.read_csv(submission_path)
    assert "record_id" in sub_df.columns, "Submission missing 'record_id' column"
    assert (
        "encoded_pixels" in sub_df.columns
    ), "Submission missing 'encoded_pixels' column"
    assert len(sub_df) > 0, "Submission file is empty"

    print(f"    Submission generated with {len(sub_df)} rows.")
    print(f"    First row: {sub_df.iloc[0].to_dict()}")
    print("    Inference: OK")

    print("\n==========================================")
    print("ALL CHECKS PASSED. DEMO COMPLETE.")
    print("==========================================")


if __name__ == "__main__":
    main()
