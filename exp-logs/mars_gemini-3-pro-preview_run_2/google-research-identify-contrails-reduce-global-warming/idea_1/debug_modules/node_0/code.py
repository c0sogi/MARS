import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, dice_coef, rle_encode
from library.dataset import ContrailDataset
from library.model import MobileNetUNet
from library.train import run_training
from library.predict import generate_submission


def main():
    print("Initializing Contrail Segmentation Pipeline Demonstration...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config parameters for a fast demonstration run
    print("\n[Step 1] Configuring environment for rapid testing...")
    Config.DEBUG_SAMPLE_SIZE = 20  # Use only 20 samples
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.setup()  # Create necessary directories
    seed_everything(Config.SEED)

    # ==========================================
    # 2. Verify Dataset Logic
    # ==========================================
    print("\n[Step 2] Verifying Dataset loading and preprocessing...")

    # Load metadata manually to check content
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    if Config.DEBUG_SAMPLE_SIZE:
        train_meta = train_meta.head(Config.DEBUG_SAMPLE_SIZE)

    # Instantiate Dataset
    dataset = ContrailDataset(train_meta, split="train")

    # Fetch a single sample
    image, mask = dataset[0]

    # Assertions
    # Image shape: (Channels, Height, Width) -> (6, 256, 256)
    assert image.shape == (
        6,
        256,
        256,
    ), f"Expected image shape (6, 256, 256), got {image.shape}"
    # Mask shape: (Channels, Height, Width) -> (1, 256, 256)
    assert mask.shape == (
        1,
        256,
        256,
    ), f"Expected mask shape (1, 256, 256), got {mask.shape}"
    # Check normalization (Ash channels 0-2 should be roughly in [0, 1])
    assert (
        image[0:3].min() >= 0.0 and image[0:3].max() <= 1.0
    ), "Ash composite channels not normalized to [0, 1]"
    # Check mask values (binary 0.0 or 1.0)
    unique_vals = torch.unique(mask)
    assert all(
        val in [0.0, 1.0] for val in unique_vals
    ), f"Mask contains non-binary values: {unique_vals}"

    print("  Dataset verification passed.")

    # ==========================================
    # 3. Verify Model Architecture
    # ==========================================
    print("\n[Step 3] Verifying Model architecture...")

    model = MobileNetUNet(in_channels=Config.IN_CHANNELS, num_classes=1)
    model.eval()

    # Create a dummy batch: (Batch_Size, Channels, Height, Width)
    dummy_input = torch.randn(2, 6, 256, 256)

    with torch.no_grad():
        output = model(dummy_input)

    # Assertions
    # Output shape: (Batch_Size, Num_Classes, Height, Width) -> (2, 1, 256, 256)
    assert output.shape == (
        2,
        1,
        256,
        256,
    ), f"Expected output shape (2, 1, 256, 256), got {output.shape}"

    print("  Model verification passed.")

    # ==========================================
    # 4. Verify Utility Functions
    # ==========================================
    print("\n[Step 4] Verifying Utility functions (Dice & RLE)...")

    # Test Dice Coefficient
    y_true = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    y_pred_perfect = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    y_pred_worst = torch.tensor([0, 0, 1, 1], dtype=torch.float32)

    dice_perfect = dice_coef(y_pred_perfect, y_true)
    dice_worst = dice_coef(y_pred_worst, y_true)

    assert abs(dice_perfect - 1.0) < 1e-5, f"Expected Dice 1.0, got {dice_perfect}"
    assert abs(dice_worst - 0.0) < 1e-5, f"Expected Dice 0.0, got {dice_worst}"

    # Test Run-Length Encoding
    # Create a simple 3x3 mask
    # Indices (Column-major):
    # 1 4 7
    # 2 5 8
    # 3 6 9
    # Let's mark pixels at (0,0) and (1,0) -> Indices 1 and 2.
    dummy_mask = np.zeros((3, 3), dtype=np.uint8)
    dummy_mask[0, 0] = 1
    dummy_mask[1, 0] = 1

    rle_str = rle_encode(dummy_mask)
    # Expect run starting at 1 with length 2
    assert rle_str == "1 2", f"Expected RLE '1 2', got '{rle_str}'"

    # Test empty mask
    empty_rle = rle_encode(np.zeros((3, 3)))
    assert empty_rle == "-", f"Expected empty RLE '-', got '{empty_rle}'"

    print("  Utility verification passed.")

    # ==========================================
    # 5. Run Training Loop
    # ==========================================
    print("\n[Step 5] Executing Training Loop (Fast Mode)...")

    # run_training uses the Config settings we modified earlier
    best_model_path = run_training()

    assert os.path.exists(best_model_path), "Best model file was not saved."
    print(f"  Training completed. Model saved to {best_model_path}")

    # ==========================================
    # 6. Run Inference & Submission
    # ==========================================
    print("\n[Step 6] Generating Submission...")

    # We need to ensure the test metadata exists. The provided environment has it.
    # We will use the model trained in Step 5.
    generate_submission(model_path=best_model_path)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Validate submission format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert "record_id" in sub_df.columns, "Submission missing 'record_id' column"
    assert (
        "encoded_pixels" in sub_df.columns
    ), "Submission missing 'encoded_pixels' column"

    # Check if number of rows matches test set (or sample size if we were sampling test,
    # but generate_submission loads full test set. For speed, we trust the pipeline logic
    # or could hack Config.TEST_METADATA_PATH, but here we just check existence).
    print(f"  Submission generated with {len(sub_df)} rows.")

    print("\n==========================================")
    print("Demonstration Completed Successfully.")
    print("==========================================")


if __name__ == "__main__":
    main()
