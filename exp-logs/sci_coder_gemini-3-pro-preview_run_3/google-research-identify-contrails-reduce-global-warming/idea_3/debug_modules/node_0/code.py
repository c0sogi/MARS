import sys
import os
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import set_seed, rle_encode, dice_score
from library.dataset import ContrailDataset
from library.model import UNetPlusPlus
from library.training import ContrailTrainer


def main():
    print("Initializing Contrail Identification Demo...")

    # 1. Setup Configuration for Fast Demonstration
    # We modify the Config class attributes directly to ensure the demo runs quickly
    print("Configuring environment for speed...")
    Config.DEBUG = True  # Limits dataset to 500 samples
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size for demo
    Config.NUM_WORKERS = 0  # Main process only to avoid multiprocessing overhead
    Config.CHECKPOINT_TOP_K = 1  # Keep only 1 checkpoint

    # Ensure reproducibility
    set_seed(Config.SEED)

    # 2. Verify Utility Functions
    print("\n--- Verifying Utility Functions ---")

    # Test RLE Encoding
    # Create a simple 3x3 mask:
    # [[0, 1, 0],
    #  [0, 1, 0],
    #  [0, 1, 0]]
    # Flattened column-major (F):
    # Col 0: 0, 0, 0
    # Col 1: 1, 1, 1
    # Col 2: 0, 0, 0
    # Sequence: 0, 0, 0, 1, 1, 1, 0, 0, 0
    # 1s are at indices (1-based) 4, 5, 6.
    # Expected RLE: "4 3" (Start at 4, length 3)
    dummy_mask = np.zeros((3, 3), dtype=int)
    dummy_mask[:, 1] = 1
    rle_result = rle_encode(dummy_mask)
    print(f"RLE Test Result: '{rle_result}'")
    assert (
        rle_result == "4 3"
    ), f"RLE Encoding failed. Expected '4 3', got '{rle_result}'"

    # Test Dice Score
    # Pred: [1, 1, 0, 0], True: [1, 1, 0, 1]
    # Intersection: 2 (indices 0, 1)
    # Union (sum): 2 + 3 = 5
    # Dice: 2*2 / 5 = 0.8
    y_pred = torch.tensor([1.0, 1.0, 0.0, 0.0])
    y_true = torch.tensor([1.0, 1.0, 0.0, 1.0])
    score = dice_score(y_pred, y_true, smooth=0, threshold=0.5)
    print(f"Dice Score Test Result: {score}")
    assert abs(score - 0.8) < 1e-5, f"Dice Score failed. Expected 0.8, got {score}"

    print("Utilities verified successfully.")

    # 3. Verify Dataset
    print("\n--- Verifying Dataset ---")

    # Initialize Dataset (Train)
    # DEBUG=True limits this to 500 samples
    train_ds = ContrailDataset(split="train", load_cached_data=False, debug=True)
    print(f"Train Dataset Size: {len(train_ds)}")
    assert len(train_ds) > 0, "Dataset is empty."

    # Fetch one sample
    img, mask = train_ds[0]
    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Mask Shape: {mask.shape}")

    # Assertions
    # Image: (Channels, H, W) -> (6, 256, 256)
    assert img.shape == (6, 256, 256), f"Unexpected image shape: {img.shape}"
    # Mask: (Channels, H, W) -> (1, 256, 256)
    assert mask.shape == (1, 256, 256), f"Unexpected mask shape: {mask.shape}"
    assert isinstance(img, torch.Tensor), "Image is not a Tensor"
    assert isinstance(mask, torch.Tensor), "Mask is not a Tensor"

    print("Dataset verified successfully.")

    # 4. Verify Model Architecture
    print("\n--- Verifying Model Architecture ---")

    model = UNetPlusPlus()
    model.eval()

    # Create dummy input batch (Batch=2, Ch=6, H=256, W=256)
    dummy_input = torch.randn(2, 6, 256, 256)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    # Output should be (Batch, 1, H, W) -> (2, 1, 256, 256)
    assert output.shape == (2, 1, 256, 256), f"Unexpected output shape: {output.shape}"

    print("Model architecture verified successfully.")

    # 5. Verify Training Pipeline
    print("\n--- Verifying Training Pipeline (Single Epoch) ---")

    # We will run the ContrailTrainer.run() method.
    # Due to Config modifications (EPOCHS=1, DEBUG=True), this should be fast.

    trainer = ContrailTrainer()

    # Ensure directories exist (handled by Config, but good to double check logic flow)
    assert os.path.exists(Config.CHECKPOINT_DIR), "Checkpoint directory not created."

    print("Starting Trainer Run...")
    trainer.run()

    # Verify Artifacts
    print("\nChecking generated artifacts...")

    # 1. Checkpoint
    checkpoints = os.listdir(Config.CHECKPOINT_DIR)
    print(f"Found checkpoints: {checkpoints}")
    assert len(checkpoints) > 0, "No checkpoints saved."

    # 2. Best Model
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model file not found."
    print(f"Best model found at: {best_model_path}")

    # 3. Submission
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file not found."

    # Validate Submission Format
    df_sub = pd.read_csv(submission_path)
    print(f"Submission rows: {len(df_sub)}")
    assert "record_id" in df_sub.columns, "Submission missing record_id column"
    assert (
        "encoded_pixels" in df_sub.columns
    ), "Submission missing encoded_pixels column"
    assert len(df_sub) > 0, "Submission file is empty"

    print("Training pipeline verified successfully.")
    print("\nAll demonstrations completed.")


if __name__ == "__main__":
    main()
