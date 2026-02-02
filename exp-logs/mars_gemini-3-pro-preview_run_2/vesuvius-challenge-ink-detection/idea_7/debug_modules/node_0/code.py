import os
import torch
import numpy as np
import pandas as pd
import shutil
from library.config import Config
from library.utils import seed_everything, rle_encoding, fbeta_score
from library.dataset import InkDataset
from library.model import build_model
from library.train import run_training
from library.inference import inference


def main():
    print("Starting Vesuvius Ink Detection Library Demonstration...")

    # --- 1. Configuration & Setup ---
    # Override Config for speed and demonstration purposes
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script
    # Lower baseline to force model saving logic to trigger if possible, though random weights might fail
    BASELINE_SCORE = 0.0

    # Ensure working directory is clean or exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print("Configuration set. Random seed applied.")

    # --- 2. Test Utility Functions ---
    print("\n--- Testing Utility Functions ---")

    # Test RLE Encoding
    # Create a simple mask: 0 0 1 1 1 0 1
    # Indices (1-based):    1 2 3 4 5 6 7
    # Expected RLE: '3 3 7 1' (Start at 3 len 3, Start at 7 len 1)
    dummy_mask = np.array([[0, 0, 1, 1], [1, 0, 1, 0]])  # Flattened: 0 0 1 1 1 0 1 0
    # Note: RLE function flattens row-major.
    # Row 1: 0, 0, 1, 1
    # Row 2: 1, 0, 1, 0
    # Concat: 0, 0, 1, 1, 1, 0, 1, 0
    # Indices: 1  2  3  4  5  6  7  8
    # Runs: Starts at 3 (len 3 -> 3,4,5), Starts at 7 (len 1 -> 7)
    expected_rle = "3 3 7 1"
    calculated_rle = rle_encoding(dummy_mask)
    assert (
        calculated_rle == expected_rle
    ), f"RLE failed. Expected '{expected_rle}', got '{calculated_rle}'"
    print("RLE Encoding verified.")

    # Test F-beta Score
    # Preds: 0.8 (Ink), 0.2 (No), 0.9 (Ink)
    # Targets: 1 (Ink), 0 (No), 1 (Ink)
    # Threshold 0.5 -> Preds Binary: 1, 0, 1 -> Perfect Match -> Score 1.0
    preds_t = torch.tensor([0.8, 0.2, 0.9])
    targets_t = torch.tensor([1.0, 0.0, 1.0])
    score = fbeta_score(preds_t, targets_t, beta=0.5, threshold=0.5)
    assert np.isclose(score, 1.0), f"F-beta score failed. Expected 1.0, got {score}"
    print("F-beta Score verified.")

    # --- 3. Test Dataset Loading ---
    print("\n--- Testing Dataset ---")

    # Load metadata and take a small subset for testing
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH).head(4)

    # Initialize Dataset
    # This will trigger MIP generation for the fragments involved in the first 4 rows
    print("Initializing InkDataset (this involves loading/processing volumes)...")
    dataset = InkDataset(train_df, mode="train", load_cached_data=False)

    assert len(dataset) == 4, f"Dataset length mismatch. Expected 4, got {len(dataset)}"

    # Fetch one sample
    image, label, mask, idx = dataset[0]

    # Verify Shapes
    # Image: (3, 512, 512) -> 3 channels for MIPs
    # Label: (1, 512, 512)
    # Mask: (1, 512, 512)
    print(
        f"Sample shapes - Image: {image.shape}, Label: {label.shape}, Mask: {mask.shape}"
    )

    assert image.shape == (
        3,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), "Incorrect image shape"
    assert label.shape == (
        1,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), "Incorrect label shape"
    assert mask.shape == (1, Config.TILE_SIZE, Config.TILE_SIZE), "Incorrect mask shape"
    assert image.dtype == torch.float32, "Image tensor should be float32"

    print("Dataset loaded and verified successfully.")

    # --- 4. Test Model Architecture ---
    print("\n--- Testing Model Architecture ---")

    model = build_model()
    model.eval()

    # Create a dummy batch
    dummy_input = torch.randn(2, 3, Config.TILE_SIZE, Config.TILE_SIZE)

    print("Forward pass with dummy input...")
    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape: (Batch, 1, H, W)
    expected_shape = (2, 1, Config.TILE_SIZE, Config.TILE_SIZE)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print("Model architecture verified.")

    # --- 5. Test Training Loop ---
    print("\n--- Testing Training Loop (Run Training) ---")

    # We run for 1 epoch on the debug subset (defined inside run_training via Config.DEBUG)
    # We set a low baseline to test the saving logic if the model learns anything (unlikely in 1 epoch with 20 samples, but functional check)
    try:
        run_training(
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            debug=True,
            baseline_score=BASELINE_SCORE,
        )
    except Exception as e:
        raise AssertionError(f"Training loop failed with error: {e}")

    # Check if model file was created (it might not be if score is 0, but usually with random weights and threshold 0.5, F0.5 > 0)
    # If not saved, we manually save a dummy one for the inference step to succeed
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(
            "Model not saved by training loop (score too low). Saving dummy model for inference test."
        )
        torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    else:
        print(f"Model successfully saved at {Config.MODEL_SAVE_PATH}")

    # --- 6. Test Inference Pipeline ---
    print("\n--- Testing Inference Pipeline ---")

    # Run inference in debug mode
    try:
        inference(
            threshold=0.5,
            batch_size=Config.BATCH_SIZE,
            num_workers=0,
            debug=True,
            use_tta=False,
        )
    except Exception as e:
        raise AssertionError(f"Inference failed with error: {e}")

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission file content head:")
    print(sub_df.head())

    assert (
        "Id" in sub_df.columns and "Predicted" in sub_df.columns
    ), "Submission columns missing"
    assert len(sub_df) > 0, "Submission file is empty"

    print("Inference pipeline verified.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
