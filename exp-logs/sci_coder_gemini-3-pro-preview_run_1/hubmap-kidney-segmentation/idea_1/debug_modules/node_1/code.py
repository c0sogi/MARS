import os
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed, rle_encode, rle_decode, dice_coeff
from library.dataset import HuBMAPDataset
from library.model import LinkNetResNet18
from library.train import train_model
from library.inference import generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def verify_utils():
    """
    Validates RLE encoding/decoding and Dice coefficient calculation.
    """
    print("\n--- Verifying Utilities ---")

    # 1. Test RLE Encoding/Decoding
    shape = (100, 100)
    # Create a synthetic binary mask with a square in the middle
    mask = np.zeros(shape, dtype=np.uint8)
    mask[25:75, 25:75] = 1

    encoded = rle_encode(mask)
    decoded = rle_decode(encoded, shape)

    assert isinstance(encoded, str), "RLE encoding should return a string."
    assert np.array_equal(mask, decoded), "Decoded mask does not match original mask."
    print("RLE Encoding/Decoding passed.")

    # 2. Test Dice Coefficient
    # Case 1: Perfect overlap
    pred_perfect = torch.ones((1, 10, 10))
    target_perfect = torch.ones((1, 10, 10))
    dice_p = dice_coeff(pred_perfect, target_perfect)
    assert np.isclose(
        dice_p, 1.0, atol=1e-4
    ), f"Dice should be 1.0 for perfect overlap, got {dice_p}"

    # Case 2: No overlap
    pred_none = torch.zeros((1, 10, 10))
    dice_n = dice_coeff(pred_none, target_perfect)
    # Smooth factor is small, so dice should be close to 0
    assert np.isclose(
        dice_n, 0.0, atol=1e-4
    ), f"Dice should be ~0.0 for no overlap, got {dice_n}"

    print("Dice Coefficient calculation passed.")


def verify_dataset():
    """
    Validates the HuBMAPDataset class.
    """
    print("\n--- Verifying Dataset ---")

    # Load metadata manually to create a small subset
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata file not found: {Config.TRAIN_METADATA_PATH}"
        )

    df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    # Use a small subset for verification
    subset_df = df.head(2)

    # Instantiate dataset
    # Note: This will generate cache files in ./working/cache
    dataset = HuBMAPDataset(subset_df, mode="train", load_cached_data=False)

    assert len(dataset) > 0, "Dataset should not be empty."

    # Fetch one sample
    img, mask, meta = dataset[0]

    # Verify shapes
    # Image: (C, H, W) -> (3, TILE_SIZE, TILE_SIZE)
    # Mask: (1, H, W) -> (1, TILE_SIZE, TILE_SIZE)
    expected_size = Config.TILE_SIZE

    assert isinstance(img, torch.Tensor), "Image should be a torch Tensor."
    assert isinstance(mask, torch.Tensor), "Mask should be a torch Tensor."

    assert img.shape == (
        3,
        expected_size,
        expected_size,
    ), f"Expected image shape (3, {expected_size}, {expected_size}), got {img.shape}"
    assert mask.shape == (
        1,
        expected_size,
        expected_size,
    ), f"Expected mask shape (1, {expected_size}, {expected_size}), got {mask.shape}"

    print(f"Dataset verification passed. Sample shape: {img.shape}")


def verify_model():
    """
    Validates the LinkNetResNet18 model architecture.
    """
    print("\n--- Verifying Model Architecture ---")

    device = torch.device("cpu")  # Use CPU for simple shape verification
    model = LinkNetResNet18(in_channels=3, classes=1)
    model.to(device)
    model.eval()

    # Create dummy input: (Batch=2, Channels=3, Height=256, Width=256)
    # Using 256 to ensure it works with downsampling/upsampling logic (divisible by 32)
    dummy_input = torch.randn(2, 3, 256, 256).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    # Expected output: (Batch=2, Classes=1, Height=256, Width=256)
    assert output.shape == (
        2,
        1,
        256,
        256,
    ), f"Expected output shape (2, 1, 256, 256), got {output.shape}"

    print("Model architecture verification passed.")


def run_training_demo():
    """
    Demonstrates the training loop using the train_model function.
    """
    print("\n--- Running Training Demo ---")

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 3  # Use only 3 images from metadata
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size

    # Run training
    # This handles dataloader creation, model instantiation, and the training loop
    best_dice = train_model(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        load_cached_data=False,  # Force regeneration for the demo subset
    )

    print(f"Training demo finished. Best Dice: {best_dice}")

    # Verify checkpoint creation
    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), f"Model checkpoint not found at {Config.MODEL_CHECKPOINT_PATH}"
    print("Model checkpoint verified.")


def run_inference_demo():
    """
    Demonstrates the inference pipeline using generate_submission.
    """
    print("\n--- Running Inference Demo ---")

    # Ensure Config is still set for speed (though generate_submission re-reads some)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 2  # Process only 2 test images

    # Run submission generation
    generate_submission(
        checkpoint_path=Config.MODEL_CHECKPOINT_PATH,
        output_path=Config.SUBMISSION_PATH,
        debug=Config.DEBUG,
        debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    )

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "id" in df_sub.columns and "predicted" in df_sub.columns
    ), "Submission file missing required columns."

    print(f"Inference demo finished. Submission shape: {df_sub.shape}")
    print("Sample submission rows:")
    print(df_sub.head())


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)

    # Ensure directories exist (handled by Config.setup(), but good to confirm)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    try:
        # 1. Verify Utilities
        verify_utils()

        # 2. Verify Dataset
        verify_dataset()

        # 3. Verify Model
        verify_model()

        # 4. Run Training
        run_training_demo()

        # 5. Run Inference
        run_inference_demo()

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        raise e
