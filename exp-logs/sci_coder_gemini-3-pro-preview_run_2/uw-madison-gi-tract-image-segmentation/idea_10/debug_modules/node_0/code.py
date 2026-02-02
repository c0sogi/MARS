import os
import torch
import numpy as np
import pandas as pd
import shutil
import warnings

# Import from the provided library
from library.config import Config
from library.utils import (
    set_seed,
    rle_encode,
    rle_decode,
    compute_hausdorff_distance,
    compute_dice_score,
)
from library.dataset import get_dataloaders
from library.model import ShuffleNetPSPNet
from library.loss import CombinedLoss
from library.post_processing import keep_largest_component_3d
from library.train import run_training

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def demo_utils():
    print("\n--- Demonstrating Utilities ---")

    # 1. RLE Encoding/Decoding
    shape = (100, 100)
    mask = np.zeros(shape, dtype=np.uint8)
    # Create a square mask
    mask[20:40, 20:40] = 1

    encoded = rle_encode(mask)
    decoded = rle_decode(encoded, shape)

    assert np.array_equal(mask, decoded), "RLE Decode -> Encode mismatch!"
    print("RLE Encoding/Decoding: Verified.")

    # 2. Dice Score
    # Perfect overlap
    dice_perfect = compute_dice_score(mask, mask)
    assert np.isclose(
        dice_perfect, 1.0
    ), f"Dice score for perfect match should be 1.0, got {dice_perfect}"

    # No overlap
    mask_inv = 1 - mask
    dice_none = compute_dice_score(mask, mask_inv)
    assert np.isclose(
        dice_none, 0.0, atol=1e-5
    ), f"Dice score for disjoint masks should be ~0.0, got {dice_none}"
    print("Dice Score Calculation: Verified.")

    # 3. Hausdorff Distance
    # Create 3D volumes: (Depth, Height, Width)
    vol_shape = (3, 100, 100)
    pred_vol = np.zeros(vol_shape, dtype=np.uint8)
    true_vol = np.zeros(vol_shape, dtype=np.uint8)

    # Place a pixel in center of middle slice
    pred_vol[1, 50, 50] = 1
    # Place a pixel 10 pixels away in Width dimension
    true_vol[1, 50, 60] = 1

    # Normalized distance: 10 pixels / 100 width = 0.1
    hd = compute_hausdorff_distance(pred_vol, true_vol, shape=(100, 100))
    print(f"Computed Hausdorff Distance: {hd:.4f}")
    assert hd > 0, "Hausdorff distance should be positive for distinct objects."
    assert np.isclose(hd, 0.1, atol=1e-4), f"Expected HD ~0.1, got {hd}"
    print("Hausdorff Distance: Verified.")


def demo_post_processing():
    print("\n--- Demonstrating Post-Processing ---")

    # Create a 3D volume with two components
    # Shape: (Depth=5, Height=50, Width=50)
    vol = np.zeros((5, 50, 50), dtype=np.uint8)

    # Component 1: Large (Main object)
    vol[1:4, 10:30, 10:30] = 1
    main_size = np.sum(vol)

    # Component 2: Small (Artifact)
    vol[1, 40:42, 40:42] = 1
    artifact_size = 4

    total_pixels_before = np.sum(vol)
    assert total_pixels_before == main_size + artifact_size

    # Apply Largest Component Retention
    cleaned_vol = keep_largest_component_3d(vol)

    total_pixels_after = np.sum(cleaned_vol)

    assert (
        total_pixels_after == main_size
    ), "Post-processing did not remove the smaller component correctly."
    assert cleaned_vol[1, 40, 40] == 0, "Artifact pixel still present."
    print("3D Largest Component Retention: Verified.")


def demo_dataset_and_model():
    print("\n--- Demonstrating Dataset and Model ---")

    # Use CPU or GPU
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # Get DataLoaders in Debug mode (small subset)
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, debug=True
    )

    # Fetch a single batch
    batch = next(iter(train_loader))
    images = batch["image"]
    masks = batch["mask"]

    print(
        f"Batch Image Shape: {images.shape}"
    )  # Expected: (B, 3, 256, 256) -> 2.5D input
    print(f"Batch Mask Shape: {masks.shape}")  # Expected: (B, 3, 256, 256) -> 3 classes

    # Verify Shapes
    assert images.shape == (
        Config.BATCH_SIZE,
        Config.IN_CHANNELS,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    )
    assert masks.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    )

    # Verify Normalization (Images should be approx 0-1)
    assert (
        images.min() >= 0.0 and images.max() <= 1.0
    ), "Image data not normalized to [0, 1]"

    # Initialize Model
    model = ShuffleNetPSPNet(
        num_classes=Config.NUM_CLASSES, in_channels=Config.IN_CHANNELS
    )
    model.to(device)

    # Forward Pass
    images = images.to(device)
    masks = masks.to(device)

    outputs = model(images)

    print(f"Model Output Shape: {outputs.shape}")
    assert outputs.shape == masks.shape, "Model output shape mismatch."

    # Loss Calculation
    loss_fn = CombinedLoss().to(device)
    loss = loss_fn(outputs, masks)

    print(f"Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.item() > 0, "Loss should be positive."

    print("Dataset, Model, and Loss: Verified.")


def demo_full_pipeline():
    print("\n--- Demonstrating Full Training Pipeline ---")

    # Run a very short training loop
    # Config overrides are handled by modifying the class attributes directly
    # before calling run_training, or relying on the defaults set in main.

    run_training(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        debug=True,
        load_cached_data=False,  # Force reprocessing for demo
    )

    # Verify Submission Generation
    if os.path.exists(Config.SUBMISSION_PATH):
        df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file generated with {len(df)} rows.")

        # Check columns
        expected_cols = ["id", "class", "predicted"]
        assert all(
            col in df.columns for col in expected_cols
        ), "Submission columns mismatch."

        # Check if predictions are RLE strings or empty
        # In debug mode with random weights, we might get empty predictions or random noise
        # Just ensure the format is string (or NaN if pandas loaded it that way, but RLE should be string)
        if len(df) > 0:
            sample_pred = df.iloc[0]["predicted"]
            # It can be NaN/None if empty, or a string
            pass

        print("Pipeline execution successful.")
    else:
        raise FileNotFoundError("Submission file was not created!")


if __name__ == "__main__":
    # 1. Setup Configuration for Demo
    set_seed(42)

    # Override Config for speed and demo purposes
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_execution"
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce compute requirements
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Very small subset
    Config.NUM_WORKERS = 2  # Reduce overhead

    # Ensure working directory exists
    Config.setup()

    # 2. Run Demonstrations
    try:
        demo_utils()
        demo_post_processing()
        demo_dataset_and_model()
        demo_full_pipeline()

        print("\nAll demonstrations passed successfully!")

    except AssertionError as e:
        print(f"\nAssertion Failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
