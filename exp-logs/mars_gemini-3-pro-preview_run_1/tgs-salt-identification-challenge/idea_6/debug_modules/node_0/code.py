import os
import shutil
import numpy as np
import torch
import pandas as pd
import warnings

# Import from provided library files
from library.utils import (
    set_seed,
    rle_encode,
    rle_decode,
    pad_image_128,
    unpad_image_101,
    do_kaggle_metric,
)
from library.dataset import get_dataloaders, SaltDataset
from library.model import ResUNetPPM
from library.losses import BCEDiceLoss, LovaszBCELoss
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def test_utils():
    print("\n=== Testing Utils ===")

    # 1. Test RLE Encoding/Decoding
    # Create a simple 101x101 mask with a 10x10 square of 1s
    mask = np.zeros((101, 101), dtype=np.uint8)
    mask[10:20, 10:20] = 1

    encoded = rle_encode(mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    assert isinstance(encoded, str), "RLE encode should return a string"
    assert np.array_equal(mask, decoded), "Decoded mask does not match original"
    print("RLE Encode/Decode: PASSED")

    # 2. Test Padding/Unpadding
    # Create a random 101x101 image
    img = np.random.randint(0, 255, (101, 101), dtype=np.uint8)
    padded = pad_image_128(img)
    unpadded = unpad_image_101(padded)

    assert padded.shape == (128, 128), f"Padded shape mismatch: {padded.shape}"
    assert unpadded.shape == (101, 101), f"Unpadded shape mismatch: {unpadded.shape}"
    assert np.array_equal(
        img, unpadded
    ), "Unpadded image content changed (reflection padding check)"
    print("Image Pad/Unpad: PASSED")

    # 3. Test Metric
    # Perfect match
    score_perfect = do_kaggle_metric(mask, mask)
    assert np.isclose(
        score_perfect, 1.0
    ), f"Perfect match should be 1.0, got {score_perfect}"

    # No overlap
    mask_inv = 1 - mask
    score_zero = do_kaggle_metric(mask, mask_inv)
    assert np.isclose(score_zero, 0.0), f"No overlap should be 0.0, got {score_zero}"
    print("Kaggle Metric: PASSED")


def test_dataset():
    print("\n=== Testing Dataset & DataLoader ===")

    # Use small batch size for testing
    batch_size = 4

    # This will trigger cache generation if not present (fast for 2400 images)
    # or load from existing cache.
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=0,  # Use 0 workers for simple debugging to avoid multiprocessing overhead
        load_cached_data=True,
    )

    # Fetch one batch
    images, masks, ids = next(iter(train_loader))

    # Check shapes
    # Images: (B, 2, 128, 128) -> Channel 0: Intensity, Channel 1: Depth
    assert images.shape == (
        batch_size,
        2,
        128,
        128,
    ), f"Image batch shape incorrect: {images.shape}"
    # Masks: (B, 1, 128, 128)
    assert masks.shape == (
        batch_size,
        1,
        128,
        128,
    ), f"Mask batch shape incorrect: {masks.shape}"
    # IDs: (B,)
    assert len(ids) == batch_size, "IDs length mismatch"

    print(f"Batch Shapes Verified: Images {images.shape}, Masks {masks.shape}")
    print("Dataset & DataLoader: PASSED")


def test_model():
    print("\n=== Testing Model Architecture ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResUNetPPM(in_channels=2, num_classes=1, filters=16).to(
        device
    )  # Reduced filters for speed

    dummy_input = torch.randn(2, 2, 128, 128).to(device)

    # Training Mode (Expect Deep Supervision)
    model.train()
    outputs = model(dummy_input)
    assert (
        len(outputs) == 4
    ), "Model in train mode should return (logits, aux1, aux2, aux3)"
    assert outputs[0].shape == (
        2,
        1,
        128,
        128,
    ), f"Logits shape mismatch: {outputs[0].shape}"

    # Eval Mode
    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)
    assert logits.shape == (
        2,
        1,
        128,
        128,
    ), f"Eval logits shape mismatch: {logits.shape}"

    print("Model Architecture: PASSED")


def test_losses():
    print("\n=== Testing Loss Functions ===")

    # Dummy data
    logits = torch.randn(4, 1, 128, 128)
    targets = torch.randint(0, 2, (4, 1, 128, 128)).float()

    # Test BCEDiceLoss
    criterion1 = BCEDiceLoss()
    loss1 = criterion1(logits, targets)
    assert loss1.dim() == 0, "BCEDiceLoss should return a scalar"
    assert not torch.isnan(loss1), "BCEDiceLoss returned NaN"

    # Test LovaszBCELoss
    criterion2 = LovaszBCELoss()
    loss2 = criterion2(logits, targets)
    assert loss2.dim() == 0, "LovaszBCELoss should return a scalar"
    assert not torch.isnan(loss2), "LovaszBCELoss returned NaN"

    print("Loss Functions: PASSED")


def test_training_pipeline():
    print("\n=== Testing Full Training Pipeline ===")

    # Define a working directory for this demo
    working_dir = "./working/demo_execution"
    os.makedirs(working_dir, exist_ok=True)

    # Initialize Trainer with debug=True
    # debug=True limits the loop to 5 batches per epoch
    trainer = Trainer(
        base_dir=working_dir,
        batch_size=8,
        num_workers=2,
        epochs=1,  # Run only 1 epoch
        debug=True,  # Run only a few steps
    )

    # 1. Run Fit
    print("Running Trainer.fit()...")
    trainer.fit()

    # Check if checkpoint was saved
    checkpoint_path = os.path.join(working_dir, "checkpoints", "best_model.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print("Training finished and checkpoint verified.")

    # 2. Run Submission Generation
    print("Running Trainer.generate_submission()...")
    trainer.generate_submission()

    # Check if submission file was created
    sub_path = "./submission/submission.csv"
    assert os.path.exists(sub_path), f"Submission file not found at {sub_path}"

    # Validate submission format
    df = pd.read_csv(sub_path)
    assert (
        "id" in df.columns and "rle_mask" in df.columns
    ), "Submission columns mismatch"
    assert len(df) > 0, "Submission file is empty"

    print(f"Submission generated successfully at {sub_path}")
    print("Training Pipeline: PASSED")


if __name__ == "__main__":
    # Set global seed
    set_seed(42)

    print("Starting Self-Contained Demo...")

    try:
        test_utils()
        test_dataset()
        test_model()
        test_losses()
        test_training_pipeline()

        print("\nAll tests passed successfully!")

    except AssertionError as e:
        print(f"\nFAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # Print full traceback for debugging if needed, but simple print is sufficient per requirements
        import traceback

        traceback.print_exc()
        exit(1)
