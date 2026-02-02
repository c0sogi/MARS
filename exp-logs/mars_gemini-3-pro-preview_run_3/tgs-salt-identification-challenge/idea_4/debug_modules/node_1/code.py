import os
import shutil
import numpy as np
import pandas as pd
import torch
import cv2
import importlib

# Import from provided library files
import library.config
import library.utils
import library.dataset
import library.model
import library.losses
import library.trainer

# Cite debug_lesson_1: Reload modules to ensure updates are picked up in persistent session
importlib.reload(library.config)
importlib.reload(library.utils)
importlib.reload(library.dataset)
importlib.reload(library.model)
importlib.reload(library.losses)
importlib.reload(library.trainer)

from library.config import Config
from library.utils import set_seed, rle_encode, rle_decode, calc_map
from library.dataset import SaltDataset
from library.model import SaltModel
from library.losses import BCEDiceLoss, LovaszHingeLoss
from library.trainer import Trainer


def clean_cache(config, mode="train"):
    """Helper to clean up numpy cache files to ensure fresh data loading."""
    debug_suffix = "_debug" if config.DEBUG else ""
    files = [
        f"cached_{mode}{debug_suffix}_images.npy",
        f"cached_{mode}{debug_suffix}_masks.npy",
        f"cached_{mode}{debug_suffix}_depths.npy",
        f"cached_{mode}{debug_suffix}_ids.npy",
    ]
    for f in files:
        p = os.path.join(config.CACHE_DIR, f)
        if os.path.exists(p):
            os.remove(p)


def run_demo():
    print("=== Salt Segmentation Pipeline Demo ===\n")

    # 1. Configuration Setup
    # We enable debug mode and set epochs to 1 for a fast demonstration.
    config = Config(debug=True, epochs=1)

    # Ensure reproducibility
    set_seed(config.SEED)
    print(f"[Config] Initialized. Device: {config.DEVICE}, Epochs: {config.EPOCHS}")

    # 2. Verify Utilities
    print("\n[Utils] Verifying utility functions...")

    # Test RLE Encoding/Decoding
    # Create a 101x101 mask with a 10x10 square of salt
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[50:60, 50:60] = 1

    encoded_rle = rle_encode(dummy_mask)
    decoded_mask = rle_decode(encoded_rle, shape=(101, 101))

    assert np.array_equal(dummy_mask, decoded_mask), "RLE Round-trip failed!"
    assert isinstance(encoded_rle, str), "RLE encode should return a string"

    # Test mAP Calculation
    # Case 1: Perfect match
    y_true = np.ones((2, 101, 101), dtype=np.uint8)
    y_pred_perfect = np.ones((2, 101, 101), dtype=np.float32)
    score_perfect = calc_map(y_true, y_pred_perfect, threshold=0.5)
    assert np.isclose(
        score_perfect, 1.0
    ), f"mAP should be 1.0 for perfect match, got {score_perfect}"

    # Case 2: No match
    y_pred_zero = np.zeros((2, 101, 101), dtype=np.float32)
    score_zero = calc_map(y_true, y_pred_zero, threshold=0.5)
    assert np.isclose(
        score_zero, 0.0
    ), f"mAP should be 0.0 for no match, got {score_zero}"

    print("[Utils] RLE and mAP functions verified.")

    # 3. Verify Dataset
    print("\n[Dataset] Verifying SaltDataset...")

    # Load metadata
    if not os.path.exists(config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {config.TRAIN_METADATA_PATH}")

    full_train_df = pd.read_csv(config.TRAIN_METADATA_PATH)

    # Initialize dataset with a small limit to test loading logic
    # This will create cache files in config.CACHE_DIR
    clean_cache(config, "train")  # Ensure we start fresh
    ds_debug = SaltDataset(full_train_df, mode="train", config=config, limit_size=10)

    # Fetch one sample
    img, mask, img_id = ds_debug[0]

    # Verify Shapes (Model expects 128x128 padded input)
    # Image: (Channels=4, H=128, W=128) -> RGB + Depth
    assert img.shape == (4, 128, 128), f"Unexpected image shape: {img.shape}"
    # Mask: (Channels=1, H=128, W=128)
    assert mask.shape == (1, 128, 128), f"Unexpected mask shape: {mask.shape}"

    # Verify Data Ranges
    assert 0.0 <= img.min() and img.max() <= 1.0, "Image should be normalized to [0, 1]"
    assert mask.max() <= 1.0, "Mask should be binary [0, 1]"

    print("[Dataset] Shapes and value ranges verified.")

    # Clean cache again so the Trainer step generates cache consistent with its own data subset
    clean_cache(config, "train")

    # 4. Verify Model and Losses
    print("\n[Model] Verifying Architecture and Losses...")
    model = SaltModel(config).to(config.DEVICE)

    # Create a dummy batch
    batch_img = img.unsqueeze(0).to(config.DEVICE)  # (1, 4, 128, 128)
    batch_mask = mask.unsqueeze(0).to(config.DEVICE)  # (1, 1, 128, 128)

    # Forward Pass
    logits = model(batch_img)
    assert logits.shape == (
        1,
        1,
        128,
        128,
    ), f"Model output shape mismatch: {logits.shape}"

    # Loss Calculation
    criterion_bce = BCEDiceLoss()
    criterion_lov = LovaszHingeLoss()

    loss1 = criterion_bce(logits, batch_mask)
    loss2 = criterion_lov(logits, batch_mask)

    assert not torch.isnan(loss1), "BCE Dice Loss returned NaN"
    assert not torch.isnan(loss2), "Lovasz Loss returned NaN"

    print(
        f"[Model] Forward pass successful. BCE+Dice: {loss1.item():.4f}, Lovasz: {loss2.item():.4f}"
    )

    # 5. Verify Trainer (Integration Test)
    print("\n[Trainer] Starting Integration Test (Train/Val/Submit Loop)...")

    # Create temporary subsets of metadata to run a fast training loop
    # We use the ./working directory for temporary files
    temp_train_path = os.path.join(config.WORKING_DIR, "temp_train.csv")
    temp_val_path = os.path.join(config.WORKING_DIR, "temp_val.csv")
    temp_test_path = os.path.join(config.WORKING_DIR, "temp_test.csv")

    # Save small subsets (batch_size=8 in debug, so use 16 samples for 2 batches)
    full_train_df.head(16).to_csv(temp_train_path, index=False)

    val_df = pd.read_csv(config.VAL_METADATA_PATH)
    val_df.head(8).to_csv(temp_val_path, index=False)

    test_df = pd.read_csv(config.TEST_METADATA_PATH)
    test_df.head(8).to_csv(temp_test_path, index=False)

    # Update Config to point to these temporary files
    config.TRAIN_METADATA_PATH = temp_train_path
    config.VAL_METADATA_PATH = temp_val_path
    config.TEST_METADATA_PATH = temp_test_path

    # Initialize Trainer
    trainer = Trainer(config)

    # Execute Fit
    # This runs: Training Loop -> Validation -> LR Sched -> Model Saving -> Threshold Opt -> Submission
    trainer.fit()

    # Verify Artifacts
    best_model_path = config.get_model_save_path("best_model.pth")
    submission_path = config.SUBMISSION_PATH

    if not os.path.exists(best_model_path):
        raise FileNotFoundError("Trainer failed to save best_model.pth")

    if not os.path.exists(submission_path):
        raise FileNotFoundError("Trainer failed to generate submission.csv")

    # Verify Submission Content
    sub_df = pd.read_csv(submission_path)
    assert len(sub_df) == 8, f"Submission should have 8 rows, got {len(sub_df)}"
    assert "rle_mask" in sub_df.columns, "Submission missing 'rle_mask' column"

    print(f"[Trainer] Integration test passed.")
    print(f"          Model saved to: {best_model_path}")
    print(f"          Submission saved to: {submission_path}")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
