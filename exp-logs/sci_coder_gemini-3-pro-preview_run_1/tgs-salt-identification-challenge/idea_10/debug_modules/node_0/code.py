import os
import shutil
import numpy as np
import torch
import pandas as pd
import warnings

# Import provided library components
from library.utils import set_seed, rle_encode, rle_decode, do_kaggle_metric
from library.dataset import get_loaders, get_test_loader
from library.model import ClassificationGatedResUNet
from library.loss import CompoundLoss
from library.train import Trainer
from library.inference import predict_and_submit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Salt Segmentation Demo ===\n")

    # 1. Configuration
    DEMO_DIR = "./working/demo_execution"
    CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(DEMO_DIR, "predictions")

    # Clean up previous demo run if exists
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Verify Utilities
    print("\n[1/6] Verifying Utilities (RLE & Metric)...")

    # Test RLE Encoding/Decoding
    # Create a 101x101 mask with a 10x10 square of 1s
    synthetic_mask = np.zeros((101, 101), dtype=np.uint8)
    synthetic_mask[10:20, 10:20] = 1

    encoded = rle_encode(synthetic_mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    assert isinstance(encoded, str), "RLE encode should return a string"
    assert np.array_equal(
        synthetic_mask, decoded
    ), "RLE decode did not match original mask"
    print("  - RLE Encode/Decode check passed.")

    # Test Metric
    # Perfect match
    score_perfect = do_kaggle_metric(
        synthetic_mask[None, ...], synthetic_mask[None, ...]
    )
    assert np.isclose(
        score_perfect, 1.0
    ), f"Perfect match should have score 1.0, got {score_perfect}"

    # No overlap
    empty_mask = np.zeros((101, 101), dtype=np.uint8)
    score_mismatch = do_kaggle_metric(synthetic_mask[None, ...], empty_mask[None, ...])
    # IoU is 0. Thresholds 0.5-0.95 will all fail. Score should be 0.
    assert np.isclose(
        score_mismatch, 0.0
    ), f"Mismatch should have score 0.0, got {score_mismatch}"
    print("  - Metric check passed.")

    # 3. Verify Dataset Loading
    print("\n[2/6] Verifying Data Loading...")

    # Use debug=True to load a tiny subset
    train_loader, val_loader = get_loaders(
        train_metadata_path="./metadata/train.csv",
        val_metadata_path="./metadata/val.csv",
        cache_dir=CACHE_DIR,
        batch_size=4,
        num_workers=2,
        load_cached_data=False,  # Force processing to verify logic
        debug=True,
    )

    batch_images, batch_masks = next(iter(train_loader))

    # Verify shapes: Images (B, 2, 128, 128), Masks (B, 1, 128, 128)
    print(f"  - Batch Image Shape: {batch_images.shape}")
    print(f"  - Batch Mask Shape: {batch_masks.shape}")

    assert batch_images.shape == (4, 2, 128, 128), "Incorrect image batch shape"
    assert batch_masks.shape == (4, 1, 128, 128), "Incorrect mask batch shape"
    assert isinstance(batch_images, torch.Tensor), "Images should be a Tensor"
    print("  - Data Loader check passed.")

    # 4. Verify Model Architecture
    print("\n[3/6] Verifying Model Architecture...")

    model = ClassificationGatedResUNet(in_channels=2, classes=1).to(device)

    # Move batch to device
    batch_images = batch_images.to(device)

    # Forward pass
    outputs = model(batch_images)

    # Check output keys
    expected_keys = ["logits", "cls", "aux_32", "aux_64"]
    assert all(
        k in outputs for k in expected_keys
    ), f"Model output missing keys. Found: {outputs.keys()}"

    # Check output shapes
    # logits: (B, 1, 128, 128)
    # cls: (B, 1)
    assert outputs["logits"].shape == (
        4,
        1,
        128,
        128,
    ), f"Logits shape mismatch: {outputs['logits'].shape}"
    assert outputs["cls"].shape == (
        4,
        1,
    ), f"Classification head shape mismatch: {outputs['cls'].shape}"
    print("  - Model forward pass check passed.")

    # 5. Verify Loss Function
    print("\n[4/6] Verifying Loss Function...")

    criterion = CompoundLoss()
    batch_masks = batch_masks.to(device)

    loss = criterion(outputs, batch_masks)

    assert torch.is_tensor(loss), "Loss should be a tensor"
    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    print(f"  - Calculated Loss: {loss.item():.4f}")
    print("  - Loss function check passed.")

    # 6. Verify Training Loop
    print("\n[5/6] Verifying Training Loop...")

    # Initialize Trainer with minimal settings for speed
    trainer = Trainer(
        epochs=1,
        batch_size=4,
        num_workers=2,
        checkpoint_dir=CHECKPOINT_DIR,
        cache_dir=CACHE_DIR,
        debug=True,  # Uses subset of data
    )

    # Run training
    trainer.fit()

    # Check if checkpoint was saved
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(
        best_model_path
    ), "best_model.pth was not saved after training"
    print("  - Training loop completed and checkpoint saved.")

    # 7. Verify Inference
    print("\n[6/6] Verifying Inference and Submission...")

    # Run inference using the newly trained checkpoint
    predict_and_submit(
        checkpoint_dir=CHECKPOINT_DIR,
        output_dir=SUBMISSION_DIR,
        cache_dir=CACHE_DIR,
        batch_size=4,
        num_workers=2,
        device=device,
    )

    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "submission.csv was not generated"

    # Validate submission format
    df_sub = pd.read_csv(submission_path)
    assert (
        "id" in df_sub.columns and "rle_mask" in df_sub.columns
    ), "Submission missing required columns"
    assert len(df_sub) > 0, "Submission file is empty"

    print(f"  - Submission generated at {submission_path}")
    print(f"  - First few rows:\n{df_sub.head()}")
    print("  - Inference check passed.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
