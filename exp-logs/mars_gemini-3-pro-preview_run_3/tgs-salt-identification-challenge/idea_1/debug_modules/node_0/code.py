import os
import sys
import numpy as np
import torch
import pandas as pd
import random
import shutil

# Import from the provided library
from library.utils import rle_encode, rle_decode, do_kaggle_metric, MinMaxNormalizer
from library.dataset import SaltDataset, get_dataloaders, get_transforms
from library.model import LinkNetResNet18
from library.train import train_segmentation_model, BCEDiceLoss

# Configuration for the demonstration
SEED = 42
WORKING_DIR = "./working/demo_run"
SUBMISSION_DIR = "./submission"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_utils():
    print("--- Verifying Utils ---")

    # 1. Test RLE Encoding/Decoding
    # Create a synthetic 101x101 mask with a square in the middle
    mask = np.zeros((101, 101), dtype=np.uint8)
    mask[50:60, 50:60] = 1

    encoded = rle_encode(mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    assert isinstance(encoded, str), "RLE encode should return a string"
    assert np.array_equal(
        mask, decoded
    ), "RLE decode should reconstruct the original mask"
    print("RLE Encode/Decode: OK")

    # 2. Test Kaggle Metric (IoU MAP)
    # Case 1: Perfect match
    score_perfect = do_kaggle_metric(mask[None, ...], mask[None, ...])
    assert np.isclose(
        score_perfect, 1.0
    ), f"Perfect match should have score 1.0, got {score_perfect}"

    # Case 2: No overlap
    mask_shifted = np.zeros((101, 101), dtype=np.uint8)
    mask_shifted[10:20, 10:20] = 1  # Disjoint from 50:60
    score_zero = do_kaggle_metric(mask[None, ...], mask_shifted[None, ...])
    assert np.isclose(
        score_zero, 0.0
    ), f"No overlap should have score 0.0, got {score_zero}"
    print("Kaggle Metric: OK")

    # 3. Test Normalizer
    data = np.array([10, 20, 30, 40, 50])
    norm = MinMaxNormalizer()
    transformed = norm.fit_transform(data)
    assert np.isclose(transformed.min(), 0.0)
    assert np.isclose(transformed.max(), 1.0)
    assert np.isclose(transformed[2], 0.5)  # 30 is mid of 10-50
    print("Normalizer: OK")


def verify_dataset():
    print("\n--- Verifying Dataset ---")

    # Load metadata manually to create a dataset instance
    meta_path = "./metadata/train_metadata.csv"
    if not os.path.exists(meta_path):
        print(f"Metadata not found at {meta_path}. Skipping dataset verification.")
        return

    df = pd.read_csv(meta_path)
    # Use a small subset for speed
    df_subset = df.head(10).copy()

    # Mock depth normalizer
    depth_norm = MinMaxNormalizer()
    depth_norm.fit(df_subset["z"].values)

    # Instantiate Dataset
    dataset = SaltDataset(
        metadata_df=df_subset,
        transform=get_transforms("train"),
        depth_normalizer=depth_norm,
        mode="train",
    )

    # Test __getitem__
    input_tensor, mask_tensor, img_id = dataset[0]

    # Verify Shapes
    # Input: (3, 128, 128) -> Channel 0,1 are image, Channel 2 is depth
    # Mask: (1, 128, 128)
    print(f"Sample Input Shape: {input_tensor.shape}")
    print(f"Sample Mask Shape: {mask_tensor.shape}")

    assert input_tensor.shape == (
        3,
        128,
        128,
    ), f"Expected input (3, 128, 128), got {input_tensor.shape}"
    assert mask_tensor.shape == (
        1,
        128,
        128,
    ), f"Expected mask (1, 128, 128), got {mask_tensor.shape}"
    assert isinstance(input_tensor, torch.Tensor), "Output should be a torch Tensor"

    # Verify DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4, num_workers=0)
    batch_inputs, batch_masks, _ = next(iter(train_loader))

    assert batch_inputs.shape[0] == 4, "Batch size should be 4"
    assert batch_inputs.shape[1] == 3, "Channel dim should be 3"
    print("Dataset & DataLoader: OK")


def verify_model():
    print("\n--- Verifying Model ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinkNetResNet18(num_classes=1, pretrained=False)  # False for speed
    model.to(device)
    model.eval()

    # Create dummy input (Batch=2, Channels=3, H=128, W=128)
    dummy_input = torch.randn(2, 3, 128, 128).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Expected output: (2, 1, 128, 128)
    assert output.shape == (
        2,
        1,
        128,
        128,
    ), f"Expected output (2, 1, 128, 128), got {output.shape}"
    print("Model Architecture: OK")


def verify_training_pipeline():
    print("\n--- Verifying Training Pipeline ---")

    # Clean up previous runs
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)

    # Run the training function with minimal parameters
    # epochs=1 to ensure it finishes quickly
    # batch_size=16 to fit easily in memory
    print("Starting 1-epoch training run...")

    try:
        train_segmentation_model(
            epochs=1,
            batch_size=16,
            learning_rate=1e-3,
            patience=1,
            checkpoint_dir=WORKING_DIR,
        )
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e

    # Check artifacts
    model_path = os.path.join(WORKING_DIR, "best_model.pth")
    submission_path = "./submission/submission.csv"

    assert os.path.exists(model_path), "Best model file was not created."
    assert os.path.exists(submission_path), "Submission file was not created."

    # Check submission content
    sub_df = pd.read_csv(submission_path)
    assert (
        "id" in sub_df.columns and "rle_mask" in sub_df.columns
    ), "Submission columns missing"
    assert len(sub_df) > 0, "Submission file is empty"

    print(f"Training Pipeline: OK. Submission generated with {len(sub_df)} rows.")


if __name__ == "__main__":
    set_seed(SEED)

    # Execute verification steps
    verify_utils()
    verify_dataset()
    verify_model()
    verify_training_pipeline()

    print("\nAll verifications passed successfully.")
