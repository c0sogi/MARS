import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Import provided library modules
from library.utils import seed_everything, compute_score
from library.dataset import RetinopathyDataset, get_dataloaders
from library.model import RetinopathyModel
from library.engine import run_training

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def test_metric():
    print("\n=== Testing Metric (QWK) ===")
    # Case 1: Perfect agreement
    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 3, 4]
    score = compute_score(y_true, y_pred)
    print(f"Perfect Agreement Score: {score}")
    assert np.isclose(score, 1.0), "Score should be 1.0 for perfect agreement"

    # Case 2: Complete disagreement
    y_true_bad = [0, 0, 0, 0, 0]
    y_pred_bad = [4, 4, 4, 4, 4]
    score_bad = compute_score(y_true_bad, y_pred_bad)
    print(f"Disagreement Score: {score_bad}")
    # Kappa can be 0 or negative for disagreement
    assert score_bad < 0.5, "Score should be low for disagreement"


def test_dataset_logic():
    print("\n=== Testing Dataset Logic ===")
    metadata_path = "./metadata/train.csv"

    # Initialize dataset
    # We use a small image size for the test to be fast
    ds = RetinopathyDataset(
        csv_file=metadata_path,
        transforms=None,  # Raw images for logic check, or use simple resize
        mode="train",
    )

    print(f"Dataset length: {len(ds)}")
    assert len(ds) > 0, "Dataset should not be empty"

    # Fetch one sample
    image, target = ds[0]
    print(f"Sample 0 Image Shape: {image.shape}")
    print(f"Sample 0 Target: {target}")

    # Verify Target Encoding (Ordinal)
    # We need to find specific examples to verify the encoding logic
    # Label 0 -> [0, 0, 0, 0]
    # Label 2 -> [1, 1, 0, 0]
    # Label 4 -> [1, 1, 1, 1]

    df = pd.read_csv(metadata_path)

    # Test Label 0
    idx_0 = df[df["diagnosis"] == 0].index[0]
    _, t0 = ds[idx_0]
    assert torch.equal(
        t0, torch.tensor([0.0, 0.0, 0.0, 0.0])
    ), f"Label 0 encoding incorrect: {t0}"

    # Test Label 2 (if exists)
    if 2 in df["diagnosis"].values:
        idx_2 = df[df["diagnosis"] == 2].index[0]
        _, t2 = ds[idx_2]
        assert torch.equal(
            t2, torch.tensor([1.0, 1.0, 0.0, 0.0])
        ), f"Label 2 encoding incorrect: {t2}"

    # Test Label 4 (if exists)
    if 4 in df["diagnosis"].values:
        idx_4 = df[df["diagnosis"] == 4].index[0]
        _, t4 = ds[idx_4]
        assert torch.equal(
            t4, torch.tensor([1.0, 1.0, 1.0, 1.0])
        ), f"Label 4 encoding incorrect: {t4}"

    print("Ordinal target encoding verified.")


def test_dataloader_and_transforms():
    print("\n=== Testing DataLoaders ===")
    batch_size = 4
    image_size = 224

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        image_size=image_size,
        num_workers=2,  # Low workers for simple test
    )

    # Fetch one batch from train
    images, targets = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    assert images.shape == (
        batch_size,
        3,
        image_size,
        image_size,
    ), "Incorrect image batch shape"
    assert targets.shape == (batch_size, 4), "Incorrect target batch shape"

    # Fetch one batch from test (returns image, id_code)
    test_images, ids = next(iter(test_loader))
    print(f"Test Batch Image Shape: {test_images.shape}")
    assert len(ids) == batch_size, "Incorrect number of IDs in test batch"


def test_model_architecture():
    print("\n=== Testing Model Architecture ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RetinopathyModel(model_name="convnext_small.fb_in1k", pretrained=False)
    model.to(device)
    model.eval()

    # Dummy input
    dummy_input = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    # Output should be (Batch, 4) for ordinal regression (4 binary tasks)
    assert output.shape == (2, 4), f"Expected output shape (2, 4), got {output.shape}"


def run_demonstration_training():
    print("\n=== Running Demonstration Training ===")

    # Parameters optimized for speed
    # 1 Epoch, small image size, small batch size
    EPOCHS = 1
    BATCH_SIZE = 16
    IMAGE_SIZE = 256
    OUTPUT_DIR = "./working/demo_run"

    run_training(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        lr=1e-4,
        weight_decay=1e-2,
        seed=42,
        patience=1,
        output_dir=OUTPUT_DIR,
    )

    # Verify outputs
    print("\n=== Verifying Outputs ===")

    # Checkpoint
    model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    if os.path.exists(model_path):
        print(f"Verified: {model_path} exists.")
    else:
        # If validation score didn't improve (unlikely with 1 epoch starting from scratch),
        # last_model might be there, or best_model might not be copied if logic is strict.
        # However, run_training saves best_model if val_score > -inf.
        # Let's check for last_model as fallback for verification.
        last_model_path = os.path.join(OUTPUT_DIR, "last_model.pth")
        if os.path.exists(last_model_path):
            print(
                f"Verified: {last_model_path} exists (best_model might not if score didn't improve)."
            )
        else:
            raise FileNotFoundError(f"No model checkpoints found in {OUTPUT_DIR}")

    # Submission
    sub_path = "./submission/submission.csv"
    if os.path.exists(sub_path):
        print(f"Verified: {sub_path} exists.")

        # Verify content format
        df = pd.read_csv(sub_path)
        print("Submission Head:")
        print(df.head())

        expected_cols = ["id_code", "diagnosis"]
        assert (
            list(df.columns) == expected_cols
        ), f"Columns mismatch. Expected {expected_cols}, got {list(df.columns)}"

        # Verify values
        assert df["diagnosis"].dtype == np.int64, "Diagnosis should be integer"
        assert (
            df["diagnosis"].min() >= 0 and df["diagnosis"].max() <= 4
        ), "Diagnosis values out of range [0, 4]"

    else:
        raise FileNotFoundError(f"Submission file not found at {sub_path}")


if __name__ == "__main__":
    # 1. Setup
    seed_everything(42)

    # 2. Unit Tests
    test_metric()
    test_dataset_logic()
    test_dataloader_and_transforms()
    test_model_architecture()

    # 3. Integration Test (Full Run)
    run_demonstration_training()

    print("\nAll demonstrations and verifications completed successfully.")
