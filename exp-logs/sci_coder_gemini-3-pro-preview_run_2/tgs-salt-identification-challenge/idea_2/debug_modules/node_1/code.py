import os
import sys
import numpy as np
import torch
import pandas as pd
import shutil

# Import from the provided library
from library.utils import set_seed, rle_encode, rle_decode, MetricMonitor
from library.dataset import SaltDataset, get_transforms
from library.model import DepthConditionedUNetPlusPlus
from library.losses import BCEDiceLoss
from library.train import run_training
from library.inference import predict_and_submit


def main():
    # 1. Setup
    print("=== 1. Setup & Configuration ===")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Clean working directory for a fresh run demonstration
    working_dir = "./working/idea_2"
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.makedirs(working_dir, exist_ok=True)
    print(f"Cleaned working directory: {working_dir}")

    # 2. Verify Utils (RLE Encoding/Decoding)
    print("\n=== 2. Verifying Utils (RLE) ===")
    # Create a synthetic mask (101x101) with a square of 1s
    mask = np.zeros((101, 101), dtype=np.uint8)
    mask[10:20, 10:20] = 1

    # Encode
    rle_str = rle_encode(mask)
    print(f"Encoded RLE string length: {len(rle_str)}")

    # Decode
    decoded_mask = rle_decode(rle_str, shape=(101, 101))

    # Verify
    assert np.array_equal(
        mask, decoded_mask
    ), "RLE Decode does not match original mask!"
    print("RLE Encode/Decode verification passed.")

    # 3. Verify Dataset
    print("\n=== 3. Verifying Dataset ===")
    # Initialize dataset (this will trigger processing from scratch since cache was cleared)
    # We use 'train' mode.
    train_ds = SaltDataset(
        mode="train",
        metadata_path="./metadata/train.csv",
        load_cached_data=True,  # Will process from scratch as files don't exist yet
        transform=get_transforms("train"),
    )

    print(f"Dataset size: {len(train_ds)}")

    # Fetch one sample
    img, mask, depth, img_id = train_ds[0]

    print(f"Sample ID: {img_id}")
    print(f"Image Shape: {img.shape}")
    print(f"Mask Shape: {mask.shape}")
    print(f"Depth Shape: {depth.shape}")

    # Assertions
    # Transforms pad to 128x128
    assert img.shape == (
        1,
        128,
        128,
    ), f"Expected image shape (1, 128, 128), got {img.shape}"
    assert mask.shape == (128, 128), f"Expected mask shape (128, 128), got {mask.shape}"
    assert isinstance(depth, torch.Tensor), "Depth should be a tensor"
    print("Dataset verification passed.")

    # 4. Verify Model Architecture
    print("\n=== 4. Verifying Model Architecture ===")
    model = DepthConditionedUNetPlusPlus(num_classes=1, deep_supervision=True)
    model.to(device)

    # Create dummy batch
    # Batch size 2, 1 channel, 128x128
    dummy_img = torch.randn(2, 1, 128, 128).to(device)
    dummy_depth = (
        torch.tensor([0.5, 0.8]).unsqueeze(1).to(device).float()
    )  # Normalized depths

    # Forward pass
    outputs = model(dummy_img, dummy_depth)

    # Verify Deep Supervision outputs
    assert isinstance(outputs, list), "Model should return a list (Deep Supervision)"
    assert len(outputs) == 4, f"Expected 4 output heads, got {len(outputs)}"

    last_output = outputs[-1]
    assert last_output.shape == (
        2,
        1,
        128,
        128,
    ), f"Expected output shape (2, 1, 128, 128), got {last_output.shape}"
    print("Model forward pass verification passed.")

    # 5. Verify Loss Function
    print("\n=== 5. Verifying Loss Function ===")
    criterion = BCEDiceLoss()

    # Dummy targets (Batch 2, 128, 128)
    dummy_targets = torch.randint(0, 2, (2, 128, 128)).float().to(device)

    loss = criterion(outputs, dummy_targets)
    print(f"Calculated Loss: {loss.item()}")

    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.item() > 0, "Loss should be positive"
    print("Loss function verification passed.")

    # 6. Run Training Loop
    print("\n=== 6. Running Training Loop (Demonstration) ===")
    # We run for 1 epoch with a reasonable batch size to ensure it completes quickly
    # The run_training function saves the best model to ./working/idea_2/best_model.pth
    run_training(
        epochs=1,
        batch_size=32,
        lr=1e-3,
        num_workers=2,
        patience=1,
        load_cached_data=True,  # Will use the cache generated in step 3
    )

    model_path = "./working/idea_2/best_model.pth"
    assert os.path.exists(model_path), "Model checkpoint was not created!"
    print("Training demonstration complete.")

    # 7. Run Inference
    print("\n=== 7. Running Inference & Submission ===")
    submission_path = "./working/idea_2/submission.csv"

    predict_and_submit(
        model_path=model_path,
        output_path=submission_path,
        batch_size=32,
        num_workers=2,
        load_cached_data=True,  # Will process test data and cache it
    )

    # Verify submission file
    assert os.path.exists(submission_path), "Submission file not found!"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission rows: {len(df_sub)}")
    print(f"Submission columns: {df_sub.columns.tolist()}")

    # Test set has 1000 images
    assert len(df_sub) == 1000, f"Expected 1000 rows in submission, got {len(df_sub)}"
    assert (
        "id" in df_sub.columns and "rle_mask" in df_sub.columns
    ), "Missing required columns"

    # Check if RLE strings are valid (or empty)
    # Just check the first one
    first_rle = df_sub.iloc[0]["rle_mask"]
    if pd.notna(first_rle) and first_rle != "":
        # Try decoding it
        decoded_test = rle_decode(first_rle, shape=(101, 101))
        assert decoded_test.shape == (101, 101)

    print("Inference and submission verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
