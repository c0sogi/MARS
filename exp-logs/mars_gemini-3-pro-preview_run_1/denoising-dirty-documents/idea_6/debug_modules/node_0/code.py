import os
import sys
import numpy as np
import torch
import pandas as pd
import cv2

# Import from the provided library
from library.config import DEVICE, WORKING_DIR, SUBMISSION_DIR, SEED, PATCH_SIZE
from library.utils import seed_everything, rmse_score
from library.model import get_data, ShallowUNet, generate_submission, predict_tta
from library.dataset import DenoisingDataset, get_transforms
from library.train_engine import run_fold


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Seeding
    seed_everything(SEED)
    print(f"Device: {DEVICE}")
    print(f"Working Directory: {WORKING_DIR}")

    # 2. Data Loading and Verification
    print("\n--- 1. Data Loading & Verification ---")
    # Load data (uses caching mechanism internally)
    (train_ids, train_noisy, train_clean), (test_ids, test_noisy) = get_data(
        load_cached_data=True
    )

    print(f"Train samples: {len(train_ids)}")
    print(f"Test samples: {len(test_ids)}")

    # Assertions to verify data integrity
    assert len(train_ids) == len(train_noisy) == len(train_clean), "Train data mismatch"
    assert len(test_ids) == len(test_noisy), "Test data mismatch"
    assert train_noisy[0].shape == train_clean[0].shape, "Input/Target shape mismatch"

    # Check image content (grayscale)
    assert len(train_noisy[0].shape) == 2, "Images should be 2D (grayscale)"
    print("Data loaded and verified successfully.")

    # 3. Dataset and Transform Verification
    print("\n--- 2. Dataset & Transform Verification ---")
    # Create a small subset for testing
    subset_size = 5
    subset_noisy = train_noisy[:subset_size]
    subset_clean = train_clean[:subset_size]

    # Initialize dataset with training transforms
    train_transform = get_transforms(mode="train")
    ds = DenoisingDataset(subset_noisy, subset_clean, transform=train_transform)

    # Fetch one sample
    img_tensor, mask_tensor = ds[0]

    # Verify shapes (C, H, W) and Patch Size
    print(f"Dataset output shape: {img_tensor.shape}")
    assert img_tensor.shape == (
        1,
        PATCH_SIZE,
        PATCH_SIZE,
    ), f"Expected (1, {PATCH_SIZE}, {PATCH_SIZE})"
    assert mask_tensor.shape == (
        1,
        PATCH_SIZE,
        PATCH_SIZE,
    ), f"Expected (1, {PATCH_SIZE}, {PATCH_SIZE})"

    # Verify normalization [0, 1]
    assert img_tensor.min() >= 0.0 and img_tensor.max() <= 1.0, "Input not normalized"
    assert (
        mask_tensor.min() >= 0.0 and mask_tensor.max() <= 1.0
    ), "Target not normalized"
    print("Dataset and Transforms verified.")

    # 4. Model Architecture Verification
    print("\n--- 3. Model Architecture Verification ---")
    model = ShallowUNet().to(DEVICE)

    # Create dummy input (Batch, Channel, Height, Width)
    dummy_input = torch.randn(2, 1, 160, 160).to(DEVICE)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model output shape: {output.shape}")
    assert output.shape == dummy_input.shape, "Model output shape mismatch"
    assert (
        output.min() >= 0 and output.max() <= 1
    ), "Model output not in [0, 1] (Sigmoid check)"

    # Verify TTA function
    tta_output = predict_tta(model, dummy_input[0:1])
    assert tta_output.shape == (1, 1, 160, 160), "TTA output shape mismatch"
    print("Model architecture verified.")

    # 5. Training Loop Demonstration
    print("\n--- 4. Training Loop Demonstration ---")
    # We will train Fold 0 for just 2 epochs on a small subset to demonstrate the pipeline
    # We need enough samples for at least one batch (Batch Size is 16 in config)
    # Let's take 20 samples: 16 for train, 4 for validation

    demo_indices = np.arange(min(30, len(train_noisy)))
    np.random.shuffle(demo_indices)

    train_idx = demo_indices[:20]  # Use first 20 for training logic
    val_idx = demo_indices[20:25]  # Use next 5 for validation logic

    # Prepare subset arrays
    t_imgs = train_noisy[train_idx]
    t_masks = train_clean[train_idx]
    v_imgs = train_noisy[val_idx]
    v_masks = train_clean[val_idx]

    print(
        f"Running Fold 0 training on {len(t_imgs)} train samples and {len(v_imgs)} val samples."
    )

    # Run training for 2 epochs
    best_rmse = run_fold(
        fold_idx=0,
        train_imgs=t_imgs,
        train_masks=t_masks,
        val_imgs=v_imgs,
        val_masks=v_masks,
        epochs=2,
    )

    print(f"Training finished. Best RMSE: {best_rmse:.6f}")

    # Verify model file creation
    model_path = os.path.join(WORKING_DIR, "model_fold_0.pth")
    assert os.path.exists(model_path), "Model checkpoint not saved"
    print(f"Checkpoint verified at: {model_path}")

    # 6. Inference and Submission Demonstration
    print("\n--- 5. Inference & Submission Demonstration ---")
    # Run inference on a small subset of test data
    test_subset_ids = test_ids[:5]
    test_subset_imgs = test_noisy[:5]

    print(f"Generating submission for {len(test_subset_ids)} test images...")

    # This function loads models from WORKING_DIR.
    # It will find 'model_fold_0.pth' generated above and use it.
    generate_submission(test_subset_ids, test_subset_imgs)

    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file not created"

    # Verify submission content
    df_sub = pd.read_csv(submission_path)
    print(f"Submission shape: {df_sub.shape}")
    print("First few rows:")
    print(df_sub.head())

    # Check columns
    assert (
        "id" in df_sub.columns and "value" in df_sub.columns
    ), "Submission columns missing"

    # Check value range
    assert (
        df_sub["value"].min() >= 0 and df_sub["value"].max() <= 1
    ), "Submission values out of range"

    # Check ID format (image_row_col)
    sample_id = df_sub.iloc[0]["id"]
    parts = sample_id.split("_")
    assert len(parts) >= 3, f"Invalid ID format: {sample_id}"

    print("Submission verified successfully.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
