import os
import shutil
import numpy as np
import torch
import pandas as pd
from library.utils import load_data, seed_everything, get_device
from library.data_loader import DualStreamDataset, get_dataloaders
from library.model import DSSVNet
from library.train import run_training
from library.predict import generate_submission


def run_demo():
    # 1. Setup
    print("--- 1. Setup ---")
    DEMO_DIR = "./working/demo_run"
    INPUT_DIR = "./input"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(42)
    device = get_device()
    print(f"Running on device: {device}")
    print(f"Working directory: {DEMO_DIR}")

    # 2. Data Loading Verification
    print("\n--- 2. Verifying Data Loading (library.utils.load_data) ---")
    limit_n = 4
    print(f"Loading subset of {limit_n} training samples...")

    # Load data without caching to avoid writing small debug files to main cache locations
    # We use a specific demo cache dir
    X, y, ids = load_data(
        split="train",
        load_cached_data=False,
        limit_size=limit_n,
        cache_dir=DEMO_DIR,
        input_dir=INPUT_DIR,
    )

    # Expected shapes:
    # X: (N, 2, 64, 256, 256) -> (Streams, Channels, H, W)
    # y: (N,)
    # ids: (N,)
    print(f"Loaded X shape: {X.shape}")
    print(f"Loaded y shape: {y.shape}")

    assert len(X) == limit_n, f"Expected {limit_n} samples, got {len(X)}"
    assert X.shape[1:] == (2, 64, 256, 256), f"Unexpected X dimensions: {X.shape}"
    assert len(y) == limit_n
    assert len(ids) == limit_n
    print("Data loading verification passed.")

    # 3. Dataset & DataLoader Verification
    print("\n--- 3. Verifying Dataset & DataLoader (library.data_loader) ---")
    dataset = DualStreamDataset(X, y, ids)

    # Test __getitem__
    item = dataset[0]
    (even_stream, odd_stream), target = item

    print(f"Even stream shape: {even_stream.shape}")
    print(f"Odd stream shape: {odd_stream.shape}")
    print(f"Target: {target}")

    assert torch.is_tensor(even_stream)
    assert torch.is_tensor(odd_stream)
    assert even_stream.shape == (64, 256, 256)
    assert odd_stream.shape == (64, 256, 256)
    assert isinstance(target.item(), float)

    # Test get_dataloaders wrapper
    # We use a small batch size
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=2,
        input_dir=INPUT_DIR,
        cache_dir=DEMO_DIR,
        load_cached_data=False,
        limit_size=4,  # Small limit for speed
        num_workers=0,  # Use 0 workers for simple debugging
        seed=42,
    )

    batch = next(iter(train_loader))
    (b_even, b_odd), b_targets = batch
    assert b_even.shape == (2, 64, 256, 256)
    assert b_targets.shape[0] == 2
    print("Dataset and DataLoader verification passed.")

    # 4. Model Architecture Verification
    print("\n--- 4. Verifying Model Architecture (library.model.DSSVNet) ---")
    model = DSSVNet(pretrained=False)  # False to avoid downloading weights during demo
    model.to(device)
    model.eval()

    # Create dummy input
    dummy_even = torch.randn(2, 64, 256, 256).to(device)
    dummy_odd = torch.randn(2, 64, 256, 256).to(device)

    with torch.no_grad():
        output = model(dummy_even, dummy_odd)

    print(f"Model output shape: {output.shape}")
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"
    print("Model architecture verification passed.")

    # 5. Training Pipeline Verification
    print("\n--- 5. Verifying Training Pipeline (library.train.run_training) ---")
    # Run training for 1 epoch with a very small dataset
    best_model_path = run_training(
        epochs=1,
        batch_size=2,
        lr=1e-4,
        input_dir=INPUT_DIR,
        cache_dir=DEMO_DIR,
        limit_size=8,  # Enough for a few batches
        seed=42,
    )

    assert os.path.exists(best_model_path), "Best model file was not created."
    print(f"Training finished. Model saved at: {best_model_path}")

    # 6. Inference Verification
    print(
        "\n--- 6. Verifying Inference Pipeline (library.predict.generate_submission) ---"
    )
    submission_path = os.path.join(DEMO_DIR, "demo_submission.csv")

    generate_submission(
        model_path=best_model_path,
        output_file=submission_path,
        input_dir=INPUT_DIR,
        cache_dir=DEMO_DIR,
        batch_size=2,
        limit_size=4,  # Predict on 4 test subjects
        seed=42,
    )

    assert os.path.exists(submission_path), "Submission file was not created."

    # Check submission content
    sub_df = pd.read_csv(submission_path)
    print("Submission head:")
    print(sub_df.head())

    assert list(sub_df.columns) == [
        "BraTS21ID",
        "MGMT_value",
    ], "Incorrect submission columns."
    assert len(sub_df) == 4, f"Expected 4 predictions, got {len(sub_df)}"
    assert not sub_df["MGMT_value"].isnull().any(), "Submission contains NaNs."
    print("Inference verification passed.")

    print("\n=== ALL DEMO CHECKS PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_demo()
