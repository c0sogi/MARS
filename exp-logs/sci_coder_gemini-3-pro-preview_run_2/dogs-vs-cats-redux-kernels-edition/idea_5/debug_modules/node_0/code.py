import os
import sys
import pandas as pd
import torch
import numpy as np
from library.config import Config
from library.utils import set_seed
from library.data import get_train_val_loaders, get_test_loader
from library.model import get_model
from library.trainer import train_fold
from library.inference import run_inference


def main():
    print("=== Starting Demonstration Script ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Demo
    # ---------------------------------------------------------
    print("Configuring parameters for rapid execution...")
    Config.epochs = 1
    Config.n_folds = 1  # Only process the first fold
    Config.batch_size = 8  # Small batch size
    Config.num_workers = 0  # Disable multiprocessing for small data to avoid overhead

    # Ensure reproducibility
    set_seed(Config.seed)

    # ---------------------------------------------------------
    # 2. Data Preparation (Subsetting)
    # ---------------------------------------------------------
    print("Creating data subsets...")

    # Load original metadata
    train_meta_full = pd.read_csv(Config.train_metadata)
    val_meta_full = pd.read_csv(Config.val_metadata)
    test_meta_full = pd.read_csv(Config.test_metadata)

    # Create a tiny training/validation set (simulating the folded structure)
    # train_fold(0) expects:
    #   - Validation data where fold == 0
    #   - Training data where fold != 0

    # Select 32 samples for training (fold=1) and 16 for validation (fold=0)
    subset_train = train_meta_full.sample(n=32, random_state=Config.seed).copy()
    subset_train["fold"] = 1

    subset_val = val_meta_full.sample(n=16, random_state=Config.seed).copy()
    subset_val["fold"] = 0

    # Combine and save to the location expected by get_folded_data
    folds_df = pd.concat([subset_train, subset_val], ignore_index=True)
    folds_cache_path = os.path.join(Config.working_dir, "folds.parquet")
    folds_df.to_parquet(folds_cache_path)
    print(f"Subset folds saved to: {folds_cache_path}")

    # Create a tiny test set
    subset_test = test_meta_full.iloc[:10].copy()
    subset_test_path = os.path.join(Config.working_dir, "test_subset.csv")
    subset_test.to_csv(subset_test_path, index=False)

    # Point Config to the new test metadata
    Config.test_metadata = subset_test_path
    print(f"Subset test metadata saved to: {subset_test_path}")

    # ---------------------------------------------------------
    # 3. Verify Data Loading
    # ---------------------------------------------------------
    print("Verifying DataLoaders...")
    train_loader, val_loader = get_train_val_loaders(fold_idx=0)

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")

    assert len(train_loader) > 0, "Train loader should not be empty."
    assert len(val_loader) > 0, "Validation loader should not be empty."

    # Check batch shapes
    images, targets = next(iter(train_loader))
    print(f"  Batch Image Shape: {images.shape}")
    print(f"  Batch Target Shape: {targets.shape}")

    assert images.shape == (
        Config.batch_size,
        3,
        Config.image_size,
        Config.image_size,
    ), "Incorrect image tensor shape."
    assert targets.shape == (Config.batch_size,), "Incorrect target tensor shape."

    # ---------------------------------------------------------
    # 4. Model Initialization
    # ---------------------------------------------------------
    print("Initializing Model...")
    model = get_model(pretrained=True)

    # Verify model output shape
    dummy_input = torch.randn(2, 3, Config.image_size, Config.image_size).to(
        Config.device
    )
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    # Expecting [Batch, 1] for binary classification
    print(f"  Model Output Shape: {output.shape}")
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"

    # ---------------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------------
    print("Starting Training (Fold 0)...")
    best_loss = train_fold(0, train_loader, val_loader, model)
    print(f"Training completed. Best Validation Loss: {best_loss:.4f}")

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.checkpoint_dir, "fold_0.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print("Checkpoint verification passed.")

    # ---------------------------------------------------------
    # 6. Inference Pipeline
    # ---------------------------------------------------------
    print("Running Inference...")
    run_inference()

    # Verify Submission
    submission_path = Config.submission_path
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    sub_df = pd.read_csv(submission_path)
    print("Submission Head:")
    print(sub_df.head())

    # Validation checks on submission
    assert len(sub_df) == 10, f"Expected 10 predictions, got {len(sub_df)}"
    assert list(sub_df.columns) == ["id", "label"], "Submission columns mismatch."
    assert sub_df["label"].between(0, 1).all(), "Probabilities out of range [0, 1]."

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
