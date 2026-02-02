import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.dataset import prepare_folds, get_dataloaders, DogDataset
from library.model import DogModel
from library.train import train_fold
from library.inference import predict_and_submit


def main():
    print("Starting Dog Breed Classification Library Demo...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[1] Configuring environment for rapid demonstration...")

    # Patch Config for speed (Debug mode, fewer epochs, fewer folds)
    Config.DEBUG = True
    Config.EPOCHS_WARMUP = 1
    Config.EPOCHS_FINE_TUNE = 1
    Config.N_FOLDS = 2  # Run only 2 folds for demo
    Config.SOUP_TOP_K = 1  # Average top 1 (since we only run 1 epoch)
    Config.OUTPUT_DIR = "./working/demo_run"

    # Initialize environment
    Config.setup(seed=42)
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Preparation & Verification
    # ==========================================
    print("\n[2] Verifying Data Pipeline...")

    # Generate folds (force reload to test logic)
    df_folds, class_map = prepare_folds(load_cached_data=False)

    # Assertions
    assert "fold" in df_folds.columns, "Folds dataframe missing 'fold' column"
    assert len(class_map) == 120, f"Expected 120 classes, found {len(class_map)}"
    assert os.path.exists(
        os.path.join(Config.OUTPUT_DIR, "folds.parquet")
    ), "folds.parquet not saved"
    print("Data preparation successful. Folds created.")

    # Verify Dataloaders
    train_loader, val_loader = get_dataloaders(fold_idx=0, load_cached_data=True)

    # Fetch one batch
    images, labels = next(iter(train_loader))

    # Check shapes
    # Batch size is 64, Image size is 224, Channels 3
    expected_img_shape = (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    expected_lbl_shape = (Config.BATCH_SIZE,)

    assert (
        images.shape == expected_img_shape
    ), f"Image shape mismatch. Got {images.shape}"
    assert (
        labels.shape == expected_lbl_shape
    ), f"Label shape mismatch. Got {labels.shape}"
    print(f"DataLoader verified. Batch shape: {images.shape}")

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\n[3] Verifying Model Architecture...")

    model = DogModel(pretrained=True)
    model.to(device)

    # Forward pass check
    with torch.no_grad():
        images = images.to(device)
        logits = model(images)

    assert logits.shape == (
        Config.BATCH_SIZE,
        120,
    ), f"Model output shape incorrect. Got {logits.shape}"
    print("Model forward pass successful.")

    # Check freezing logic
    model.freeze_backbone()
    # Check if backbone is frozen (first layer)
    first_param = next(model.model.parameters())
    assert first_param.requires_grad is False, "Backbone should be frozen"
    print("Model freezing logic verified.")

    del model, images, labels, logits
    torch.cuda.empty_cache()

    # ==========================================
    # 4. Training Pipeline (Folds 0 & 1)
    # ==========================================
    print("\n[4] Executing Training Pipeline...")

    # Train Fold 0
    soup_path_0 = train_fold(fold_idx=0)
    assert os.path.exists(
        soup_path_0
    ), f"Soup model for Fold 0 not found at {soup_path_0}"

    # Train Fold 1
    soup_path_1 = train_fold(fold_idx=1)
    assert os.path.exists(
        soup_path_1
    ), f"Soup model for Fold 1 not found at {soup_path_1}"

    print("Training complete for Folds 0 and 1.")

    # ==========================================
    # 5. Inference & Submission
    # ==========================================
    print("\n[5] Executing Inference Pipeline...")

    # This function loads the soup models for folds 0 and 1 (since N_FOLDS=2),
    # runs TTA, averages predictions, and saves submission.csv
    predict_and_submit()

    sub_path = "submission/submission.csv"
    assert os.path.exists(sub_path), "Submission file was not generated."

    # Verify submission format
    df_sub = pd.read_csv(sub_path)

    # In DEBUG mode, test set is limited to 100 samples
    # Columns should be id + 120 breeds = 121 columns
    assert df_sub.shape == (
        100,
        121,
    ), f"Submission shape mismatch. Expected (100, 121), got {df_sub.shape}"
    assert "id" in df_sub.columns, "Submission missing 'id' column"

    # Check probabilities sum to approx 1
    row_sums = df_sub.iloc[:, 1:].sum(axis=1)
    # Allow small float tolerance
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1.0"

    print(f"Submission verified. Shape: {df_sub.shape}")
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
