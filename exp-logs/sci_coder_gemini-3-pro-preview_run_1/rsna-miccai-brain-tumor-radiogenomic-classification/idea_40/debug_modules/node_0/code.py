import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import library modules
from library.config import (
    WORKING_DIR,
    CACHE_DIR,
    MODELS_DIR,
    SUBMISSION_PATH,
    SEED,
    IMG_SIZE,
    DEVICE,
)
from library.utils import seed_everything
from library.preprocessing import process_dataset
from library.dataset import RASSEDataset, get_transforms
from library.model import ExpertNet
from library.trainer import run_training
from library.inference import run_inference


def run_demo():
    # 1. Setup
    seed_everything(SEED)
    print(f"Working Directory: {WORKING_DIR}")
    print(f"Device: {DEVICE}")

    # 2. Verify Preprocessing Logic (Small Subset)
    print("\n=== 1. Verifying Preprocessing Logic ===")
    train_meta_path = "./metadata/train_metadata.csv"
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(f"Metadata not found: {train_meta_path}")

    df_train = pd.read_csv(train_meta_path)
    # Take top 3 subjects for quick verification
    df_subset = df_train.head(3).copy()

    print(f"Processing subset of {len(df_subset)} subjects...")
    # Use a unique save_name to avoid conflict with full training cache
    subset_images, subset_ids, subset_labels = process_dataset(
        df_subset, load_cached_data=False, save_name="demo_subset"
    )

    # Assertions
    # Shape expected: (N, 3, IMG_SIZE, IMG_SIZE, 3)
    # N=3, Experts=3 (Lower, Center, Upper), H=224, W=224, C=3 (FLAIR, T1wCE, T2w)
    expected_shape = (3, 3, IMG_SIZE, IMG_SIZE, 3)
    assert (
        subset_images.shape == expected_shape
    ), f"Image shape mismatch. Expected {expected_shape}, got {subset_images.shape}"
    assert len(subset_ids) == 3
    assert len(subset_labels) == 3
    print("Preprocessing verification passed. Shapes are correct.")

    # 3. Verify Dataset and Model Forward Pass
    print("\n=== 2. Verifying Dataset & Model ===")
    # Create a dataset for the 'center' plane
    ds = RASSEDataset(
        subset_images,
        subset_ids,
        subset_labels,
        plane_name="center",
        transform=get_transforms(phase="train"),
    )

    dl = DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(dl))

    imgs = batch["image"].to(DEVICE)
    lbls = batch["label"].to(DEVICE)

    # Check tensor shape: (B, C, H, W) -> (2, 3, 224, 224)
    # Note: Albumentations ToTensorV2 produces (C, H, W)
    assert imgs.shape == (
        2,
        3,
        IMG_SIZE,
        IMG_SIZE,
    ), f"Batch image shape mismatch: {imgs.shape}"

    # Initialize Model
    model = ExpertNet().to(DEVICE)

    # Forward pass
    with torch.no_grad():
        logits = model(imgs)

    assert logits.shape == (2, 1), f"Logits shape mismatch: {logits.shape}"
    print("Dataset and Model forward pass verified.")

    # 4. Run Training (Full Pipeline for 1 Fold, 1 Plane, 1 Epoch)
    print("\n=== 3. Running Training Demo ===")
    # We train only 'center' plane, fold 0 to save time.
    # This will trigger processing of the FULL dataset (train+val) and caching it.
    # This might take a few minutes.

    plane = "center"
    fold = 0
    epochs = 1

    print(f"Training Plane: {plane}, Fold: {fold}, Epochs: {epochs}")
    best_auc = run_training(
        plane_name=plane,
        fold_idx=fold,
        epochs=epochs,
        batch_size=32,  # A100 can handle this easily
        patience=1,
    )

    # Check if model file was created
    model_path = os.path.join(MODELS_DIR, f"best_model_{plane}_fold{fold}.pth")
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    print(f"Training demo complete. Model saved. Best AUC: {best_auc}")

    # 5. Run Inference
    print("\n=== 4. Running Inference Demo ===")
    # This will process the test set and use the available model(s) to predict.
    # Since we only trained 'center' fold 0, it will warn about others but produce a result.

    run_inference(batch_size=32, load_cached_data=False)

    # Verify submission file
    assert os.path.exists(SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(SUBMISSION_PATH)
    print(f"Submission generated with {len(df_sub)} rows.")

    # Check format
    assert "BraTS21ID" in df_sub.columns
    assert "MGMT_value" in df_sub.columns
    assert not df_sub.isnull().values.any(), "Submission contains NaNs"

    print("Inference demo complete.")
    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    run_demo()
