import os
import sys
import importlib
import torch
import numpy as np
import pandas as pd

# =============================================================================
# 1. CONFIGURATION OVERRIDE & IMPORTS
# =============================================================================
# We import config first and modify it to run a fast demonstration.
import library.config

print("Configuring environment for demonstration...")
# Enable DEBUG mode to use a small subset of data (defined by DEBUG_SAMPLE_SIZE)
library.config.DEBUG = True
library.config.DEBUG_SAMPLE_SIZE = 32  # Small sample for quick execution
library.config.NUM_WORKERS = (
    0  # Use 0 workers to avoid multiprocessing overhead in demo
)
library.config.EPOCHS = 1  # Run only 1 epoch
library.config.BATCH_SIZE = 4  # Small batch size

# We must reload library.data and library.train because they import variables
# from library.config at the top level. Reloading ensures they see the updated values.
import library.data
import library.model
import library.train

importlib.reload(library.data)
importlib.reload(library.model)
importlib.reload(library.train)

from library.utils import seed_everything, probabilistic_f1
from library.data import (
    get_dataloaders,
    MammographyDataset,
    get_transforms,
    process_metadata,
)
from library.model import SpatialSiameseModel
from library.train import run_training

# Set reproducibility
seed_everything(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {DEVICE}")

if __name__ == "__main__":

    # =============================================================================
    # 2. METRIC VERIFICATION
    # =============================================================================
    print("\n==== Testing Probabilistic F1 Score ====")
    # Manual Test Case
    # y_true: [1, 0, 1]
    # y_pred: [0.9, 0.1, 0.8]
    # pTP = 0.9*1 + 0.1*0 + 0.8*1 = 1.7
    # pFP = 0.9*0 + 0.1*1 + 0.8*0 = 0.1
    # TP_FN (Total Positives) = 2
    # pPrecision = 1.7 / (1.7 + 0.1) = 0.9444...
    # pRecall = 1.7 / 2 = 0.85
    # pF1 = 2 * (0.9444 * 0.85) / (0.9444 + 0.85) = 1.6055 / 1.7944 = 0.8947...

    y_true_test = np.array([1, 0, 1])
    y_pred_test = np.array([0.9, 0.1, 0.8])

    pf1_score = probabilistic_f1(y_true_test, y_pred_test)
    print(f"Calculated pF1: {pf1_score:.4f}")

    expected_pf1 = 0.8947
    assert np.isclose(
        pf1_score, expected_pf1, atol=1e-3
    ), f"pF1 calculation mismatch! Expected ~{expected_pf1}, got {pf1_score}"
    print("Metric verification passed.")

    # =============================================================================
    # 3. DATA PIPELINE VERIFICATION
    # =============================================================================
    print("\n==== Testing Data Pipeline ====")

    # Generate/Load DataLoaders (this uses the DEBUG subset)
    # Note: load_cached_data=False forces reprocessing to ensure logic runs
    print("Initializing DataLoaders (Debug Mode)...")
    try:
        train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)
    except FileNotFoundError as e:
        print(f"Skipping Data Pipeline test due to missing files in environment: {e}")
        # Create dummy data for model test if real data fails (fallback for robustness)
        train_loader = []

    if len(train_loader) > 0:
        # Fetch one batch
        batch = next(iter(train_loader))
        images = batch["image"]
        contralateral = batch["contralateral"]
        labels = batch["label"]

        print(f"Batch keys: {batch.keys()}")
        print(f"Image Shape: {images.shape}")
        print(f"Contralateral Shape: {contralateral.shape}")
        print(f"Labels Shape: {labels.shape}")

        # Assertions
        # Expected shape: [BATCH_SIZE, 3, 768, 768] (3 channels: Image, Age, Implant)
        expected_shape = (
            library.config.BATCH_SIZE,
            3,
            library.config.IMG_HEIGHT,
            library.config.IMG_WIDTH,
        )
        assert (
            images.shape == expected_shape
        ), f"Image shape mismatch. Expected {expected_shape}, got {images.shape}"
        assert (
            contralateral.shape == expected_shape
        ), f"Contralateral shape mismatch. Expected {expected_shape}, got {contralateral.shape}"
        assert labels.shape == (
            library.config.BATCH_SIZE,
        ), f"Label shape mismatch. Expected ({library.config.BATCH_SIZE},), got {labels.shape}"

        print("Data pipeline verification passed.")
    else:
        print("Warning: DataLoader is empty. Check input data availability.")

    # =============================================================================
    # 4. MODEL ARCHITECTURE VERIFICATION
    # =============================================================================
    print("\n==== Testing Model Architecture ====")

    model = SpatialSiameseModel()
    model.to(DEVICE)
    model.eval()

    # Create dummy input if loader failed, else use real batch
    if "images" not in locals():
        dummy_shape = (
            library.config.BATCH_SIZE,
            3,
            library.config.IMG_HEIGHT,
            library.config.IMG_WIDTH,
        )
        images = torch.randn(dummy_shape).to(DEVICE)
        contralateral = torch.randn(dummy_shape).to(DEVICE)
    else:
        images = images.to(DEVICE)
        contralateral = contralateral.to(DEVICE)

    with torch.no_grad():
        logits = model(images, contralateral)

    print(f"Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        library.config.BATCH_SIZE,
        1,
    ), f"Output shape mismatch. Expected {(library.config.BATCH_SIZE, 1)}, got {logits.shape}"

    print("Model architecture verification passed.")

    # =============================================================================
    # 5. TRAINING LOOP INTEGRATION
    # =============================================================================
    print("\n==== Testing Training Loop Integration ====")

    # Run a short training session
    # This calls library.train.run_training which uses the overridden EPOCHS=1 and DEBUG loaders
    try:
        run_training(load_cached_data=True, epochs=1, patience=1)

        # Verify output file creation
        if os.path.exists(library.config.MODEL_SAVE_PATH):
            print(
                f"Training successful. Model saved to {library.config.MODEL_SAVE_PATH}"
            )
        else:
            # Note: Model is only saved if validation improves.
            # With 1 epoch and random init, it might not save if val score is 0.
            # This is acceptable behavior.
            print(
                "Training routine finished (Model might not be saved if metric didn't improve)."
            )

    except Exception as e:
        print(f"Training loop failed with error: {e}")
        raise e

    print("\nAll demonstrations completed successfully.")
