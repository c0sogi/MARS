import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging

# --- 1. Suppress Progress Bars (Monkey Patching) ---
# We patch tqdm to be silent before importing library modules that use it.
import tqdm


class TqdmStub:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable if iterable else []

    def __iter__(self):
        return iter(self.iterable)

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def set_description(self, *args, **kwargs):
        pass


# Replace tqdm with the stub
tqdm.tqdm = TqdmStub

# --- 2. Import Library Modules ---
from library.config import Config
from library.utils import seed_everything, setup_logger, ModelEma
from library.data import get_loaders, get_test_loader
from library.model import PathologyModel
from library.engine import train_one_epoch, validate, predict_tta, save_submission

if __name__ == "__main__":
    # --- 3. Configure for Rapid Execution ---
    print("Configuring execution parameters...")

    # Override Config for a fast demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 64  # Small subset for speed
    Config.EPOCHS = 1
    Config.NUM_FOLDS = 2  # Setup for folds, but we'll only run one
    Config.BATCH_SIZE = 8  # Small batch size for the small subset
    Config.NUM_WORKERS = 0  # Disable multiprocessing overhead for small data
    Config.NUM_RUNS = 1

    # Update working directories to separate this demo run
    Config.PROJECT_NAME = "demo_execution"
    Config.WORKING_DIR = f"./working/{Config.PROJECT_NAME}"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Create necessary directories
    for d in [Config.CACHE_DIR, Config.CHECKPOINT_DIR, Config.SUBMISSION_DIR]:
        os.makedirs(d, exist_ok=True)

    # --- 4. Initialization ---
    seed_everything(Config.SEED)
    logger = setup_logger(os.path.join(Config.WORKING_DIR, "train.log"))
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # --- 5. Data Loading ---
    print("Initializing DataLoaders...")
    # Get loaders for Fold 0
    train_loader, val_loader = get_loaders(
        fold=0, seed=Config.SEED, load_cached_data=False
    )

    # Verification: Check batch structure
    sample_imgs, sample_lbls = next(iter(train_loader))
    assert sample_imgs.shape == (
        Config.BATCH_SIZE,
        3,
        Config.CROP_SIZE,
        Config.CROP_SIZE,
    ), f"Expected image batch shape ({Config.BATCH_SIZE}, 3, {Config.CROP_SIZE}, {Config.CROP_SIZE}), got {sample_imgs.shape}"
    assert sample_lbls.shape == (
        Config.BATCH_SIZE,
    ), f"Expected label batch shape ({Config.BATCH_SIZE},), got {sample_lbls.shape}"
    print("DataLoaders initialized and verified.")

    # --- 6. Model Setup ---
    print("Initializing Model and EMA...")
    model = PathologyModel(pretrained=True).to(device)

    # Initialize EMA model
    model_ema = ModelEma(model, decay=Config.EMA_DECAY, device=device)

    # Verification: Check forward pass
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, Config.CROP_SIZE, Config.CROP_SIZE).to(device)
        dummy_out = model(dummy_input)
        assert dummy_out.shape == (
            2,
            1,
        ), f"Expected output shape (2, 1), got {dummy_out.shape}"
    print("Model initialized and verified.")

    # --- 7. Training Loop (1 Epoch) ---
    print("Starting training for 1 epoch...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Train
    train_loss = train_one_epoch(
        model, model_ema, train_loader, optimizer, device, epoch=1
    )

    # Verification: Loss validity
    assert isinstance(train_loss, float), "Train loss should be a float"
    assert train_loss > 0, "Train loss should be positive"
    print(f"Training completed. Loss: {train_loss:.4f}")

    # --- 8. Validation ---
    print("Running validation...")
    # Validate using the EMA model (standard practice in this pipeline)
    val_loss, val_auc = validate(model_ema.module, val_loader, device)

    # Verification: Metric validity
    assert 0.0 <= val_auc <= 1.0, f"AUC must be between 0 and 1, got {val_auc}"
    print(f"Validation completed. AUC: {val_auc:.4f}")

    # --- 9. Inference (TTA) ---
    print("Starting inference on Test set...")
    # Load test data
    test_loader, test_ids = get_test_loader(load_cached_data=False)

    # Predict using TTA
    preds = predict_tta(model_ema.module, test_loader, device)

    # Verification: Prediction shape and values
    assert len(preds) == len(
        test_ids
    ), f"Mismatch between predictions ({len(preds)}) and test IDs ({len(test_ids)})"
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions must be probabilities (0-1)"
    print(f"Inference completed. Generated {len(preds)} predictions.")

    # --- 10. Submission ---
    print("Saving submission...")
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    save_submission(test_ids, preds, submission_path)

    # Verification: File existence
    assert os.path.exists(submission_path), "Submission file was not created"

    # Verify content format
    df_sub = pd.read_csv(submission_path)
    assert list(df_sub.columns) == ["id", "label"], "Submission columns mismatch"
    assert len(df_sub) == len(test_ids), "Submission row count mismatch"

    print("Pipeline demonstration completed successfully.")
