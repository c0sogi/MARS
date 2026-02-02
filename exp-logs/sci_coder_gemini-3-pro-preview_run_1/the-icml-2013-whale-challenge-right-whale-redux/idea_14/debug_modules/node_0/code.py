import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, mixup_data, load_checkpoint
from library.data import get_dataloaders
from library.model import AdaptiveResNetCRNN
from library.trainer import Trainer
from library.layers import CoordinateAttention

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Configuration Override for Speed and Demo Isolation
    print("\n[1] Configuring environment...")
    Config.WORKING_DIR = "./working/demo_execution"
    Config.BATCH_SIZE = 8  # Small batch size for debug
    Config.EPOCHS = 2  # Minimal epochs
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small debug run
    Config.DEBUG = True  # Use debug flag logic if applicable

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds
    set_seed(42)
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # 2. Data Pipeline Verification
    print("\n[2] Initializing DataLoaders (Debug Mode)...")
    # debug=True loads a tiny subset (100 train, 50 val, 50 test)
    # load_cached_data=False forces processing to demonstrate the raw audio -> spec pipeline
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, load_cached_data=False
    )

    print("Verifying batch shapes...")
    # Fetch one batch
    data_iter = iter(train_loader)
    images, targets = next(data_iter)

    # Expected: (Batch, 1, F, T) -> (8, 1, 128, 4000) roughly, T depends on padding/duration
    print(f"Input Batch Shape: {images.shape}")
    print(f"Target Batch Shape: {targets.shape}")

    assert images.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert images.shape[1] == 1, "Channel dimension mismatch (should be 1)"
    assert targets.shape[0] == Config.BATCH_SIZE, "Target batch size mismatch"

    print("Verifying Mixup augmentation...")
    mixed_x, y_a, y_b, lam = mixup_data(images, targets, alpha=0.4)
    assert mixed_x.shape == images.shape, "Mixup output shape mismatch"
    assert 0 <= lam <= 1, "Mixup lambda out of range"
    print("Data pipeline verified.")

    # 3. Model Architecture Verification
    print("\n[3] Initializing Model...")
    model = AdaptiveResNetCRNN().to(Config.DEVICE)

    # Test Forward Pass
    print("Testing forward pass...")
    images = images.to(Config.DEVICE)
    with torch.no_grad():
        output = model(images)

    # Output should be (Batch, 1)
    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"

    # Verify specific layer logic (Coordinate Attention)
    print("Verifying Coordinate Attention Block...")
    ca_block = CoordinateAttention(in_channels=16, reduction=4).to(Config.DEVICE)
    dummy_input = torch.randn(4, 16, 32, 32).to(Config.DEVICE)
    ca_out = ca_block(dummy_input)
    assert (
        ca_out.shape == dummy_input.shape
    ), "Coordinate Attention changed tensor shape unexpectedly"
    print("Model architecture verified.")

    # 4. Training Loop Execution
    print("\n[4] Starting Training Loop...")
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=1, verbose=False
    )

    trainer = Trainer(model, optimizer, scheduler, Config.DEVICE)

    save_name = "demo_best_model.pth"
    best_auc = trainer.train_model(
        train_loader, val_loader, num_epochs=Config.EPOCHS, save_name=save_name
    )

    print(f"Training complete. Best Validation AUC: {best_auc:.4f}")
    assert 0.0 <= best_auc <= 1.0, "AUC score out of valid range"
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, save_name)
    ), "Checkpoint file not created"

    # 5. Inference & Submission
    print("\n[5] Running Inference on Test Set...")

    # Load best model
    best_model = AdaptiveResNetCRNN().to(Config.DEVICE)
    load_checkpoint(
        best_model,
        filename=save_name,
        load_dir=Config.WORKING_DIR,
        device=Config.DEVICE,
    )
    best_model.eval()

    predictions = []
    clips = []

    # We need the clip IDs. The dataset in the loader returns (data, dummy_label).
    # The loader doesn't yield IDs directly. We can access the dataset's internal ID list if needed,
    # or rely on the order if shuffle=False.
    # The provided WhaleDataset doesn't return IDs in __getitem__.
    # However, `load_or_create_cache` saved `_ids.npy`.
    # For this demo, we will rely on the order of the test_loader which is shuffle=False.
    # We need to load the test IDs corresponding to the debug subset.

    # Reload test IDs from the cache file created by get_dataloaders
    test_ids_path = os.path.join(Config.WORKING_DIR, "test_debug_ids.npy")
    test_ids = np.load(test_ids_path)

    print(f"Loaded {len(test_ids)} test IDs.")

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(Config.DEVICE)
            output = best_model(data)
            output = output.squeeze(1)
            probs = torch.sigmoid(output).cpu().numpy()
            predictions.extend(probs)

    # Truncate or pad predictions to match IDs (in case of drop_last or batch size mismatch)
    # Since drop_last=False for test, lengths should match exactly.
    assert len(predictions) == len(
        test_ids
    ), f"Mismatch: {len(predictions)} preds vs {len(test_ids)} IDs"

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"clip": test_ids, "probability": predictions})

    print("Sample Predictions:")
    print(df_sub.head())

    # Save Submission
    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    assert os.path.exists(submission_path), "Submission file was not created."

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
