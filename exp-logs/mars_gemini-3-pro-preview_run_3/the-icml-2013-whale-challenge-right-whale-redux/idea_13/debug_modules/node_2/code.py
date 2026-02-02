import os
import sys
import shutil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.dataset import get_dataloaders
from library.model import WhaleModel
from library.trainer import Trainer, generate_submission, get_pos_weight

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Right Whale Detection Pipeline Demo ===")

    # 1. Configure for Speed and Demo
    print("\n[1] Configuring environment...")
    Config.PROJECT_NAME = "demo_execution"
    Config.WORKING_DIR = os.path.join("./working", Config.PROJECT_NAME)
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINTS_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Enable Debug mode to use a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 files per split
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize directories
    Config.init_directories()
    seed_everything(Config.SEED)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")

    # 2. Data Loading
    print("\n[2] Loading Data...")
    # Force reload to ensure we use the debug subset
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, config=Config
    )

    # Verify DataLoaders
    print("    Verifying DataLoader shapes...")
    try:
        train_batch, train_targets = next(iter(train_loader))
        val_batch, val_targets = next(iter(val_loader))
        test_batch, test_names = next(iter(test_loader))

        # Check Image Shape: (Batch, Channels, Freq, Time)
        # N_MELS=384, Duration=2s -> Time ~ 200 frames (depends on hop length)
        # Expected: (8, 1, 384, ~200)
        assert train_batch.ndim == 4, f"Expected 4D tensor, got {train_batch.ndim}"
        assert (
            train_batch.shape[1] == 1
        ), f"Expected 1 channel, got {train_batch.shape[1]}"
        assert (
            train_batch.shape[2] == Config.N_MELS
        ), f"Expected {Config.N_MELS} mels, got {train_batch.shape[2]}"

        # Check Targets
        assert train_targets.shape[0] == Config.BATCH_SIZE, "Target batch size mismatch"

        print("    Data loading verification passed.")
    except StopIteration:
        raise ValueError(
            "DataLoaders are empty. Check input data or debug sample size."
        )

    # 3. Model Initialization
    print("\n[3] Initializing Model...")
    model = WhaleModel(
        config=Config, pretrained=False
    )  # False for speed, we don't need weights download
    model = model.to(Config.DEVICE)

    # Verify Forward Pass
    dummy_input = torch.randn(2, 1, Config.N_MELS, 201).to(Config.DEVICE)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"
    print("    Model forward pass verification passed.")

    # 4. Training Loop
    print("\n[4] Running Training Loop...")

    # Setup Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # Calculate pos_weight for imbalance handling
    # Accessing the underlying dataset from the loader
    pos_weight = get_pos_weight(train_loader.dataset)
    print(f"    Calculated positive class weight: {pos_weight:.4f}")

    trainer = Trainer(
        model, optimizer, scheduler, device=Config.DEVICE, pos_weight=pos_weight
    )

    # Run Fit
    best_auc = trainer.fit(
        train_loader,
        val_loader,
        epochs=Config.EPOCHS,
        patience=1,
        use_mixup=True,
        save_name="demo_best.pth",
    )

    print(f"    Training complete. Best AUC: {best_auc:.4f}")

    # Verify Checkpoint
    checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, "demo_best.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created."
    print("    Checkpoint verification passed.")

    # 5. Inference
    print("\n[5] Running Inference...")
    # Load best model weights
    epoch, metric = load_checkpoint(
        model, filename="demo_best.pth", device=Config.DEVICE
    )
    print(f"    Loaded checkpoint from epoch {epoch} with metric {metric:.4f}")

    predictions = trainer.predict(test_loader)

    assert len(predictions) > 0, "No predictions generated."
    sample_key = list(predictions.keys())[0]
    assert isinstance(
        predictions[sample_key], (float, np.float32, np.float64)
    ), "Prediction is not a float."
    print(f"    Generated {len(predictions)} predictions.")

    # 6. Submission Generation
    print("\n[6] Generating Submission...")
    generate_submission(predictions, Config.SUBMISSION_PATH)
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    # Check content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert df_sub.shape[1] == 2, "Submission should have 2 columns"
    assert (
        "clip" in df_sub.columns and "probability" in df_sub.columns
    ), "Submission headers mismatch"
    print("    Submission file verification passed.")

    # 7. Pseudo-Labeling (Self-Training) Demo
    print("\n[7] Demonstrating Pseudo-Labeling Integration...")

    # We use the predictions from step 5 as pseudo-labels
    # In a real scenario, we would filter these by confidence, but for demo we pass all
    pseudo_labels_map = predictions

    # Reload dataloaders with pseudo-labels
    # Note: We use cached data=True here to speed up, as the base arrays are already cached in step 2
    train_loader_student, _, _ = get_dataloaders(
        load_cached_data=True, config=Config, pseudo_labels=pseudo_labels_map
    )

    # Verify size increase
    original_size = len(train_loader.dataset)
    new_size = len(train_loader_student.dataset)

    print(f"    Original Train Size: {original_size}")
    print(f"    Student Train Size: {new_size}")

    # Since we are in debug mode, the intersection of test clips and pseudo labels
    # should result in an increase equal to the number of test samples loaded
    if new_size > original_size:
        print("    Pseudo-labeling verification passed: Dataset size increased.")
    else:
        # This might happen if debug sampling caused disjoint sets between metadata and actual files processed
        # or if no test files matched the debug criteria.
        print(
            "    Warning: Dataset size did not increase. This is acceptable if debug sets are disjoint."
        )

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
