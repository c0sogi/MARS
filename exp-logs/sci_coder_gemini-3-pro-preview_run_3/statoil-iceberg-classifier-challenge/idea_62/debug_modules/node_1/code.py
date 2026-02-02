import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed
from library.data_loader import get_fold_loaders, get_test_loader
from library.model import LSEIsomorphicCNN
from library.train import run_cross_validation
from library.predict import generate_submission


def main():
    print(">>> Initializing Demo Script...")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Isolation
    # -------------------------------------------------------------------------
    # We modify the global Config class to run a minimal version of the pipeline.

    # Set a separate working directory for the demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce computational load
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.N_FOLDS = 2  # Run only 2 folds (instead of 5)
    Config.BATCH_SIZE = 16  # Smaller batch size
    Config.DEBUG = True  # Enable debug mode (if applicable)

    # Initialize environment with new config
    Config.setup()
    set_seed(Config.SEED)

    print(
        f"Configured: Epochs={Config.NUM_EPOCHS}, Folds={Config.N_FOLDS}, Batch={Config.BATCH_SIZE}"
    )
    print(f"Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Verify Data Loading
    # -------------------------------------------------------------------------
    print("\n>>> Testing Data Loader...")

    # Fetch loaders for Fold 0.
    # load_cached_data=False ensures we test the raw JSON processing logic at least once.
    train_loader, val_loader = get_fold_loaders(fold_idx=0, load_cached_data=False)

    # Fetch a single batch
    images, angles, labels = next(iter(train_loader))

    # Verify Shapes
    print(f"  Batch Images Shape: {images.shape}")
    print(f"  Batch Angles Shape: {angles.shape}")
    print(f"  Batch Labels Shape: {labels.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        75,
        75,
    ), "Image tensor shape mismatch."
    assert angles.shape == (Config.BATCH_SIZE,), "Angle tensor shape mismatch."
    assert labels.shape == (Config.BATCH_SIZE,), "Label tensor shape mismatch."

    # Verify Data Integrity
    assert not torch.isnan(images).any(), "Images contain NaN values."
    assert not torch.isnan(
        angles
    ).any(), "Angles contain NaN values (Imputation failed)."

    print("  Data Loader verification passed.")

    # -------------------------------------------------------------------------
    # 3. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n>>> Testing Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = LSEIsomorphicCNN().to(device)

    # Move batch to device
    images_dev = images.to(device)
    angles_dev = angles.to(device)

    # Forward Pass
    logits = model(images_dev, angles_dev)

    print(f"  Output Logits Shape: {logits.shape}")

    # Verify Output
    assert logits.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch."
    assert not torch.isnan(logits).any(), "Model output contains NaNs."

    print("  Model architecture verification passed.")

    # -------------------------------------------------------------------------
    # 4. Execute Training Pipeline (Mini-Run)
    # -------------------------------------------------------------------------
    print("\n>>> Executing Cross-Validation (Reduced)...")

    # This runs the training loop using the modified Config (1 epoch, 2 folds)
    # It will save checkpoints to ./working/demo_execution/checkpoints
    run_cross_validation()

    # Verify Checkpoints
    for i in range(Config.N_FOLDS):
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"model_best_fold_{i}.pth")
        assert os.path.exists(ckpt_path), f"Checkpoint for fold {i} was not created."
        print(f"  Verified checkpoint: {os.path.basename(ckpt_path)}")

    # -------------------------------------------------------------------------
    # 5. Execute Inference and Submission
    # -------------------------------------------------------------------------
    print("\n>>> Generating Submission...")

    # This uses the checkpoints generated above to predict on test.json
    generate_submission()

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission Head:\n{df_sub.head(3)}")

    # Verify Content
    assert list(df_sub.columns) == [
        "id",
        "is_iceberg",
    ], "Submission columns are incorrect."
    assert len(df_sub) > 0, "Submission file is empty."

    # Verify Probabilities
    probs = df_sub["is_iceberg"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities are out of [0, 1] range."
    assert (
        probs.dtype == float or probs.dtype == np.float64
    ), "Probabilities are not floats."

    print("  Submission verification passed.")
    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    main()
