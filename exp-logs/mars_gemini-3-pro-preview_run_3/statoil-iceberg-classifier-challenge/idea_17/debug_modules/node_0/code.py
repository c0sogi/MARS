import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import library components
from library.config import Config
from library.utils import set_seed, save_submission
from library.data_loader import process_and_cache_data, get_dataloaders
from library.model import WideSEResNet
from library.engine import train_fold, inference


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # 1. Configure for Speed and Isolation
    # We modify the Config class directly to ensure the demo runs quickly and
    # uses a separate workspace.
    print("\n[Step 1] Configuring environment...")

    Config.WORK_DIR = "./working/demo_usage"
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORK_DIR, "checkpoints")
    Config.SUBMISSION_DIR = (
        Config.WORK_DIR
    )  # Save submission in root of work dir for demo
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Enable Debug mode to use a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Enough for a few batches

    # Training Hyperparameters for fast execution
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_FOLDS = 2  # We will only run fold 0

    # Setup directories
    Config.setup()
    set_seed(Config.SEED)

    print(f"Working Directory: {Config.WORK_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")

    # 2. Data Processing
    print("\n[Step 2] Processing and Caching Data...")
    # This reads the json files, processes bands, imputes angles, and saves .npy files
    X_train, angles_train, y_train, X_test, angles_test, ids_test = (
        process_and_cache_data(load_cached_data=False)
    )

    # Verification
    assert os.path.exists(
        os.path.join(Config.CACHE_DIR, "X_train.npy")
    ), "X_train cache missing"
    assert len(X_train) > 0, "X_train is empty"
    print(f"Data processed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 3. Data Loading
    print("\n[Step 3] Creating DataLoaders...")
    # Get loaders for Fold 0
    train_loader, val_loader, test_loader = get_dataloaders(
        fold_idx=0, load_cached_data=True
    )

    # Verify Batch Shapes
    images, angles, labels = next(iter(train_loader))
    print(
        f"Batch Shapes -> Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    # Assertions
    # Images: (Batch, 3, 75, 75)
    assert images.dim() == 4
    assert images.size(1) == 3
    assert images.size(2) == 75
    assert images.size(3) == 75
    # Angles: (Batch)
    assert angles.dim() == 1
    assert angles.size(0) == images.size(0)
    # Labels: (Batch)
    assert labels.dim() == 1
    assert labels.size(0) == images.size(0)

    # 4. Model Initialization and Forward Pass
    print("\n[Step 4] Initializing Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WideSEResNet().to(device)

    # Move batch to device
    images = images.to(device)
    angles = angles.to(device)

    # Forward pass
    print("Running dummy forward pass...")
    model.eval()
    with torch.no_grad():
        logits = model(images, angles)

    print(f"Output Logits Shape: {logits.shape}")

    # Assertions
    assert logits.dim() == 1, "Model output should be 1D tensor (Batch Size)"
    assert logits.size(0) == images.size(0), "Output batch size mismatch"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"

    # 5. Training Engine (Single Fold)
    print("\n[Step 5] Running Training for Fold 0 (1 Epoch)...")
    # This uses the engine.train_fold function
    best_val_loss = train_fold(fold_idx=0)

    print(f"Training completed. Best Val Loss: {best_val_loss}")

    # Verify Checkpoint
    checkpoint_path = Config.get_checkpoint_path(0)
    assert os.path.exists(
        checkpoint_path
    ), f"Checkpoint file not found at {checkpoint_path}"
    print("Checkpoint verified.")

    # 6. Inference
    print("\n[Step 6] Running Inference...")
    # We use the inference function from engine.py
    # Note: test_loader was created in Step 3
    probs, ids = inference(fold_idx=0, test_loader=test_loader)

    print(f"Inference complete. Predictions shape: {probs.shape}")

    # Assertions
    assert len(probs) == len(ids), "Mismatch between probabilities and IDs count"
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of [0, 1] range"

    # 7. Submission
    print("\n[Step 7] Saving Submission...")
    save_submission(ids, probs, output_path=Config.SUBMISSION_FILE)

    # Verify File
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not created"

    # Check content format
    df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission DataFrame Head:\n{df.head()}")

    assert list(df.columns) == [
        "id",
        "is_iceberg",
    ], "Incorrect columns in submission file"
    assert len(df) == len(ids), "Row count mismatch in submission file"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Suppress LightGBM warnings if any (though we aren't using it here, good practice based on prompt)
    os.environ["LGBM_SILENT"] = "1"

    try:
        run_demo()
    except AssertionError as e:
        print(f"\n[ERROR] Verification Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
