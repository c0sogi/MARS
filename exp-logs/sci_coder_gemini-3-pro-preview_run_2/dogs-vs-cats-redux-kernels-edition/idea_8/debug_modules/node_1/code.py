import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
import glob

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, AverageMeter, get_score
from library.dataset import (
    PetDataset,
    get_transforms,
    prepare_folds,
    get_train_val_loaders,
    get_test_loader,
)
from library.models import get_model
from library.train import train_one_epoch, validate_one_epoch, run_fold
from library.inference import run_inference


def run_demo():
    print("=== Starting Library Usage Demonstration ===")

    # ------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Testing
    # ------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Redirect outputs to a demo directory to avoid messing up real experiment paths
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Modify Config attributes directly
    Config.WORKING_DIR = DEMO_DIR
    Config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

    # Create directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set Debug mode to True to use small data subsets (500 samples)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Even smaller for this demo script

    # Reduce training complexity
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.N_FOLDS = 2  # We will only run fold 0, but setup for 2
    Config.MODEL_ARCHS = ["resnet18"]  # Use a lightweight model for demo
    Config.NUM_WORKERS = 2  # Reduce workers for small demo

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Model Architecture: {Config.MODEL_ARCHS}")

    # ------------------------------------------------------------------------
    # 2. Verify Utilities
    # ------------------------------------------------------------------------
    print("\n[2] Verifying Utilities...")

    # Test Seeding
    seed_everything(Config.SEED)
    print("    Seed set successfully.")

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(10, n=2)
    meter.update(20, n=1)
    # Total val = 10*2 + 20*1 = 40. Total count = 3. Avg = 13.333
    assert abs(meter.avg - 13.3333) < 1e-4, "AverageMeter calculation incorrect"
    print("    AverageMeter logic verified.")

    # ------------------------------------------------------------------------
    # 3. Verify Data Pipeline
    # ------------------------------------------------------------------------
    print("\n[3] Verifying Data Pipeline...")

    # Test Fold Preparation
    # This should create folds.parquet in the demo working dir
    df_folds = prepare_folds(load_cached_data=False)
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "folds.parquet")
    ), "folds.parquet not created"
    assert "fold" in df_folds.columns, "Fold column missing in dataframe"
    print(f"    Folds prepared. Total samples in metadata: {len(df_folds)}")

    # Test Dataset Class directly
    # Get a single sample
    train_transforms = get_transforms(mode="train")
    dataset = PetDataset(df_folds.head(10), transforms=train_transforms, mode="train")
    img, label = dataset[0]

    # Check tensor shapes and types
    assert isinstance(img, torch.Tensor), "Image is not a tensor"
    assert img.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Incorrect image shape: {img.shape}"
    assert isinstance(label, torch.Tensor), "Label is not a tensor"
    print("    PetDataset __getitem__ verified.")

    # Test DataLoaders
    train_loader, val_loader = get_train_val_loaders(fold_idx=0, load_cached_data=True)

    # Fetch one batch
    images, labels = next(iter(train_loader))
    assert images.size(0) == Config.BATCH_SIZE, "Train loader batch size mismatch"
    assert labels.size(0) == Config.BATCH_SIZE, "Train loader label size mismatch"
    print(f"    DataLoaders initialized. Train batch shape: {images.shape}")

    # ------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # ------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    device = Config.DEVICE
    model = get_model(
        "resnet18", pretrained=False
    )  # No need to download weights for logic check
    model.to(device)

    # Forward pass check
    dummy_input = torch.randn(4, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        4,
        1,
    ), f"Model output shape mismatch. Expected (4, 1), got {output.shape}"
    print("    Model instantiation and forward pass verified.")

    # ------------------------------------------------------------------------
    # 5. Verify Training Loop (Single Fold)
    # ------------------------------------------------------------------------
    print("\n[5] Simulating Training Run (Fold 0)...")

    # We run the actual run_fold function.
    # Because DEBUG=True and EPOCHS=1, this should be very fast.
    best_loss = run_fold("resnet18", fold_idx=0)

    # Verify checkpoint creation
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "resnet18_fold_0.pth")
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"
    assert isinstance(best_loss, float), "run_fold did not return a float loss"
    print(
        f"    Training simulation complete. Checkpoint saved. Best Val Loss: {best_loss:.4f}"
    )

    # ------------------------------------------------------------------------
    # 6. Verify Inference Pipeline
    # ------------------------------------------------------------------------
    print("\n[6] Verifying Inference Pipeline...")

    # run_inference iterates over Config.MODEL_ARCHS and Config.N_FOLDS.
    # We set N_FOLDS=2 but only trained Fold 0.
    # run_inference logic skips missing checkpoints, so it should handle Fold 1 missing gracefully.

    run_inference()

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not generated"

    # Validate submission format
    sub_df = pd.read_csv(submission_path)
    assert list(sub_df.columns) == ["id", "label"], "Submission columns mismatch"
    assert len(sub_df) > 0, "Submission file is empty"

    # Check if IDs match the test set (or debug subset of test set)
    test_loader = get_test_loader()
    # In debug mode, test loader has Config.DEBUG_SAMPLE_SIZE samples
    # The submission should match this count
    expected_len = min(
        len(pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))),
        Config.DEBUG_SAMPLE_SIZE,
    )
    assert (
        len(sub_df) == expected_len
    ), f"Submission length mismatch. Expected {expected_len}, got {len(sub_df)}"

    print(f"    Inference successful. Submission shape: {sub_df.shape}")
    print("    Sample predictions:")
    print(sub_df.head())

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
