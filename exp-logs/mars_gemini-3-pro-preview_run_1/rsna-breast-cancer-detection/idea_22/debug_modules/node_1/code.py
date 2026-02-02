import os
import pandas as pd
import numpy as np
import torch
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, probabilistic_f1
from library.data import get_dataloaders
from library.model import SiameseEfficientNet
from library.train import run_training, generate_submission_file


def test_metric():
    """Verifies the Probabilistic F1 Score calculation."""
    print("\n[1/5] Testing Metric: Probabilistic F1...")

    # Case 1: Perfect prediction
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1.0, 0.0, 1.0, 0.0])
    score = probabilistic_f1(y_true, y_pred)
    assert abs(score - 1.0) < 1e-6, f"Expected 1.0, got {score}"

    # Case 2: Complete failure
    y_true = np.array([1, 1])
    y_pred = np.array([0.0, 0.0])
    score = probabilistic_f1(y_true, y_pred)
    assert abs(score - 0.0) < 1e-6, f"Expected 0.0, got {score}"

    print("Metric verification passed.")


def prepare_subset_data():
    """
    Creates a small subset of the metadata to speed up the demonstration.
    Updates Config paths to point to these subsets.
    """
    print("\n[2/5] Preparing Data Subsets for Speed...")

    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Load original metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Sample subsets (ensure we have enough for a few batches)
    # We keep the logic simple: random sample.
    # The dataset class handles missing contralateral pairs gracefully (zeros),
    # so we don't need to enforce pair integrity for this smoke test.
    subset_size = 32
    df_train_sub = df_train.head(subset_size).copy()
    df_val_sub = df_val.head(subset_size).copy()
    df_test_sub = df_test.head(subset_size).copy()

    # Save subsets
    train_sub_path = os.path.join(Config.WORKING_DIR, "train_subset.csv")
    val_sub_path = os.path.join(Config.WORKING_DIR, "val_subset.csv")
    test_sub_path = os.path.join(Config.WORKING_DIR, "test_subset.csv")

    df_train_sub.to_csv(train_sub_path, index=False)
    df_val_sub.to_csv(val_sub_path, index=False)
    df_test_sub.to_csv(test_sub_path, index=False)

    # Patch Config to use these subsets
    Config.TRAIN_METADATA_PATH = train_sub_path
    Config.VAL_METADATA_PATH = val_sub_path
    Config.TEST_METADATA_PATH = test_sub_path

    # Patch Config for Speed
    Config.IMG_SIZE = (256, 256)  # Smaller images for faster CPU/GPU processing
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    print(f"Subsets created. Training set size: {len(df_train_sub)}")


def verify_data_loading():
    """Verifies DataLoader construction and batch structure."""
    print("\n[3/5] Verifying Data Loading...")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_stats=False,  # Force recompute on subset
    )

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify Keys
    expected_keys = {"target", "contra", "label"}
    assert expected_keys.issubset(
        batch.keys()
    ), f"Missing keys in batch. Found: {batch.keys()}"

    # Verify Shapes
    # Shape: (Batch, Channels, H, W)
    # Channels = 3 (Image + Age + Implant)
    target = batch["target"]
    contra = batch["contra"]
    label = batch["label"]

    assert target.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), f"Incorrect target shape: {target.shape}"
    assert contra.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), f"Incorrect contra shape: {contra.shape}"
    assert label.shape == (Config.BATCH_SIZE,), f"Incorrect label shape: {label.shape}"

    print("Data loading verification passed.")
    return train_loader


def verify_model(loader):
    """Verifies Model instantiation and forward pass."""
    print("\n[4/5] Verifying Model Architecture & Forward Pass...")

    device = torch.device(Config.DEVICE)
    model = SiameseEfficientNet().to(device)

    # Get a batch
    batch = next(iter(loader))
    target = batch["target"].to(device)
    contra = batch["contra"].to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(target, contra)

    # Verify Output
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected output shape ({Config.BATCH_SIZE}, 1), got {logits.shape}"

    print("Model verification passed.")


def run_full_pipeline_demo():
    """Runs the training loop and submission generation using the library functions."""
    print("\n[5/5] Running Training and Inference Pipeline...")

    # 1. Train
    # This calls library.train.run_training, which handles loops, saving, etc.
    best_model_path = run_training(num_epochs=Config.NUM_EPOCHS)

    assert os.path.exists(best_model_path), "Best model file was not saved."
    print("Training finished successfully.")

    # 2. Generate Submission
    # This calls library.train.generate_submission_file
    generate_submission_file()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify Submission Content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert "prediction_id" in df_sub.columns, "Missing prediction_id column"
    assert "cancer" in df_sub.columns, "Missing cancer column"
    assert len(df_sub) > 0, "Submission file is empty"

    print(
        f"Pipeline demonstration complete. Submission saved to {Config.SUBMISSION_PATH}"
    )
    print(df_sub.head())


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(Config.SEED)

    # 1. Test Metric Logic
    test_metric()

    # 2. Prepare Environment (Subsets & Config)
    prepare_subset_data()

    # 3. Verify Data Loading
    train_loader = verify_data_loading()

    # 4. Verify Model
    verify_model(train_loader)

    # 5. Run Pipeline (Train -> Inference)
    run_full_pipeline_demo()
