import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import IDCRCN
from library.loss import MultiStageLoss
from library.train import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Demonstration of Gesture Recognition Pipeline ===")

    # 1. Setup & Configuration Overrides for Speed
    # We override the Config class attributes to run a lightweight version of the task
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)

    # Update Config paths and parameters
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.CHECKPOINT_DIR = os.path.join(demo_dir, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")

    # Reduce model size for speed
    Config.LSTM_HIDDEN_SIZE = 64
    Config.TCN_NUM_CHANNELS = [64] * 2  # Only 2 layers instead of 10
    Config.BATCH_SIZE = 2
    Config.NUM_EPOCHS = 1
    Config.PATIENCE = 1

    # Ensure directories exist
    Config.ensure_dirs()

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print(f"Configuration updated. Working directory: {Config.WORKING_DIR}")

    # 2. Prepare Data Subsets
    # We create small subsets of the metadata to avoid processing the full dataset
    print("\n--- Preparing Data Subsets ---")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Create subsets (top 5 samples each)
    train_subset = orig_train.head(5)
    val_subset = orig_val.head(5)
    test_subset = orig_test.head(5)

    # Save subsets to the demo directory
    train_subset_path = os.path.join(demo_dir, "train_subset.csv")
    val_subset_path = os.path.join(demo_dir, "val_subset.csv")
    test_subset_path = os.path.join(demo_dir, "test_subset.csv")

    train_subset.to_csv(train_subset_path, index=False)
    val_subset.to_csv(val_subset_path, index=False)
    test_subset.to_csv(test_subset_path, index=False)

    # Point Config to these new subset files
    Config.TRAIN_METADATA_PATH = train_subset_path
    Config.VAL_METADATA_PATH = val_subset_path
    Config.TEST_METADATA_PATH = test_subset_path

    print(
        f"Subsets created: {len(train_subset)} train, {len(val_subset)} val, {len(test_subset)} test samples."
    )

    # 3. Verify Data Loading
    print("\n--- Verifying Data Loading ---")
    train_loader, val_loader, test_loader = get_dataloaders(debug=False)

    # Fetch one batch
    batch = next(iter(train_loader))

    # Assertions to verify batch structure
    assert "features" in batch, "Batch missing 'features' key"
    assert "mask" in batch, "Batch missing 'mask' key"
    assert "frame_labels" in batch, "Batch missing 'frame_labels' key"
    assert "lengths" in batch, "Batch missing 'lengths' key"

    features = batch["features"]
    mask = batch["mask"]
    labels = batch["frame_labels"]
    lengths = batch["lengths"]

    print(f"Batch loaded successfully.")
    print(f"Features shape: {features.shape} (Expected: [Batch, Time, InputDim])")
    print(f"Mask shape: {mask.shape} (Expected: [Batch, Time])")

    # Verify dimensions
    assert features.dim() == 3, "Features should be 3D tensor"
    assert (
        features.shape[2] == Config.INPUT_DIM
    ), f"Feature dim should be {Config.INPUT_DIM}"
    assert mask.shape == (features.shape[0], features.shape[1]), "Mask shape mismatch"
    assert labels.shape == (
        features.shape[0],
        features.shape[1],
    ), "Labels shape mismatch"

    # Verify masking logic: Mask should be False where index >= length
    for i, length in enumerate(lengths):
        assert torch.all(
            mask[i, :length]
        ), f"Mask should be True for valid frames (Sample {i})"
        if features.shape[1] > length:
            assert not torch.any(
                mask[i, length:]
            ), f"Mask should be False for padding (Sample {i})"

    # 4. Verify Model and Loss
    print("\n--- Verifying Model and Loss ---")
    device = Config.get_device()
    model = IDCRCN().to(device)
    criterion = MultiStageLoss().to(device)

    # Move batch to device
    features = features.to(device)
    mask = mask.to(device)
    lengths = lengths.to(device)
    labels = labels.to(device)

    # Forward Pass
    outputs = model(features, mask, lengths)

    # Verify Output Structure
    assert "stage1" in outputs
    assert "stage2" in outputs
    assert "stage3" in outputs

    p3 = outputs["stage3"]
    assert p3.shape == (
        features.shape[0],
        Config.NUM_CLASSES,
        features.shape[1],
    ), f"Output shape mismatch. Got {p3.shape}"

    # Verify Probabilities (Softmax applied)
    # Sum over classes should be approx 1.0 for valid frames
    # Note: The model applies masking at the end, so padded regions might be 0.
    # We check a valid frame.
    valid_idx = 0
    valid_time = 0
    prob_sum = torch.sum(p3[valid_idx, :, valid_time]).item()
    assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities should sum to 1. Got {prob_sum}"

    # Compute Loss
    loss, metrics = criterion(outputs, labels, mask)

    print(f"Loss computed: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    assert isinstance(metrics, dict), "Metrics should be a dictionary"

    # 5. Run Training Loop (Trainer)
    print("\n--- Executing Training Loop (1 Epoch) ---")
    trainer = Trainer(debug=False)

    # We expect this to run for 1 epoch and save a checkpoint
    trainer.fit()

    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not created."
    print("Training finished and checkpoint saved.")

    # 6. Run Prediction
    print("\n--- Executing Prediction ---")
    trainer.predict()

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    # Verify Submission Content
    with open(submission_path, "r") as f:
        lines = f.readlines()

    print(f"Submission generated with {len(lines)} lines.")
    # We expect 5 lines (one for each test sample in subset)
    assert len(lines) == 5, f"Expected 5 predictions, got {len(lines)}"

    # Check format of first line: SessionID,Label1,Label2...
    sample_line = lines[0].strip().split(",")
    print(f"Sample prediction: {sample_line}")
    assert len(sample_line) >= 1, "Invalid submission format"
    assert sample_line[0].startswith("Sample"), "First column should be Sample ID"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
