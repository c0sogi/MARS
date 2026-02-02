import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import provided library components
from library.config import Config
from library.utils import set_seed, MetricTracker
from library.dataset import TumorDataset
from library.model import get_model
from library.train import run_training
from library.inference import run_inference


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast execution...")

    # Override Config attributes for a quick debug run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Use only 100 samples for train/val
    Config.NUM_EPOCHS = 1  # Run only 1 epoch
    Config.PRETRAINED = False  # Disable downloading weights for speed/offline safety
    Config.PROJECT_NAME = "demo_run"

    # Manually update paths that were derived at import time
    # We redirect output to a specific demo folder in working directory
    Config.WORKING_DIR = os.path.join("./working", Config.PROJECT_NAME)
    Config.CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.LOG_PATH = os.path.join(Config.WORKING_DIR, "train.log")

    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create necessary directories
    Config.setup()

    # Create a subset of test metadata for fast inference
    full_test_df = pd.read_csv(Config.TEST_CSV)
    test_subset_path = os.path.join(Config.WORKING_DIR, "test_subset.csv")
    full_test_df.head(50).to_csv(test_subset_path, index=False)
    Config.TEST_CSV = test_subset_path  # Point Config to the subset
    print(f"    Created test subset at {test_subset_path} (50 samples)")

    # -------------------------------------------------------------------------
    # 2. Component Verification: Dataset
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Dataset Logic...")

    # Load a small slice of training data
    train_df = pd.read_csv(Config.TRAIN_CSV).head(10)
    dataset = TumorDataset(train_df, split="train")

    # Check length
    assert (
        len(dataset) == 10
    ), f"Dataset length mismatch. Expected 10, got {len(dataset)}"

    # Check item structure
    img, label = dataset[0]

    # Expected shape: (Channels, Height, Width) -> (3, 48, 48)
    # 48x48 comes from Config.CENTER_CROP_SIZE
    expected_shape = (3, 48, 48)
    assert (
        img.shape == expected_shape
    ), f"Image shape mismatch. Expected {expected_shape}, got {img.shape}"
    assert isinstance(img, torch.Tensor), "Image should be a torch.Tensor"
    assert isinstance(label, torch.Tensor), "Label should be a torch.Tensor"

    print("    PASS: Dataset structure and shapes are correct.")

    # -------------------------------------------------------------------------
    # 3. Component Verification: Model
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = get_model()
    model.to(Config.DEVICE)
    model.eval()

    # Create a dummy batch: (Batch Size, Channels, Height, Width)
    dummy_input = torch.randn(2, 3, 48, 48).to(Config.DEVICE)

    with torch.no_grad():
        output = model(dummy_input)

    # Expected output: (Batch Size, Num Classes) -> (2, 1)
    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"

    print("    PASS: Model forward pass successful with correct output shape.")

    # -------------------------------------------------------------------------
    # 4. Component Verification: MetricTracker
    # -------------------------------------------------------------------------
    print("\n[4] Verifying MetricTracker...")

    tracker = MetricTracker()

    # Simulate a perfect prediction scenario
    # Targets: [0, 1]
    # Preds:   [0.1, 0.9] (Probabilities)
    fake_targets = np.array([0, 1])
    fake_preds = np.array([0.1, 0.9])

    tracker.update(loss=0.5, preds=fake_preds, targets=fake_targets)
    auc_score = tracker.get_auc()

    assert auc_score == 1.0, f"AUC calculation failed. Expected 1.0, got {auc_score}"
    print("    PASS: MetricTracker calculates AUC correctly.")

    # -------------------------------------------------------------------------
    # 5. Pipeline Execution: Training
    # -------------------------------------------------------------------------
    print("\n[5] Executing Training Pipeline...")

    # run_training uses the Config class we modified
    run_training()

    # Verify artifact creation
    if not os.path.exists(Config.CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Training failed to create checkpoint at {Config.CHECKPOINT_PATH}"
        )

    print(f"    PASS: Training completed. Checkpoint saved.")

    # -------------------------------------------------------------------------
    # 6. Pipeline Execution: Inference
    # -------------------------------------------------------------------------
    print("\n[6] Executing Inference Pipeline...")

    # run_inference uses Config, but we pass paths explicitly to be sure
    run_inference(
        checkpoint_path=Config.CHECKPOINT_PATH, output_path=Config.SUBMISSION_PATH
    )

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Inference failed to create submission at {Config.SUBMISSION_PATH}"
        )

    # Verify submission content format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    required_cols = {"id", "label"}
    assert required_cols.issubset(
        sub_df.columns
    ), f"Submission missing columns. Found {sub_df.columns}"
    assert (
        len(sub_df) == 50
    ), f"Submission length mismatch. Expected 50 (subset), got {len(sub_df)}"

    print(
        f"    PASS: Inference completed. Submission generated with {len(sub_df)} rows."
    )

    print("\n=== Demonstration Complete: All checks passed ===")


if __name__ == "__main__":
    main()
