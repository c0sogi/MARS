import os
import shutil
import torch
import pandas as pd
import numpy as np
import sys
from unittest.mock import MagicMock

# Import provided library modules
import library.config as config
import library.utils as utils
import library.features as features
import library.dataset as dataset
import library.model as model_lib
import library.loss as loss_lib
import library.train as train_lib
import library.predict as predict_lib


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Patching for Speed/Safety
    # -------------------------------------------------------------------------
    # We will run on a small subset of data and use a separate working directory.

    DEMO_DIR = "./working/demo_run"
    DEMO_CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    DEMO_CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    DEMO_SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

    # Clean up previous demo run if exists
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)

    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)
    os.makedirs(DEMO_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DEMO_SUBMISSION_DIR, exist_ok=True)

    print(f"Working directory set to: {DEMO_DIR}")

    # Define a mock load_metadata function to return a subset of data
    original_load_metadata = utils.load_metadata

    def mock_load_metadata(split="train"):
        # Load the real full metadata
        df = original_load_metadata(split)

        # Return a small subset to ensure speed
        if split == "train":
            return df.head(10).copy()
        elif split == "val":
            return df.head(5).copy()
        elif split == "test":
            return df.head(5).copy()
        return df

    # Apply Patches to Library Modules
    # We need to patch the variables/functions where they are used/imported

    # Patch CACHE_DIR in features and config
    features.CACHE_DIR = DEMO_CACHE_DIR
    config.CACHE_DIR = DEMO_CACHE_DIR

    # Patch load_metadata in utils, dataset, features, and train
    utils.load_metadata = mock_load_metadata
    dataset.load_metadata = mock_load_metadata
    features.load_metadata = mock_load_metadata
    train_lib.load_metadata = mock_load_metadata

    # Patch directories in train and predict
    train_lib.CHECKPOINT_DIR = DEMO_CHECKPOINT_DIR
    predict_lib.CHECKPOINT_DIR = DEMO_CHECKPOINT_DIR
    predict_lib.SUBMISSION_DIR = DEMO_SUBMISSION_DIR

    # Set fixed seed
    utils.set_seed(42)

    print("Configuration patched for demo execution.")

    # -------------------------------------------------------------------------
    # 2. Dataset & Feature Extraction Demo
    # -------------------------------------------------------------------------
    print("\n--- Dataset & Feature Extraction ---")

    # Initialize Dataset (this will trigger processing and caching of the subset)
    # augment=False for deterministic checks
    ds_train = dataset.GestureDataset(
        split="train", augment=False, load_cached_data=False
    )

    print(f"Training Subset Size: {len(ds_train)}")
    assert len(ds_train) == 10, "Dataset subset size mismatch."

    # Retrieve one sample
    sample = ds_train[0]

    # Verify keys
    expected_keys = {"features", "class_target", "boundary_target", "mask", "sample_id"}
    assert expected_keys.issubset(
        sample.keys()
    ), f"Missing keys in dataset item. Found: {sample.keys()}"

    # Verify Shapes
    # Features: (T, INPUT_DIM) -> INPUT_DIM is 118
    feats = sample["features"]
    cls_target = sample["class_target"]
    bnd_target = sample["boundary_target"]
    mask = sample["mask"]

    T = feats.shape[0]
    print(f"Sample 0 Sequence Length: {T}")
    print(f"Feature Shape: {feats.shape}")

    assert (
        feats.shape[1] == config.INPUT_DIM
    ), f"Feature dim mismatch. Expected {config.INPUT_DIM}, got {feats.shape[1]}"
    assert cls_target.shape[0] == T, "Class target length mismatch."
    assert bnd_target.shape[0] == T, "Boundary target length mismatch."
    assert mask.shape[0] == T, "Mask length mismatch."

    # Check Data Types
    assert feats.dtype == torch.float32, "Features should be float32"
    assert cls_target.dtype == torch.long, "Class target should be long"
    assert bnd_target.dtype == torch.float32, "Boundary target should be float32"

    print("Dataset verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model & Loss Logic Demo
    # -------------------------------------------------------------------------
    print("\n--- Model & Loss Logic ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate Model
    model = model_lib.SBG_CRCN().to(device)

    # Create a dummy batch using collate_fn
    batch_list = [ds_train[0], ds_train[1]]
    batch = dataset.collate_fn(batch_list)

    b_feats = batch["features"].to(device)
    b_mask = batch["mask"].to(device)
    b_cls = batch["class_target"].to(device)
    b_bnd = batch["boundary_target"].to(device)

    print(f"Batch Input Shape: {b_feats.shape}")  # (B, T_max, D)

    # Forward Pass
    outputs = model(b_feats, b_mask)

    # Verify Output Structure
    assert "stage1" in outputs
    assert "stage2" in outputs
    assert "stage3" in outputs

    stage3_out = outputs["stage3"]
    s3_cls = stage3_out["class_probs"]
    s3_bnd = stage3_out["boundary_probs"]

    print(f"Stage 3 Class Probs Shape: {s3_cls.shape}")  # (B, T, NumClasses)
    assert s3_cls.shape[2] == config.NUM_CLASSES, "Output classes mismatch"
    assert s3_bnd.shape[2] == 1, "Output boundary dim mismatch"

    # Loss Calculation
    criterion = loss_lib.ActionSegmentationLoss().to(device)
    loss, metrics = criterion(outputs, b_cls, b_bnd, b_mask)

    print(f"Computed Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    print("Model and Loss verification passed.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Demo
    # -------------------------------------------------------------------------
    print("\n--- Training Loop Demo ---")

    # Run training for 1 epoch with a small batch size
    # We use load_cached_data=True now since we computed it in step 2 (for train)
    # Note: validation set will be computed now and cached

    best_model_path = train_lib.train_model(
        num_epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        patience=1,
        load_cached_data=True,  # Use the cache we just populated/defined
        augment=False,
    )

    assert os.path.exists(best_model_path), "Best model checkpoint was not created."
    print(f"Training finished. Checkpoint saved at: {best_model_path}")

    # -------------------------------------------------------------------------
    # 5. Prediction & Submission Demo
    # -------------------------------------------------------------------------
    print("\n--- Prediction & Submission Demo ---")

    submission_file = "demo_submission.csv"

    predict_lib.generate_submission(
        checkpoint_path=best_model_path,
        output_file=submission_file,
        batch_size=2,
        load_cached_data=False,  # Compute test features fresh
    )

    submission_path = os.path.join(DEMO_SUBMISSION_DIR, submission_file)
    assert os.path.exists(submission_path), "Submission file not created."

    # Verify content format
    with open(submission_path, "r") as f:
        lines = f.readlines()
        print(f"Submission lines generated: {len(lines)}")
        if len(lines) > 0:
            print(f"Sample line: {lines[0].strip()}")
            # Check format: SessionID,labels...
            parts = lines[0].strip().split(",")
            assert len(parts) >= 1, "Invalid submission line format"

    print("Prediction verification passed.")

    # -------------------------------------------------------------------------
    # 6. Metric Utility Demo
    # -------------------------------------------------------------------------
    print("\n--- Metric Utility Demo ---")

    hyp = [1, 2, 3]
    ref = [1, 2, 3]
    score_perfect = utils.compute_levenshtein_score([hyp], [ref])
    print(f"Perfect Score (0.0 expected): {score_perfect}")
    assert score_perfect == 0.0

    hyp_bad = [1, 2]
    ref_bad = [1, 2, 3]
    score_bad = utils.compute_levenshtein_score([hyp_bad], [ref_bad])
    print(f"Error Score (0.33 expected): {score_bad:.2f}")
    assert abs(score_bad - 1.0 / 3.0) < 1e-5

    print("Metric verification passed.")

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
