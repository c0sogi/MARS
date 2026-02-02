import os
import shutil
import pandas as pd
import torch
import numpy as np
import library.config

# ==============================================================================
# 1. PREPARATION: Create Subset Data & Configure Environment
# ==============================================================================
# Define temporary working directories for this demonstration
DEMO_DIR = "./working/demo_run"
SUBSET_META_DIR = os.path.join(DEMO_DIR, "metadata")
DEMO_SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

os.makedirs(DEMO_DIR, exist_ok=True)
os.makedirs(SUBSET_META_DIR, exist_ok=True)

# Create subset metadata to speed up execution
# We take a small number of samples from the original metadata
print("Creating subset metadata for demonstration...")
splits = ["train", "val", "test"]
counts = {"train": 10, "val": 4, "test": 4}

for split in splits:
    original_path = f"./metadata/{split}_metadata.csv"
    if os.path.exists(original_path):
        df = pd.read_csv(original_path)
        subset_df = df.head(counts[split])
        subset_path = os.path.join(SUBSET_META_DIR, f"{split}_metadata.csv")
        subset_df.to_csv(subset_path, index=False)
        print(f"  Created {split} subset with {len(subset_df)} samples.")

# Monkey-patch the configuration module BEFORE importing other library modules
# This ensures that the classes use our demo settings (paths, epochs, etc.)
print("Configuring library parameters...")
import importlib

importlib.reload(library.config)  # Cite debug_lesson_2
library.config.METADATA_DIR = SUBSET_META_DIR
library.config.WORKING_DIR = DEMO_DIR
library.config.SUBMISSION_PATH = DEMO_SUBMISSION_PATH
library.config.EPOCHS = 1
library.config.BATCH_SIZE = 2
library.config.WARMUP_EPOCHS = 0
# Reduce pillar counts for speed
library.config.MAX_PILLARS_TRAIN = 1000
library.config.MAX_PILLARS_TEST = 1000

# Now import the library modules which will use the updated config
import library.dataset
import library.model
import library.solver

importlib.reload(library.dataset)  # Cite debug_lesson_2
importlib.reload(library.model)  # Cite debug_lesson_2
importlib.reload(library.solver)  # Cite debug_lesson_2

from library.dataset import LidarDataset
from library.model import PointPillars
from library.solver import Solver

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

if __name__ == "__main__":
    print("\n" + "=" * 40)
    print("STARTING LIBRARY DEMONSTRATION")
    print("=" * 40)

    # --------------------------------------------------------------------------
    # 2. DATASET VERIFICATION
    # --------------------------------------------------------------------------
    print("\n[1/3] Verifying LidarDataset...")

    # Initialize dataset with the subset
    # Note: This will trigger GT Database building on the small subset
    train_dataset = LidarDataset(split="train", load_cached_data=False)

    # Assertions
    assert len(train_dataset) == 10, f"Expected 10 samples, got {len(train_dataset)}"
    assert train_dataset.is_train is True

    # Fetch a single sample
    sample = train_dataset[0]
    required_keys = [
        "pillar_features",
        "pillar_coords",
        "num_points",
        "gt_boxes",
        "gt_labels",
        "sample_token",
    ]
    for key in required_keys:
        assert key in sample, f"Missing key in dataset sample: {key}"

    print(f"  Sample 0 Token: {sample['sample_token']}")
    print(f"  Pillar Features Shape: {sample['pillar_features'].shape}")
    print("  Dataset verification successful.")

    # --------------------------------------------------------------------------
    # 3. MODEL VERIFICATION
    # --------------------------------------------------------------------------
    print("\n[2/3] Verifying PointPillars Model...")

    # Create a batch
    batch_list = [train_dataset[i] for i in range(2)]
    batch_dict = LidarDataset.collate_fn(batch_list)

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointPillars().to(device)

    # Prepare batch for device
    for key, val in batch_dict.items():
        if isinstance(val, torch.Tensor):
            batch_dict[key] = val.to(device)
        elif isinstance(val, list):
            # gt_boxes is a list of tensors
            batch_dict[key] = [
                v.to(device) if isinstance(v, torch.Tensor) else v for v in val
            ]

    # Forward Pass (Training Mode)
    model.train()
    loss_dict = model(batch_dict)

    # Check outputs
    assert "loss" in loss_dict, "Model output missing 'loss' key"
    assert "cls_loss" in loss_dict
    assert "box_loss" in loss_dict
    assert isinstance(loss_dict["loss"].item(), float), "Loss is not a float"

    print(f"  Forward pass successful. Total Loss: {loss_dict['loss'].item():.4f}")

    # --------------------------------------------------------------------------
    # 4. SOLVER (TRAINING & INFERENCE) VERIFICATION
    # --------------------------------------------------------------------------
    print("\n[3/3] Verifying Solver (Train & Inference Loop)...")

    solver = Solver()

    # Run Training (1 Epoch on subset)
    print("  Starting Training...")
    solver.fit()

    # Check if checkpoint was created (might not be if val loss didn't improve, but usually it saves once)
    # In this short run, val loss might be inf initially, so any valid loss improves it.
    checkpoint_path = os.path.join(DEMO_DIR, "model_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print("  Checkpoint created successfully.")
    else:
        print("  Notice: No checkpoint created (validation might not have improved).")

    # Run Inference
    print("  Starting Inference...")
    solver.inference()

    # Verify Submission File
    if os.path.exists(DEMO_SUBMISSION_PATH):
        submission_df = pd.read_csv(DEMO_SUBMISSION_PATH)
        print(f"  Submission file created at {DEMO_SUBMISSION_PATH}")
        print(f"  Rows in submission: {len(submission_df)}")
        assert (
            len(submission_df) == 4
        ), f"Expected 4 predictions (test subset size), got {len(submission_df)}"
        assert "PredictionString" in submission_df.columns
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n" + "=" * 40)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 40)
