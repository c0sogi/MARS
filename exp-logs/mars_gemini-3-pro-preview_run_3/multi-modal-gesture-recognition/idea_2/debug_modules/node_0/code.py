import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
import shutil

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.utils import seed_everything
import library.features as features_lib
import library.dataset as dataset_lib
import library.model as model_lib
import library.loss as loss_lib
import library.trainer as trainer_lib
import library.inference as inference_lib


def run_demo():
    print("=== Starting MS-TCN Gesture Recognition Demo ===")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[1] Configuring Demo Environment...")

    # Modify Config for a quick demo run
    # We use a separate working directory to avoid messing with real training artifacts
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Update Config class attributes directly
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "mstcn_demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Update Cache paths to point to demo dir
    Config.TRAIN_CACHE_PATH = os.path.join(DEMO_DIR, "train_features.npz")
    Config.VAL_CACHE_PATH = os.path.join(DEMO_DIR, "val_features.npz")
    Config.TEST_CACHE_PATH = os.path.join(DEMO_DIR, "test_features.npz")

    # Hyperparameters for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG_SUBSET_SIZE = 10  # Only process 10 samples
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure reproducibility
    seed_everything(Config.RANDOM_SEED)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Subset Size: {Config.DEBUG_SUBSET_SIZE}")
    print(f"    Device: {Config.DEVICE}")

    # ==========================================
    # 2. Feature Extraction & Data Loading
    # ==========================================
    print("\n[2] Testing Feature Extraction & Data Loading...")

    # Load training data (subset)
    # This triggers feature extraction if cache doesn't exist
    train_data = features_lib.get_train_data(
        load_cached_data=False, subset_size=Config.DEBUG_SUBSET_SIZE
    )

    # Verification
    assert (
        len(train_data["ids"]) == Config.DEBUG_SUBSET_SIZE
    ), f"Expected {Config.DEBUG_SUBSET_SIZE} samples, got {len(train_data['ids'])}"

    # Check feature dimensions: (Time, Input_Dim)
    sample_feat = train_data["features"][0]
    assert (
        sample_feat.shape[1] == Config.INPUT_DIM
    ), f"Expected input dim {Config.INPUT_DIM}, got {sample_feat.shape[1]}"

    print("    Feature extraction successful.")
    print(f"    Sample Feature Shape: {sample_feat.shape}")

    # Create DataLoaders
    train_loader, val_loader, test_loader = dataset_lib.create_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug_subset_size=Config.DEBUG_SUBSET_SIZE,
    )

    # Verify DataLoader and Collate Function
    batch = next(iter(train_loader))
    padded_features, padded_labels, mask, lengths, ids = batch

    assert padded_features.dim() == 3, "Features should be (Batch, Time, Dim)"
    assert padded_labels.dim() == 2, "Labels should be (Batch, Time)"
    assert mask.dim() == 2, "Mask should be (Batch, Time)"
    assert padded_features.size(0) == Config.BATCH_SIZE, "Batch size mismatch"

    print("    DataLoader & Collate function verified.")

    # ==========================================
    # 3. Model & Loss Verification
    # ==========================================
    print("\n[3] Testing Model Architecture & Loss...")

    model = model_lib.MSTCN().to(Config.DEVICE)
    criterion = loss_lib.ActionSegmentationLoss()

    # Move batch to device
    padded_features = padded_features.to(Config.DEVICE)
    padded_labels = padded_labels.to(Config.DEVICE)
    mask = mask.to(Config.DEVICE)

    # Forward Pass
    outputs = model(padded_features, mask)

    # Verify Output Structure
    # MSTCN returns a list of outputs, one for each stage
    assert isinstance(outputs, list), "Model output should be a list"
    assert (
        len(outputs) == Config.NUM_STAGES
    ), f"Expected {Config.NUM_STAGES} stages, got {len(outputs)}"

    # Check shape of final stage output: (Batch, Num_Classes, Time)
    final_out = outputs[-1]
    assert final_out.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
        padded_features.shape[1],
    ), f"Output shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES, padded_features.shape[1])}, got {final_out.shape}"

    # Compute Loss
    loss_val = criterion(outputs, padded_labels, mask)
    assert not torch.isnan(loss_val), "Loss is NaN"
    assert loss_val.item() > 0, "Loss should be positive"

    print(f"    Forward pass successful. Loss: {loss_val.item():.4f}")

    # ==========================================
    # 4. Training Loop
    # ==========================================
    print("\n[4] Running Training Loop (1 Epoch)...")

    trainer = trainer_lib.Trainer(config=Config)

    # Run fit (trains for Config.NUM_EPOCHS = 1)
    trainer.fit(debug_subset_size=Config.DEBUG_SUBSET_SIZE)

    # Verify Checkpoint creation
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"

    print("    Training completed. Checkpoint saved.")

    # ==========================================
    # 5. Inference & Submission
    # ==========================================
    print("\n[5] Running Inference & Generating Submission...")

    # Run inference using the trained model
    inference_lib.run_inference(debug_subset_size=Config.DEBUG_SUBSET_SIZE)

    # Verify Submission File
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Check content format
    with open(Config.SUBMISSION_PATH, "r") as f:
        lines = f.readlines()

    print(f"    Generated {len(lines)} prediction lines.")
    if len(lines) > 0:
        parts = lines[0].strip().split(",")
        # Format: SessionID, Label1, Label2...
        # SessionID usually starts with 'Sample' or 'Session'
        assert len(parts) >= 1, "Invalid submission line format"
        print(f"    Sample prediction: {lines[0].strip()}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
