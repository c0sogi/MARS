import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# 1. Setup & Configuration Patching
# -----------------------------------------------------------------------------
# We patch the Config class to use a temporary directory and a small subset of data
# to ensure the demonstration runs quickly and does not modify the original metadata.

# Define temporary directories
DEMO_DIR = "./working/demo_run"
META_DIR = os.path.join(DEMO_DIR, "metadata")
CACHE_DIR = os.path.join(DEMO_DIR, "cache")
SUB_DIR = os.path.join(DEMO_DIR, "submission")

# Clean up previous run if exists
if os.path.exists(DEMO_DIR):
    shutil.rmtree(DEMO_DIR)

os.makedirs(META_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)

# Create mini metadata files (top 10 samples)
for split in ["train", "val", "test"]:
    src_path = f"./metadata/{split}.csv"
    dst_path = os.path.join(META_DIR, f"{split}.csv")
    if os.path.exists(src_path):
        df = pd.read_csv(src_path)
        df_mini = df.head(10)  # Use only 10 samples for speed
        df_mini.to_csv(dst_path, index=False)

# Suppress tqdm progress bars
import tqdm


def silent_tqdm(iterable, *args, **kwargs):
    return iterable


tqdm.tqdm = silent_tqdm

# Import and Patch Config
from library.config import Config

Config.METADATA_DIR = META_DIR
Config.WORKING_DIR = CACHE_DIR
Config.SUBMISSION_DIR = SUB_DIR
Config.NUM_EPOCHS = 1  # Run only 1 epoch
Config.BATCH_SIZE = 2  # Small batch size
Config.PATIENCE = 1
Config.SEED = 42

# Import Library Modules
from library.utils import set_seed, compute_levenshtein
from library.dataset import GestureDataset
from library.model import GLT_CRCN
from library.loss import CombinedLoss
from library.trainer import Trainer
from library.postprocessing import (
    apply_median_filter,
    decode_predictions,
    generate_submission,
)

if __name__ == "__main__":
    print("=== Starting GLT-CRCN Demonstration ===\n")

    # Set Random Seed
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Dataset & DataLoader
    # -------------------------------------------------------------------------
    print("\n[Step 1] Initializing Dataset and DataLoader...")
    # Initialize Training Dataset
    train_dataset = GestureDataset(split="train", augment=True, debug=False)
    print(f"Train Dataset Size: {len(train_dataset)}")

    # Verify single item
    sample_item = train_dataset[0]
    print(f"Sample ID: {sample_item['sample_id']}")
    print(f"Features Shape: {sample_item['features'].shape}")  # Expected: (T, 85)
    print(f"Targets Shape: {sample_item['targets'].shape}")  # Expected: (T,)

    assert (
        sample_item["features"].shape[1] == Config.INPUT_DIM
    ), "Incorrect feature dimension"

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=GestureDataset.collate_fn,
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    features = batch["features"].to(device)
    targets = batch["targets"].to(device)
    mask = batch["mask"].to(device)

    print(f"Batch Features Shape: {features.shape}")  # (B, T_max, 85)
    print(f"Batch Mask Shape: {mask.shape}")  # (B, T_max)

    # -------------------------------------------------------------------------
    # 3. Model Instantiation & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[Step 2] Model Forward Pass...")
    model = GLT_CRCN().to(device)

    # Forward pass
    outputs = model(features, mask)

    # Verify outputs
    print("Model Output Keys:", outputs.keys())
    stage3_out = outputs["stage3_cls"]
    print(f"Stage 3 Output Shape: {stage3_out.shape}")  # (B, T_max, 21)

    assert (
        stage3_out.shape[2] == Config.NUM_CLASSES
    ), "Incorrect number of classes in output"
    assert (
        stage3_out.shape[:2] == features.shape[:2]
    ), "Output temporal dimension mismatch"

    # -------------------------------------------------------------------------
    # 4. Loss Computation
    # -------------------------------------------------------------------------
    print("\n[Step 3] Computing Loss...")
    criterion = CombinedLoss().to(device)
    loss = criterion(outputs, targets, mask)

    print(f"Calculated Loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss contains NaNs"

    # -------------------------------------------------------------------------
    # 5. Training Loop (Trainer)
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Trainer (1 Epoch)...")
    # Initialize Validation Loader
    val_dataset = GestureDataset(split="val", augment=False, debug=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=GestureDataset.collate_fn,
    )

    trainer = Trainer(device=device)

    # Run fit (Training + Validation)
    trainer.fit(train_loader, val_loader, num_epochs=Config.NUM_EPOCHS)

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint successfully saved at: {checkpoint_path}")
    else:
        print(
            "Warning: Checkpoint was not created (likely due to no improvement or short run)."
        )

    # -------------------------------------------------------------------------
    # 6. Inference & Post-processing
    # -------------------------------------------------------------------------
    print("\n[Step 5] Inference and Post-processing...")
    # Initialize Test Dataset
    test_dataset = GestureDataset(split="test", augment=False, debug=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=GestureDataset.collate_fn,
    )

    # Generate raw predictions
    predictions = trainer.predict(test_loader)
    print(f"Generated predictions for {len(predictions)} test samples.")

    # Demonstrate Post-processing on the first prediction
    sample_id = list(predictions.keys())[0]
    raw_labels = predictions[sample_id]

    # 1. Median Filter
    smoothed_labels = apply_median_filter(raw_labels, kernel_size=15)

    # 2. Decode (Collapse & Remove Background)
    decoded_gestures = decode_predictions(smoothed_labels)

    print(f"Sample {sample_id} Raw Length: {len(raw_labels)}")
    print(f"Sample {sample_id} Decoded Gestures: {decoded_gestures}")

    # 3. Generate Submission File
    generate_submission(predictions, output_filename="submission_demo.csv")
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission_demo.csv")
    assert os.path.exists(submission_path), "Submission file not created"

    # -------------------------------------------------------------------------
    # 7. Metric Calculation
    # -------------------------------------------------------------------------
    print("\n[Step 6] Metric Verification (Levenshtein)...")
    seq1 = [1, 2, 3, 4]
    seq2 = [1, 2, 5, 4]  # Substitution: 3 -> 5
    dist = compute_levenshtein(seq1, seq2)
    print(f"Sequence 1: {seq1}")
    print(f"Sequence 2: {seq2}")
    print(f"Levenshtein Distance: {dist}")

    assert dist == 1, "Levenshtein distance calculation incorrect"

    print("\n=== Demonstration Complete ===")
