import os
import sys
import shutil
import pandas as pd
import torch
import numpy as np

# Import from the provided library
from library.config import Config
from library.utils import set_seed, get_device
from library.data_loader import get_dataloaders
from library.model import SBMD_CRCN
from library.losses import CombinedLoss
from library.train import train_one_epoch, validate
from library.predict import generate_predictions


def main():
    print("=== Starting Library Usage Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment for demo...")

    # Set a specific working directory for this demo to avoid clutter
    demo_working_dir = "./working/demo_run"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Modify Config singleton for this run
    # We use a very small batch size and 1 epoch for speed
    Config.WORKING_DIR = demo_working_dir
    Config.CACHE_DIR = os.path.join(demo_working_dir, "cache")
    Config.CHECKPOINT_DIR = os.path.join(demo_working_dir, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(demo_working_dir, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    Config.BATCH_SIZE = 2
    Config.NUM_EPOCHS = 1
    Config.EARLY_STOPPING_PATIENCE = 1

    # Set seed for reproducibility
    set_seed(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Prepare Data Subsets (for Speed)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Creating data subsets...")

    # We create small CSVs pointing to the first few samples of the real data
    # This allows the data loader to run the real processing logic without
    # waiting for the full dataset.

    metadata_dir = "./metadata"
    demo_meta_dir = os.path.join(demo_working_dir, "metadata")
    os.makedirs(demo_meta_dir, exist_ok=True)

    # Helper to subset metadata
    def create_subset(filename, n_rows=10):
        src = os.path.join(metadata_dir, filename)
        dst = os.path.join(demo_meta_dir, filename)
        if os.path.exists(src):
            df = pd.read_csv(src)
            # Take top n_rows
            subset = df.head(n_rows)
            subset.to_csv(dst, index=False)
            return dst
        return None

    Config.TRAIN_METADATA_PATH = create_subset("train.csv", n_rows=20)
    Config.VAL_METADATA_PATH = create_subset("val.csv", n_rows=10)
    Config.TEST_METADATA_PATH = create_subset("test.csv", n_rows=10)

    print(f"Created subsets in {demo_meta_dir}")

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    print("\n[Step 3] Loading data (processing raw .mat and .wav files)...")

    # This will trigger process_dataset internally for our subsets
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Verify batch structure
    sample_batch = next(iter(train_loader))
    features = sample_batch["features"]
    mask = sample_batch["mask"]
    cls_targets = sample_batch["cls_targets"]
    bnd_targets = sample_batch["bnd_targets"]

    print(f"Batch Features Shape: {features.shape} (Batch, Time, Feats)")
    print(f"Batch Mask Shape: {mask.shape}")
    print(f"Batch Class Targets Shape: {cls_targets.shape}")

    # Assertions
    assert features.dim() == 3, "Features should be (B, T, D)"
    assert (
        features.shape[2] == Config.INPUT_DIM
    ), f"Feature dim should be {Config.INPUT_DIM}"
    assert mask.shape == cls_targets.shape, "Mask and targets should align"

    # -------------------------------------------------------------------------
    # 4. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[Step 4] Initializing SBMD_CRCN Model...")
    model = SBMD_CRCN().to(device)

    # Move batch to device
    features = features.to(device)
    mask = mask.to(device)

    print("Running forward pass...")
    outputs = model(features, mask)

    # Verify outputs
    expected_keys = [
        "stage1_cls",
        "stage1_bnd",
        "stage2_cls",
        "stage2_bnd",
        "stage3_cls",
        "stage3_bnd",
    ]
    for k in expected_keys:
        assert k in outputs, f"Missing output key: {k}"
        assert outputs[k].shape[0] == features.shape[0], f"Batch dim mismatch for {k}"
        assert outputs[k].shape[1] == features.shape[1], f"Time dim mismatch for {k}"

    print("Forward pass successful. Output shapes verified.")

    # -------------------------------------------------------------------------
    # 5. Loss Computation
    # -------------------------------------------------------------------------
    print("\n[Step 5] Computing Loss...")
    criterion = CombinedLoss().to(device)

    targets = {
        "cls_targets": cls_targets.to(device),
        "bnd_targets": bnd_targets.to(device),
        "mask": mask,
    }

    loss_dict = criterion(outputs, targets)
    total_loss = loss_dict["loss"]

    print(f"Total Loss: {total_loss.item():.4f}")
    print(
        f"Component Losses: { {k: f'{v:.4f}' for k, v in loss_dict.items() if k != 'loss'} }"
    )

    assert not torch.isnan(total_loss), "Loss is NaN"
    assert total_loss > 0, "Loss should be positive"

    # -------------------------------------------------------------------------
    # 6. Training Loop
    # -------------------------------------------------------------------------
    print("\n[Step 6] Running Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Capture initial weights to verify update
    initial_weight = model.stage1.fc_cls.weight.clone()

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Train Loss: {train_loss:.4f}")

    # Check if weights updated
    final_weight = model.stage1.fc_cls.weight
    assert not torch.equal(
        initial_weight, final_weight
    ), "Model weights did not update!"
    print("Model weights updated successfully.")

    # Validation
    print("Running Validation...")
    val_loss, val_score = validate(model, val_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f} | Val Score (Levenshtein): {val_score:.4f}")

    # Save dummy best model for prediction step
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    print(f"Saved model to {Config.BEST_MODEL_PATH}")

    # -------------------------------------------------------------------------
    # 7. Prediction / Inference
    # -------------------------------------------------------------------------
    print("\n[Step 7] Generating Predictions on Test Subset...")

    # We force load_cached_data=True to use the cache we just generated in Step 3
    # (Though get_dataloaders handles this logic, we call generate_predictions wrapper)

    # Note: generate_predictions in library/predict.py calls get_dataloaders internally.
    # Since we already populated the cache in Step 3 for the 'test' split, it should run fast.

    # Redirect stdout temporarily if we want to suppress print, but here we keep it
    # to show progress.
    generate_predictions(load_cached_data=True)

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file found at {Config.SUBMISSION_PATH}")
        with open(Config.SUBMISSION_PATH, "r") as f:
            lines = f.readlines()
            print(f"Number of predictions: {len(lines)}")
            if len(lines) > 0:
                print(f"Sample prediction: {lines[0].strip()}")

        # Basic format check
        # Expected: SessionID,Label1,Label2...
        parts = lines[0].strip().split(",")
        assert len(parts) >= 1, "Invalid submission format"
        assert (
            "Sample" in parts[0] or "Session" in parts[0]
        ), "Invalid Session ID format"
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
