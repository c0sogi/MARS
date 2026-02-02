import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import provided library modules
from library.config import Config
from library.utils import set_seed
from library.features import prepare_datasets
from library.dataset import GestureDataset, get_dataloader
from library.model import SymG_CRCN
from library.loss import MultiStageLoss
from library.engine import Trainer
from library.predict import Predictor


def main():
    print("Initializing demonstration...")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for speed optimization
    Config.DEBUG_SUBSET_SIZE = 20  # Use only 20 samples for demo
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.EARLY_STOPPING_PATIENCE = 1

    # Ensure working directory is clean for demo purposes (optional but good for isolation)
    # We keep the structure but maybe clear old checkpoints if needed.
    # For this run, we just rely on overwriting.

    set_seed(Config.SEED)
    print("Configuration updated for rapid demonstration.")

    # -------------------------------------------------------------------------
    # 2. Data Preparation & Loading
    # -------------------------------------------------------------------------
    print("\n--- Step 2: Data Preparation ---")
    # Force re-computation to ensure we use the DEBUG_SUBSET_SIZE
    # We remove existing cache files if they exist to force regeneration with subset
    if os.path.exists(Config.TRAIN_CACHE_PATH):
        os.remove(Config.TRAIN_CACHE_PATH)
    if os.path.exists(Config.VAL_CACHE_PATH):
        os.remove(Config.VAL_CACHE_PATH)
    if os.path.exists(Config.TEST_CACHE_PATH):
        os.remove(Config.TEST_CACHE_PATH)

    # This will generate .npz files in ./working/idea_23/ using the subset
    data_dicts = prepare_datasets(load_cached_data=False)

    # Verify data generation
    assert "train" in data_dicts
    assert len(data_dicts["train"]["features"]) <= Config.DEBUG_SUBSET_SIZE
    print(f"Data prepared. Train subset size: {len(data_dicts['train']['features'])}")

    # Instantiate Dataset
    train_dataset = GestureDataset("train", augment=False)
    val_dataset = GestureDataset("val", augment=False)

    # Verify Dataset item
    feat, lbl, bnd, sid = train_dataset[0]
    print(f"Sample 0 Feature Shape: {feat.shape}")
    print(f"Sample 0 Label Shape: {lbl.shape}")

    assert (
        feat.shape[1] == Config.INPUT_DIM
    ), f"Expected input dim {Config.INPUT_DIM}, got {feat.shape[1]}"
    assert feat.shape[0] == lbl.shape[0], "Feature time dim matches label time dim"

    # Instantiate DataLoader
    train_loader = get_dataloader("train", batch_size=Config.BATCH_SIZE, shuffle=True)

    # Fetch one batch to verify Collate function
    batch = next(iter(train_loader))
    features = batch["features"]
    mask = batch["mask"]
    lengths = batch["lengths"]
    labels = batch["labels"]
    boundaries = batch["boundaries"]

    print(f"Batch Features Shape: {features.shape}")  # (B, T, D)
    print(f"Batch Mask Shape: {mask.shape}")  # (B, T)

    assert features.size(0) == Config.BATCH_SIZE
    assert features.size(2) == Config.INPUT_DIM
    assert mask.size(0) == Config.BATCH_SIZE
    assert mask.size(1) == features.size(1)

    # -------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n--- Step 3: Model Forward Pass ---")
    device = torch.device(Config.DEVICE)
    model = SymG_CRCN().to(device)

    # Move batch to device
    features = features.to(device)
    mask = mask.to(device)
    lengths = lengths.to(device)

    # Forward
    outputs = model(features, mask, lengths)

    # Verify Outputs
    expected_keys = [
        "stage1_cls",
        "stage1_bnd",
        "stage2_cls",
        "stage2_bnd",
        "stage3_cls",
        "stage3_bnd",
    ]
    for k in expected_keys:
        assert k in outputs, f"Missing key {k} in model output"
        assert outputs[k].shape[0] == Config.BATCH_SIZE, f"Batch size mismatch in {k}"
        # Check time dimension matches input
        assert (
            outputs[k].shape[1] == features.shape[1]
        ), f"Time dimension mismatch in {k}"

    print("Model forward pass successful. Output keys verified.")

    # -------------------------------------------------------------------------
    # 4. Loss Computation
    # -------------------------------------------------------------------------
    print("\n--- Step 4: Loss Computation ---")
    criterion = MultiStageLoss().to(device)

    labels = labels.to(device)
    boundaries = boundaries.to(device)

    loss, metrics = criterion(outputs, labels, boundaries, mask)

    print(f"Total Loss: {loss.item():.4f}")
    print("Metrics:", list(metrics.keys()))

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    # -------------------------------------------------------------------------
    # 5. Training Loop (Trainer)
    # -------------------------------------------------------------------------
    print("\n--- Step 5: Training Loop ---")
    trainer = Trainer()

    # Run training for 1 epoch (as configured)
    trainer.train(num_epochs=Config.NUM_EPOCHS)

    # Check if best model was saved
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint not found after training."
    print(f"Training complete. Checkpoint saved at {Config.BEST_MODEL_PATH}")

    # -------------------------------------------------------------------------
    # 6. Inference (Predictor)
    # -------------------------------------------------------------------------
    print("\n--- Step 6: Inference ---")
    predictor = Predictor()

    # Run prediction
    predictor.predict(batch_size=Config.BATCH_SIZE)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    # Check content of submission file
    with open(Config.SUBMISSION_PATH, "r") as f:
        lines = f.readlines()
        print(f"Submission file contains {len(lines)} lines.")
        if len(lines) > 0:
            print(f"Sample prediction: {lines[0].strip()}")
            # Basic format check: SessionID,label,label...
            parts = lines[0].strip().split(",")
            assert len(parts) >= 1, "Invalid submission format"

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    main()
