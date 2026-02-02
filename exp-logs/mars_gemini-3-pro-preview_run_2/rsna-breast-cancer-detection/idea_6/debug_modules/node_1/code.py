import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shutil

# Ensure the current directory is in the path for imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_dataloaders
from library.model import MultiTaskEfficientNet
from library.engine import train_one_epoch, validate, inference


def run_demo():
    # 1. Setup and Configuration Override
    print("Setting up configuration for demo...")
    seed_everything(42)

    # Override Config for speed and demonstration purposes
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Use 0 workers to avoid multiprocessing overhead in demo
    Config.IMG_SIZE = (128, 128)  # Small image size for fast processing
    Config.MODEL_NAME = "tf_efficientnet_b0.in1k"  # Use a smaller backbone
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Create working directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Loading
    print("\nInitializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # Validate Train Loader
    print("Validating training batch structure...")
    batch = next(iter(train_loader))

    # Check keys
    expected_keys = ["image", "tabular", "target", "aux_birads", "aux_density"]
    for k in expected_keys:
        assert k in batch, f"Missing key {k} in train batch"

    # Check Image Shape
    images = batch["image"]
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), f"Incorrect image shape: {images.shape}"

    # Check Tabular Data
    tabular = batch["tabular"]
    assert "age" in tabular and "machine_id" in tabular, "Missing tabular features"
    assert tabular["view"].shape == (
        Config.BATCH_SIZE,
        6,
    ), "Incorrect one-hot view shape"

    # Check Targets
    targets = batch["target"]
    assert targets.shape[0] == Config.BATCH_SIZE, "Target batch size mismatch"

    print("Data loading verification successful.")

    # 3. Model Instantiation
    print("\nInitializing Model...")
    # Retrieve number of machine IDs from the dataset mapping
    num_machine_ids = len(train_loader.dataset.machine_id_map) + 1  # +1 for unknown

    # Use pretrained=False to avoid downloading weights during the timed run
    model = MultiTaskEfficientNet(num_machine_ids=num_machine_ids, pretrained=False)
    model.to(device)

    # Validate Forward Pass
    print("Validating model forward pass...")
    images = images.to(device)
    tabular_device = {k: v.to(device) for k, v in tabular.items()}

    with torch.no_grad():
        outputs = model(images, tabular_device)

    assert "cancer" in outputs, "Model output missing 'cancer' logits"
    assert "birads" in outputs, "Model output missing 'birads' logits"
    assert "density" in outputs, "Model output missing 'density' logits"
    assert outputs["cancer"].shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Output shape mismatch: {outputs['cancer'].shape}"

    print("Model architecture verification successful.")

    # 4. Training Loop
    print("\nStarting Training Loop Demo...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda")

    # Train for one epoch (on the debug subset)
    train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
    print(f"Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # Validate
    print("Running Validation...")
    val_loss, val_pf1 = validate(model, val_loader, device)
    print(f"Val Loss: {val_loss:.4f} | Val pF1: {val_pf1:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0.0 <= val_pf1 <= 1.0, "Validation pF1 score out of range"

    # 5. Inference
    print("\nRunning Inference...")
    submission_df = inference(model, test_loader, device)

    # Validate Submission
    print("Validating submission file...")
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    # Check structure
    assert "prediction_id" in submission_df.columns, "Missing prediction_id column"
    assert "cancer" in submission_df.columns, "Missing cancer column"

    # Check values
    assert submission_df["cancer"].min() >= 0.0, "Probabilities < 0"
    assert submission_df["cancer"].max() <= 1.0, "Probabilities > 1"

    # In debug mode, test_df is subsampled to 50 rows.
    # The number of unique prediction_ids might be less than 50 if multiple images share an ID.
    print(f"Generated {len(submission_df)} predictions.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
