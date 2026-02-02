import os
import shutil
import pandas as pd
import torch
import numpy as np
import logging

# Import from the provided library files
from library.utils import Config, set_seed, ensure_dirs, get_device
from library.dataset import get_dataloaders, PlantDataset
from library.model import HierarchicalConvNeXt
from library.loss import HierarchicalLoss, get_class_weights
from library.trainer import Trainer
from library.inference import generate_submission


def run_demo():
    print("=== Starting Demonstration of Plant Classification Library ===")

    # 1. Configuration and Setup for Fast Demonstration
    # We override Config attributes to run a small-scale test (Unit & Integration)
    print("\n[Step 1] Configuring for fast demonstration...")

    # Create a separate working directory for this demo to avoid conflicts
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Patch Config
    Config.WORKING_DIR = DEMO_DIR
    Config.OUTPUT_DIR = os.path.join(DEMO_DIR, "submission")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 samples for train/val
    Config.BATCH_SIZE = 8  # Small batch size
    Config.EPOCHS = 1  # Only 1 epoch
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution
    Config.MODEL_NAME = "resnet18"  # Use a lighter model for speed (timm supports this)
    Config.ACCUMULATION_STEPS = 1

    # Create a subset of test.csv to speed up inference/submission generation
    # The original test.csv is large, so we create a dummy one with 20 rows.
    # We read the original test csv to get valid image paths.
    original_test_df = pd.read_csv(Config.TEST_CSV, nrows=20)
    subset_test_csv_path = os.path.join(DEMO_DIR, "test_subset.csv")
    original_test_df.to_csv(subset_test_csv_path, index=False)
    Config.TEST_CSV = subset_test_csv_path
    print(
        f"Created subset test CSV at {Config.TEST_CSV} with {len(original_test_df)} rows."
    )

    ensure_dirs()
    set_seed(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # 2. Verify Data Loading
    print("\n[Step 2] Verifying Data Loading...")
    # load_cached_data=False forces re-computation of taxonomy maps for this run
    train_loader, val_loader, test_loader, maps = get_dataloaders(
        load_cached_data=False
    )

    # Assertions for DataLoaders
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Val loader is empty"
    assert len(test_loader) > 0, "Test loader is empty"

    # Check a single batch structure
    batch = next(iter(train_loader))
    assert "image" in batch
    assert "species" in batch
    assert "genus" in batch
    assert "family" in batch
    # Check shapes: (Batch, Channels, Height, Width)
    assert batch["image"].shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    )
    print("DataLoaders initialized and batch structure verified.")
    print(
        f"Taxonomy Maps Loaded: {maps['num_species']} species, {maps['num_genera']} genera."
    )

    # 3. Verify Model Architecture
    print("\n[Step 3] Verifying Model Architecture...")
    # pretrained=False to avoid potentially slow downloads during this quick check
    # (Trainer will use pretrained=True, assuming environment has cache/internet)
    model = HierarchicalConvNeXt(pretrained=False)
    model.to(device)

    # Forward pass check with dummy data
    dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(device)
    outputs = model(dummy_input)

    assert "species" in outputs
    assert "genus" in outputs
    assert "family" in outputs
    # Check output shapes: (Batch, Num_Classes)
    assert outputs["species"].shape == (2, Config.NUM_CLASSES_SPECIES)
    assert outputs["genus"].shape == (2, Config.NUM_CLASSES_GENUS)
    assert outputs["family"].shape == (2, Config.NUM_CLASSES_FAMILY)
    print("Model forward pass successful. Output shapes correct.")

    # 4. Verify Loss Function
    print("\n[Step 4] Verifying Loss Function...")
    # Generate dummy targets
    dummy_targets = {
        "species": torch.randint(0, Config.NUM_CLASSES_SPECIES, (2,)).to(device),
        "genus": torch.randint(0, Config.NUM_CLASSES_GENUS, (2,)).to(device),
        "family": torch.randint(0, Config.NUM_CLASSES_FAMILY, (2,)).to(device),
    }

    # Compute weights (reads full train.csv but is reasonably fast)
    class_weights = get_class_weights(load_cached_data=True)
    loss_fn = HierarchicalLoss(device, class_weights=class_weights)

    loss, loss_dict = loss_fn(outputs, dummy_targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
    assert "loss_species" in loss_dict
    print(f"Loss computation successful. Total Loss: {loss.item():.4f}")

    # 5. Verify Training Loop (Integration Test)
    print("\n[Step 5] Running Training Loop (Integration Test)...")
    # We use the Trainer class which encapsulates the loop
    trainer = Trainer()

    # This will run for 1 epoch on the subset data (50 samples)
    # It will also run validation and generate submission on the subset test data (20 samples)
    trainer.run_training()

    # 6. Verify Outputs
    print("\n[Step 6] Verifying Outputs...")
    submission_path = os.path.join(Config.OUTPUT_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not generated."

    submission_df = pd.read_csv(submission_path)
    assert (
        len(submission_df) == 20
    ), f"Expected 20 predictions, found {len(submission_df)}"
    assert "Id" in submission_df.columns
    assert "Predicted" in submission_df.columns

    # Check if predictions are valid integers
    assert pd.api.types.is_integer_dtype(
        submission_df["Predicted"]
    ), "Predicted column should be integers"

    print("Submission file verified.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
