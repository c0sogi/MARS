import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Import provided library modules
from library import config, utils, dataset, model, trainer


def main():
    print("Starting library demonstration...")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides for Speed
    # -------------------------------------------------------------------------
    # We define a separate working directory for this demo to avoid conflicts
    # and to ensure we trigger recalculation of cached files (mappings, weights).
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override config paths and parameters to run a fast, small-scale test
    config.WORKING_DIR = DEMO_DIR
    config.SUBMISSION_DIR = DEMO_DIR
    config.MAPPING_CACHE_PATH = os.path.join(DEMO_DIR, "category_mappings.parquet")
    config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "model_best.pth")
    config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Override training hyperparameters
    config.NUM_EPOCHS = 1
    config.BATCH_SIZE = 8
    config.NUM_WORKERS = 2

    # Create subset CSVs to simulate a tiny dataset
    print("Creating dataset subsets for rapid demonstration...")

    # Load original metadata
    full_train = pd.read_csv(config.TRAIN_CSV)
    full_val = pd.read_csv(config.VAL_CSV)
    full_test = pd.read_csv(config.TEST_CSV)

    # Sample 50 rows for each split
    subset_size = 50
    train_subset = full_train.head(subset_size).copy()
    # Use the same subset for validation to ensure classes are present in the mapping
    # This prevents RuntimeWarnings about invalid values and ensures valid evaluation
    val_subset = train_subset.copy()
    test_subset = full_test.head(subset_size).copy()

    # Save subsets to the demo directory
    subset_train_path = os.path.join(DEMO_DIR, "train_subset.csv")
    subset_val_path = os.path.join(DEMO_DIR, "val_subset.csv")
    subset_test_path = os.path.join(DEMO_DIR, "test_subset.csv")

    train_subset.to_csv(subset_train_path, index=False)
    val_subset.to_csv(subset_val_path, index=False)
    test_subset.to_csv(subset_test_path, index=False)

    # Point config to these new subset files
    config.TRAIN_CSV = subset_train_path
    config.VAL_CSV = subset_val_path
    config.TEST_CSV = subset_test_path

    print("Configuration updated for demo run.")

    # -------------------------------------------------------------------------
    # 2. Verify Utils
    # -------------------------------------------------------------------------
    print("\nVerifying library.utils...")
    utils.seed_everything(42)

    # Test AverageMeter
    meter = utils.AverageMeter()
    meter.update(10, n=1)
    meter.update(20, n=1)
    assert meter.avg == 15.0, f"AverageMeter failed: expected 15.0, got {meter.avg}"

    # Test Metric Calculation
    y_true = [0, 1, 2, 0]
    y_pred = [0, 1, 2, 0]
    score = utils.calculate_metrics(y_true, y_pred)
    assert score == 1.0, f"Metric calculation failed: expected 1.0, got {score}"
    print("Utils verified.")

    # -------------------------------------------------------------------------
    # 3. Verify Dataset & DataLoader
    # -------------------------------------------------------------------------
    print("\nVerifying library.dataset...")

    # This will trigger mapping generation and weight calculation for the subset
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        load_cached_data=False,  # Force reload for the subset
    )

    # Verify Train Loader
    assert len(train_loader) > 0, "Train loader is empty"
    images, species_labels, genus_labels = next(iter(train_loader))

    # Check shapes
    # Image: (Batch, 3, H, W)
    assert images.dim() == 4, "Image tensor has incorrect dimensions"
    assert images.shape[1] == 3, "Image tensor does not have 3 channels"
    assert (
        images.shape[2:] == config.IMG_SIZE
    ), f"Image size mismatch: {images.shape[2:]}"
    assert species_labels.shape[0] == config.BATCH_SIZE, "Label batch size mismatch"

    print(f"DataLoader verified. Batch shape: {images.shape}")

    # -------------------------------------------------------------------------
    # 4. Verify Model
    # -------------------------------------------------------------------------
    print("\nVerifying library.model...")

    # Retrieve mapping info to init model correctly
    _, num_species, num_genus = config.get_mappings(load_cached=True)

    # Instantiate model
    net = model.HierarchicalResNet(
        num_species=num_species,
        num_genus=num_genus,
        backbone_name=config.BACKBONE,
        pretrained=False,  # False for speed in demo, logic remains same
    )
    net.eval()

    # Move to CPU for simple verification (or GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    dummy_input = images.to(device)

    # Forward pass
    with torch.no_grad():
        # Passing species_label=None simulates inference
        species_logits, genus_logits = net(dummy_input, species_label=None)

    # Verify Output Shapes
    assert species_logits.shape == (
        config.BATCH_SIZE,
        num_species,
    ), f"Species logits shape mismatch: {species_logits.shape}"
    assert genus_logits.shape == (
        config.BATCH_SIZE,
        num_genus,
    ), f"Genus logits shape mismatch: {genus_logits.shape}"

    print(
        f"Model verified. Output shapes: Species {species_logits.shape}, Genus {genus_logits.shape}"
    )

    # -------------------------------------------------------------------------
    # 5. Verify Trainer (Training Loop)
    # -------------------------------------------------------------------------
    print("\nVerifying library.trainer...")

    # Re-initialize model with pretrained=True (as in real scenario) and move to device
    net = model.HierarchicalResNet(
        num_species=num_species,
        num_genus=num_genus,
        backbone_name=config.BACKBONE,
        pretrained=True,
    )

    # Initialize Trainer
    demo_trainer = trainer.Trainer(net, train_loader, val_loader, device=device)

    # Run training for 1 epoch (as set in config override)
    best_f1 = demo_trainer.fit(num_epochs=config.NUM_EPOCHS)

    # Verify artifacts
    assert os.path.exists(
        config.MODEL_SAVE_PATH
    ), "Best model checkpoint was not saved."
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device)
    assert "state_dict" in checkpoint, "Checkpoint missing state_dict"

    print(f"Training cycle verified. Best F1: {best_f1}")

    # -------------------------------------------------------------------------
    # 6. Verify Submission Generation
    # -------------------------------------------------------------------------
    print("\nVerifying submission generation...")

    # Load the best model state
    net.load_state_dict(checkpoint["state_dict"])

    # Generate submission
    trainer.generate_submission(net, test_loader, device=device)

    # Verify output file
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not found"

    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    assert list(sub_df.columns) == ["Id", "Predicted"], "Submission columns mismatch"
    assert len(sub_df) == len(
        test_subset
    ), f"Submission length mismatch: expected {len(test_subset)}, got {len(sub_df)}"

    print("Submission generation verified.")
    print("\nAll library components demonstrated successfully.")


if __name__ == "__main__":
    main()
