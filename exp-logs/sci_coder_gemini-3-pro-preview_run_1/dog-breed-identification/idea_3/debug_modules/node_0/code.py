import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library import utils, dataset, model, engine


def main():
    print("=== Starting Dog Breed Classification Demo ===")

    # 1. Setup and Reproducibility
    # Set seed for deterministic behavior
    utils.set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading and Processing
    print("\n[Data] Loading metadata...")
    # Load full metadata
    train_df_full = dataset.load_data("train")
    val_df_full = dataset.load_data("val")
    test_df_full = dataset.load_data("test")

    # Create small subsets for speed optimization
    # We use a small number of samples to demonstrate functionality quickly
    train_subset = train_df_full.iloc[:32].copy()
    val_subset = val_df_full.iloc[:16].copy()
    test_subset = test_df_full.iloc[:16].copy()

    print(
        f"Subset sizes -> Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}"
    )

    # Instantiate Datasets
    # We use the transforms defined in dataset.py
    train_ds = dataset.DogDataset(
        train_subset, transform=dataset.get_transforms("train"), mode="train"
    )
    val_ds = dataset.DogDataset(
        val_subset, transform=dataset.get_transforms("val"), mode="val"
    )
    test_ds = dataset.DogDataset(
        test_subset, transform=dataset.get_transforms("test"), mode="test"
    )

    # Instantiate DataLoaders
    # Using a small batch size compatible with the subset size
    batch_size = 8
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Verify Data Batch
    print("[Data] Verifying batch shapes...")
    images, labels = next(iter(train_loader))

    # Expected: (Batch, 3, 224, 224)
    expected_img_shape = (batch_size, 3, Config.CROP_SIZE, Config.CROP_SIZE)
    assert (
        images.shape == expected_img_shape
    ), f"Image shape mismatch. Got {images.shape}, expected {expected_img_shape}"
    assert labels.shape == (
        batch_size,
    ), f"Label shape mismatch. Got {labels.shape}, expected {(batch_size,)}"
    print("Batch verification passed.")

    # 3. Model Initialization
    print("\n[Model] Building model...")
    # We set pretrained=False to avoid downloading weights during this demo run.
    # In a real training scenario, pretrained=True is preferred.
    net = model.build_model(pretrained=False)
    net = net.to(device)

    # Verify Forward Pass
    print("[Model] Verifying forward pass...")
    with torch.no_grad():
        dummy_input = images.to(device)
        outputs = net(dummy_input)

    # Expected output: (Batch, Num Classes)
    assert outputs.shape == (
        batch_size,
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Got {outputs.shape}, expected {(batch_size, Config.NUM_CLASSES)}"
    print("Forward pass verification passed.")

    # 4. Training Loop Execution
    print("\n[Engine] Starting training simulation...")
    # We override epochs to 1 for speed
    engine.run_training(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        phase1_epochs=1,
        phase2_epochs=1,
        patience=1,
    )

    # Verify Checkpoint
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise AssertionError(f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}")
    print(f"Checkpoint verified at {Config.MODEL_SAVE_PATH}")

    # 5. Submission Generation
    print("\n[Engine] Generating submission...")
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    engine.generate_submission(
        model=net, test_loader=test_loader, device=device, output_path=submission_path
    )

    # Verify Submission File
    if not os.path.exists(submission_path):
        raise AssertionError("Submission file was not created.")

    sub_df = pd.read_csv(submission_path)
    print(f"Submission generated with shape: {sub_df.shape}")

    # Verify Columns (id + 120 breeds)
    _, classes = dataset.get_label_mapping()
    expected_cols = ["id"] + list(classes)

    # Check if columns match (order matters for submission)
    if list(sub_df.columns) != expected_cols:
        raise AssertionError("Submission columns do not match the expected format.")

    # Check row count
    if len(sub_df) != len(test_subset):
        raise AssertionError(
            f"Submission row count {len(sub_df)} does not match test set size {len(test_subset)}."
        )

    print("Submission verification passed.")

    # 6. Metric Calculation Demo
    print("\n[Metrics] Demonstrating Log Loss calculation...")
    # Use validation labels and random probabilities to test the metric function
    y_true = val_subset["label"].values

    # Create random probabilities (normalized)
    y_pred = np.random.rand(len(val_subset), Config.NUM_CLASSES)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)

    loss = utils.calculate_log_loss(y_true, y_pred)
    print(f"Calculated Log Loss (Random Predictions): {loss:.4f}")

    # Basic sanity check: Loss should be positive
    assert loss > 0, "Log loss should be positive."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
