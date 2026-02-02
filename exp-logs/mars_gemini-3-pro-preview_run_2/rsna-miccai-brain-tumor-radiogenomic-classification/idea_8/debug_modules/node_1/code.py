import os
import sys
import shutil
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library import config, utils, data, model, engine

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting MGMT Classification Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # -------------------------------------------------------------------------
    # Define a separate working directory for this demo to avoid conflicts
    demo_working_dir = "./working/demo_run"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Override config paths and parameters for the demo
    config.WORKING_DIR = demo_working_dir
    config.BEST_MODEL_PATH = os.path.join(demo_working_dir, "best_model.pth")
    config.ROI_CACHE_PATH = os.path.join(demo_working_dir, "roi_cache.parquet")
    config.SUBMISSION_PATH = os.path.join(demo_working_dir, "submission.csv")

    # Reduce compute requirements for speed
    config.BATCH_SIZE = 4
    config.NUM_EPOCHS = 1
    config.NUM_WORKERS = 2

    # Set seed for reproducibility
    engine.set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Configuration configured. Working dir: {config.WORKING_DIR}")
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n--- Verifying Utility Functions ---")

    # Load metadata
    if not os.path.exists(config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {config.TRAIN_METADATA_PATH}")

    df_full = pd.read_csv(config.TRAIN_METADATA_PATH)
    sample_row = df_full.iloc[0]

    # Test DICOM Reading
    flair_dir = os.path.join(config.INPUT_DIR, sample_row["path_FLAIR"])
    flair_files = utils.get_sorted_files(flair_dir)

    if flair_files:
        sample_file = flair_files[len(flair_files) // 2]
        img = utils.read_dicom_robust(sample_file)

        # Assertions
        assert isinstance(img, np.ndarray), "read_dicom_robust must return numpy array"
        assert img.shape == (
            config.IMAGE_SIZE,
            config.IMAGE_SIZE,
        ), f"Expected shape ({config.IMAGE_SIZE}, {config.IMAGE_SIZE}), got {img.shape}"
        assert img.dtype == np.float32, "Image must be float32"
        assert 0.0 <= img.max() <= 1.0, "Image must be normalized to [0, 1]"
        print(f"Verified read_dicom_robust: Shape {img.shape}, Max {img.max():.4f}")

    # Test Anchor Computation
    # This also verifies the logic inside compute_consensus_anchor
    anchor_idx = utils.compute_consensus_anchor(flair_dir)
    assert isinstance(anchor_idx, int), "Anchor index must be an integer"
    assert anchor_idx >= 0, "Anchor index must be non-negative"
    print(f"Verified compute_consensus_anchor: Calculated anchor {anchor_idx}")

    # -------------------------------------------------------------------------
    # 3. Verify Dataset & DataLoader
    # -------------------------------------------------------------------------
    print("\n--- Verifying Dataset & DataLoader ---")

    # Create a small subset for speed (e.g., 8 samples)
    df_subset = df_full.head(8).copy()

    # Save subset metadata for reference (optional)
    subset_csv_path = os.path.join(demo_working_dir, "train_subset.csv")
    df_subset.to_csv(subset_csv_path, index=False)

    # Instantiate Dataset
    # load_cached_anchors=True triggers utils.get_roi_anchors which creates the cache
    train_dataset = data.MGMTDataset(
        df_subset, transforms=data.get_transforms("train"), load_cached_anchors=True
    )

    # Verify Cache Creation
    assert os.path.exists(config.ROI_CACHE_PATH), "ROI cache file was not created"

    # Verify __getitem__
    volume, target = train_dataset[0]

    # Expected shape: (12, 256, 256) -> 4 modalities * 3 slices
    assert volume.shape == (
        12,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
    ), f"Unexpected volume shape: {volume.shape}"
    assert isinstance(target, torch.Tensor), "Target must be a tensor"
    print(f"Verified Dataset item: Volume {volume.shape}, Target {target.item()}")

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Use 0 for simple debugging/demo
    )

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n--- Verifying Model Architecture ---")

    net = model.AsymmetricEfficientNet().to(device)

    # Create dummy input based on verified dataset output
    dummy_input = torch.randn(
        config.BATCH_SIZE, 12, config.IMAGE_SIZE, config.IMAGE_SIZE
    ).to(device)

    # Forward pass
    logits = net(dummy_input)

    assert logits.shape == (
        config.BATCH_SIZE,
        1,
    ), f"Output shape mismatch. Expected ({config.BATCH_SIZE}, 1), got {logits.shape}"
    print(f"Verified Model Forward Pass: Output shape {logits.shape}")

    # -------------------------------------------------------------------------
    # 5. Demonstrate Training & Validation (Engine)
    # -------------------------------------------------------------------------
    print("\n--- Demonstrating Training Loop (Engine) ---")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Train One Epoch
    print("Running train_one_epoch...")
    train_loss, train_auc = engine.train_one_epoch(
        net, train_loader, criterion, optimizer, device
    )

    assert not np.isnan(train_loss), "Training loss is NaN"
    print(f"Train Result -> Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")

    # Validate (using the same loader for demo purposes)
    print("Running validate...")
    val_loss, val_auc = engine.validate(net, train_loader, criterion, device)

    assert not np.isnan(val_loss), "Validation loss is NaN"
    print(f"Val Result   -> Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # Save the "best" model (simulating the loop)
    torch.save(net.state_dict(), config.BEST_MODEL_PATH)
    print(f"Model saved to {config.BEST_MODEL_PATH}")

    # -------------------------------------------------------------------------
    # 6. Demonstrate Inference (TTA)
    # -------------------------------------------------------------------------
    print("\n--- Demonstrating Inference with TTA ---")

    # Use the test metadata logic (simulated with subset)
    # We strip the target column to mimic test data structure
    df_test_sim = df_subset.drop(columns=["MGMT_value"])
    test_dataset = data.MGMTDataset(
        df_test_sim, transforms=data.get_transforms("valid"), load_cached_anchors=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Run Prediction
    predictions = engine.predict_tta(net, test_loader, device)

    # Verify Predictions
    assert len(predictions) == len(
        df_test_sim
    ), "Number of predictions does not match number of samples"
    sample_id = df_test_sim.iloc[0]["BraTS21ID"]
    assert sample_id in predictions, f"Prediction for ID {sample_id} missing"
    assert 0.0 <= predictions[sample_id] <= 1.0, "Prediction probability out of range"

    print(f"Generated {len(predictions)} predictions successfully.")
    print(f"Sample Prediction (ID {sample_id}): {predictions[sample_id]:.4f}")

    # -------------------------------------------------------------------------
    # 7. Generate Submission File
    # -------------------------------------------------------------------------
    print("\n--- Generating Submission File ---")

    submission_df = pd.DataFrame(
        list(predictions.items()), columns=["BraTS21ID", "MGMT_value"]
    )
    submission_df = submission_df.sort_values("BraTS21ID")
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)

    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not created"
    print(f"Submission saved to {config.SUBMISSION_PATH}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
