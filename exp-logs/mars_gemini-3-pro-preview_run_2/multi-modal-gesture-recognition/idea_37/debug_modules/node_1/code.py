import os
import shutil
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
import library.config as config
import library.utils as utils
from library.data_loader import GestureDataset, collate_fn
from library.model import DCSGCN
from library.loss import DCSGCNLoss
from library.train import train_one_epoch, validate
from library.inference import run_inference


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup Directories and Configuration
    # Define a temporary directory for this demo to avoid overwriting production files
    demo_dir = "./working/demo_run"
    demo_metadata_dir = os.path.join(demo_dir, "metadata")
    demo_cache_dir = os.path.join(demo_dir, "cache")
    demo_checkpoint_dir = os.path.join(demo_dir, "checkpoints")
    demo_submission_dir = os.path.join(demo_dir, "submission")

    for d in [
        demo_metadata_dir,
        demo_cache_dir,
        demo_checkpoint_dir,
        demo_submission_dir,
    ]:
        os.makedirs(d, exist_ok=True)

    print(f"Created demo directories at {demo_dir}")

    # Override Config for Speed
    # We patch the global variables in the config module to redirect paths and reduce load
    config.METADATA_DIR = demo_metadata_dir
    config.WORKING_DIR = demo_dir
    config.CHECKPOINT_DIR = demo_checkpoint_dir
    config.SUBMISSION_DIR = demo_submission_dir

    # Reduce hyperparameters for the demo
    config.HYPERPARAMS["batch_size"] = 4
    config.HYPERPARAMS["num_epochs"] = 1
    config.HYPERPARAMS["hidden_dim"] = 32  # Smaller model for speed
    config.HYPERPARAMS["lstm_layers"] = 1
    config.HYPERPARAMS["tcn_layers"] = 2

    # Set seed for reproducibility
    utils.set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Prepare Data Subset
    print("\n--- Preparing Data Subset ---")
    # Load original metadata and take a small slice
    orig_train_csv = "./metadata/train.csv"
    orig_val_csv = "./metadata/val.csv"
    orig_test_csv = "./metadata/test.csv"

    # Create subset CSVs (5 samples each)
    subset_size = 5

    df_train = pd.read_csv(orig_train_csv).head(subset_size)
    df_train.to_csv(os.path.join(demo_metadata_dir, "train.csv"), index=False)

    df_val = pd.read_csv(orig_val_csv).head(subset_size)
    df_val.to_csv(os.path.join(demo_metadata_dir, "val.csv"), index=False)

    df_test = pd.read_csv(orig_test_csv).head(subset_size)
    df_test.to_csv(os.path.join(demo_metadata_dir, "test.csv"), index=False)

    print(f"Created subset metadata with {subset_size} samples each.")

    # 3. Test Data Loader
    print("\n--- Testing Data Loader ---")
    # Instantiate Dataset
    # We explicitly pass the path to our new subset metadata
    train_ds = GestureDataset(
        metadata_file=os.path.join(demo_metadata_dir, "train.csv"),
        mode="train",
        augment=False,  # Disable augmentation for deterministic check
        load_cached_data=False,  # Force processing
    )

    # Verify Dataset length
    assert (
        len(train_ds) == subset_size
    ), f"Dataset length mismatch. Expected {subset_size}, got {len(train_ds)}"

    # Instantiate DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=config.HYPERPARAMS["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify Batch Keys
    expected_keys = [
        "features",
        "labels",
        "boundaries",
        "mask",
        "lengths",
        "sample_ids",
    ]
    for k in expected_keys:
        assert k in batch, f"Batch missing key: {k}"

    features = batch["features"]
    labels = batch["labels"]
    mask = batch["mask"]

    # Verify Shapes
    # Features: (B, T, D) -> D should be 85 (72 skeleton + 13 MFCC)
    B, T, D = features.shape
    assert B == config.HYPERPARAMS["batch_size"], f"Batch size mismatch. Got {B}"
    assert D == 85, f"Feature dimension mismatch. Expected 85, got {D}"
    assert labels.shape == (
        B,
        T,
    ), f"Labels shape mismatch. Expected ({B}, {T}), got {labels.shape}"
    assert mask.shape == (
        B,
        T,
    ), f"Mask shape mismatch. Expected ({B}, {T}), got {mask.shape}"

    print("Data Loader verification passed.")

    # 4. Test Model
    print("\n--- Testing Model Architecture ---")
    model = DCSGCN().to(device)

    # Move batch to device
    features = features.to(device)
    mask = mask.to(device)

    # Forward Pass
    outputs = model(features, mask)

    # Verify Output Structure
    assert "stage1" in outputs
    assert "stage2" in outputs
    assert "stage3" in outputs

    # Verify Output Shapes for Stage 3
    s3_cls, s3_bnd = outputs["stage3"]
    # cls: (B, T, NumClasses), bnd: (B, T, 1)
    num_classes = config.HYPERPARAMS["num_classes"]
    assert s3_cls.shape == (
        B,
        T,
        num_classes,
    ), f"Class output shape mismatch. Expected ({B}, {T}, {num_classes}), got {s3_cls.shape}"
    assert s3_bnd.shape == (
        B,
        T,
        1,
    ), f"Boundary output shape mismatch. Expected ({B}, {T}, 1), got {s3_bnd.shape}"

    print("Model forward pass verification passed.")

    # 5. Test Loss Function
    print("\n--- Testing Loss Function ---")
    criterion = DCSGCNLoss().to(device)

    labels = labels.to(device)
    boundaries = batch["boundaries"].to(device)

    # Compute Loss
    loss = criterion(outputs, labels, boundaries, mask)

    # Verify Loss
    assert torch.is_tensor(loss), "Loss is not a tensor"
    assert loss.dim() == 0, "Loss is not a scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

    print(f"Loss computation passed. Loss value: {loss.item():.4f}")

    # 6. Test Training Loop
    print("\n--- Testing Training Loop (1 Epoch) ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Run one epoch of training
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Train Epoch Loss: {train_loss:.4f}")

    # Run validation
    # Create val loader
    val_ds = GestureDataset(
        os.path.join(demo_metadata_dir, "val.csv"),
        mode="val",
        augment=False,
        load_cached_data=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.HYPERPARAMS["batch_size"], collate_fn=collate_fn
    )

    val_loss, val_lev = validate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}, Levenshtein Error: {val_lev:.4f}")

    # Save checkpoint for inference test
    torch.save(model.state_dict(), os.path.join(demo_checkpoint_dir, "best_model.pth"))
    print("Training loop verification passed.")

    # 7. Test Inference
    print("\n--- Testing Inference Pipeline ---")

    # Run inference using the saved checkpoint and test subset
    # Note: run_inference uses get_loaders internally which uses config.METADATA_DIR
    # We patched config.METADATA_DIR earlier, so it should pick up our subset test.csv

    # We need to ensure the cache for test data is created or ignored.
    # run_inference calls get_loaders. We need to make sure get_loaders picks up the right file.
    # Since we updated config.METADATA_DIR, get_loaders will look in demo_metadata_dir.

    # However, GestureDataset inside get_loaders will try to load cache.
    # Since we haven't processed test data yet, it will process it now.

    output_csv = os.path.join(demo_submission_dir, "submission.csv")

    run_inference(
        checkpoint_path=os.path.join(demo_checkpoint_dir, "best_model.pth"),
        output_path=output_csv,
        batch_size=config.HYPERPARAMS["batch_size"],
        device=device,
    )

    # Verify Submission File
    assert os.path.exists(output_csv), "Submission file was not created"

    with open(output_csv, "r") as f:
        lines = f.readlines()

    # Check content
    # Should have 5 lines (subset size)
    assert (
        len(lines) == subset_size
    ), f"Submission line count mismatch. Expected {subset_size}, got {len(lines)}"

    # Check format: SessionID,label1,label2...
    sample_line = lines[0].strip()
    parts = sample_line.split(",")
    assert len(parts) >= 1, "Invalid submission format"
    # First part should be sample ID (e.g., SampleXXXXX)
    assert "Sample" in parts[0], f"Invalid Sample ID in submission: {parts[0]}"

    print("Inference pipeline verification passed.")

    # Cleanup
    print("\n--- Cleaning Up ---")
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    print("Demo directory cleaned up.")

    print("\n=== Demonstration Complete: All Systems Operational ===")


if __name__ == "__main__":
    main()
