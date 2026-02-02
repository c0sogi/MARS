import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.model as model_lib
import library.losses as losses_lib
import library.trainer as trainer_lib
import library.inference as inference_lib

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting BMGCN Pipeline Demo ===")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # -------------------------------------------------------------------------
    print("\n[1] Setting up configuration for demo...")

    # Define a separate working directory for this demo to avoid conflicts
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override global config paths
    config.WORKING_DIR = DEMO_DIR
    config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Override training hyperparameters for speed
    config.TRAIN_CONFIG["num_epochs"] = 1
    config.TRAIN_CONFIG["batch_size"] = 4
    config.TRAIN_CONFIG["debug_subset_size"] = 12  # Very small subset
    config.TRAIN_CONFIG["num_workers"] = (
        0  # Avoid multiprocessing overhead for small data
    )

    # Set seed for reproducibility
    utils.set_seed(config.SEED)
    device = utils.get_device()
    print(f"    Device: {device}")
    print(f"    Working Directory: {config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Prepare Metadata Subsets
    # -------------------------------------------------------------------------
    print("\n[2] Preparing metadata subsets...")

    # Create a small subset of test metadata for quick inference
    meta_dir = os.path.join(DEMO_DIR, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    full_test_df = pd.read_csv(config.TEST_METADATA_PATH)
    subset_test_df = full_test_df.head(6)  # Take 6 samples
    subset_test_path = os.path.join(meta_dir, "test.csv")
    subset_test_df.to_csv(subset_test_path, index=False)

    print(
        f"    Created subset test metadata at {subset_test_path} ({len(subset_test_df)} samples)"
    )

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    print("\n[3] Testing Data Loading...")

    # Load dataloaders (Train/Val use debug_subset_size from config)
    train_loader, val_loader, _ = data_loader.get_dataloaders(
        train_path=config.TRAIN_METADATA_PATH,
        val_path=config.VAL_METADATA_PATH,
        test_path=subset_test_path,  # Not used here but good to verify path works
        batch_size=config.TRAIN_CONFIG["batch_size"],
        num_workers=config.TRAIN_CONFIG["num_workers"],
        limit=config.TRAIN_CONFIG["debug_subset_size"],
    )

    print(f"    Train Loader batches: {len(train_loader)}")
    print(f"    Val Loader batches: {len(val_loader)}")

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    features = batch["features"]
    cls_labels = batch["cls_labels"]
    bnd_labels = batch["bnd_labels"]
    mask = batch["mask"]

    print(
        f"    Batch Shapes -> Features: {features.shape}, Cls: {cls_labels.shape}, Mask: {mask.shape}"
    )

    # Assertions
    assert features.dim() == 3, "Features should be (B, T, D)"
    assert (
        features.shape[-1] == config.INPUT_DIM
    ), f"Feature dim should be {config.INPUT_DIM}"
    assert cls_labels.dim() == 2, "Class labels should be (B, T)"
    assert mask.dtype == torch.bool, "Mask should be boolean"

    # -------------------------------------------------------------------------
    # 4. Model Instantiation & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[4] Testing Model Architecture...")

    model = model_lib.BMGCN().to(device)

    # Move batch to device
    features = features.to(device)
    mask = mask.to(device)

    # Forward pass
    outputs = model(features, mask)

    # Verify outputs
    assert "stage1" in outputs
    assert "stage2" in outputs
    assert "stage3" in outputs

    s3_out = outputs["stage3"]
    assert "cls_probs" in s3_out
    assert s3_out["cls_probs"].shape == (
        features.shape[0],
        features.shape[1],
        config.NUM_CLASSES,
    )

    print("    Forward pass successful. Output shapes verified.")

    # -------------------------------------------------------------------------
    # 5. Loss Computation
    # -------------------------------------------------------------------------
    print("\n[5] Testing Loss Function...")

    criterion = losses_lib.CombinedSegmentationLoss(config.TRAIN_CONFIG)

    # Prepare targets
    targets = {
        "cls_labels": cls_labels.to(device),
        "bnd_labels": bnd_labels.to(device),
        "mask": mask,
    }

    loss, stats = criterion(outputs, targets)

    print(f"    Total Loss: {loss.item():.4f}")
    print(f"    Loss Components: {list(stats.keys())}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    # -------------------------------------------------------------------------
    # 6. Training Loop
    # -------------------------------------------------------------------------
    print("\n[6] Testing Training Loop...")

    trainer = trainer_lib.Trainer(model, device, config.TRAIN_CONFIG)
    checkpoint_path = os.path.join(DEMO_DIR, "checkpoints", "best_model.pth")

    # Run fit (1 epoch as configured)
    trainer.fit(
        train_loader,
        val_loader,
        epochs=config.TRAIN_CONFIG["num_epochs"],
        patience=1,
        checkpoint_path=checkpoint_path,
    )

    assert os.path.exists(checkpoint_path), "Checkpoint file was not created"
    print(f"    Training complete. Checkpoint saved to {checkpoint_path}")

    # -------------------------------------------------------------------------
    # 7. Inference & Submission Generation
    # -------------------------------------------------------------------------
    print("\n[7] Testing Inference Engine...")

    inference_engine = inference_lib.InferenceEngine(checkpoint_path, device)
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    # Run inference on the subset test data
    inference_engine.run(
        test_metadata_path=subset_test_path,
        output_path=submission_path,
        batch_size=2,
        num_workers=0,
    )

    assert os.path.exists(submission_path), "Submission file was not created"

    # Verify submission content
    print("    Verifying submission content:")
    with open(submission_path, "r") as f:
        lines = f.readlines()
        print(f"    Total lines in submission: {len(lines)}")
        if len(lines) > 0:
            print(f"    First line: {lines[0].strip()}")

            # Check format: SessionID,label1,label2...
            parts = lines[0].strip().split(",")
            assert len(parts) >= 1, "Line should have at least SessionID"
            assert parts[0].startswith("Sample"), "SessionID should start with Sample"

            # If there are predictions, check they are integers
            if len(parts) > 1 and parts[1]:
                try:
                    int(parts[1])
                except ValueError:
                    raise AssertionError("Predicted labels should be integers")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
