import os
import pandas as pd
import torch
import numpy as np
import warnings
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library import config, utils, data_loader, model, trainer, inference


def run_demonstration():
    print("============================================================")
    print("       Gesture Recognition Library Demonstration            ")
    print("============================================================")

    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides
    # -------------------------------------------------------------------------
    print("\n[1] Setting up configuration for fast demonstration...")

    # Define a temporary directory for this demo run
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override config paths to point to the demo directory
    config.WORKING_DIR = DEMO_DIR
    config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    config.SUBMISSION_PATH = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    # Create necessary subdirectories
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Override hyperparameters for speed (Tiny Model, Few Epochs)
    config.HIDDEN_DIM = 32  # Reduced from 256
    config.LSTM_LAYERS = 1  # Reduced from 2
    config.NUM_LAYERS = 2  # Reduced TCN layers from 10
    config.NUM_F_MAPS = 16  # Reduced feature maps
    config.EPOCHS = 2  # Only 2 epochs
    config.BATCH_SIZE = 2  # Small batch size
    config.EARLY_STOPPING_PATIENCE = 2

    print(f"    Working Directory: {config.WORKING_DIR}")
    print(f"    Epochs: {config.EPOCHS}, Batch Size: {config.BATCH_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Prepare Data Subsets
    # -------------------------------------------------------------------------
    print("\n[2] Creating data subsets from metadata...")

    # Load the first 6 rows of existing metadata to create a tiny dataset
    # This ensures we use valid file paths from ./input without scanning everything
    try:
        train_full = pd.read_csv("./metadata/train.csv")
        val_full = pd.read_csv("./metadata/val.csv")
        test_full = pd.read_csv("./metadata/test.csv")
    except FileNotFoundError as e:
        print(f"    Error: Metadata files not found. {e}")
        return

    subset_size = 6
    train_subset = train_full.head(subset_size)
    val_subset = val_full.head(subset_size)
    test_subset = test_full.head(subset_size)

    # Save subsets to the demo directory
    subset_train_path = os.path.join(DEMO_DIR, "train_subset.csv")
    subset_val_path = os.path.join(DEMO_DIR, "val_subset.csv")
    subset_test_path = os.path.join(DEMO_DIR, "test_subset.csv")

    train_subset.to_csv(subset_train_path, index=False)
    val_subset.to_csv(subset_val_path, index=False)
    test_subset.to_csv(subset_test_path, index=False)

    # Update config to point to these new subset metadata files
    config.TRAIN_METADATA_PATH = subset_train_path
    config.VAL_METADATA_PATH = subset_val_path
    config.TEST_METADATA_PATH = subset_test_path

    print(f"    Created subsets with {subset_size} samples each.")

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    print("\n[3] Processing data and creating DataLoaders...")

    # Set seed for reproducibility
    utils.set_seed(config.SEED)

    # Load data (load_cached_data=False forces processing of our new subsets)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders(
        load_cached_data=False
    )

    # Verify DataLoader outputs
    batch = next(iter(train_loader))
    print(f"    Batch Keys: {list(batch.keys())}")

    assert "features" in batch, "Batch missing 'features'"
    assert "targets" in batch, "Batch missing 'targets'"
    assert "mask" in batch, "Batch missing 'mask'"

    features = batch["features"]
    targets = batch["targets"]
    mask = batch["mask"]

    # Check shapes: [Batch, Time, Dim]
    print(f"    Features Shape: {features.shape}")
    print(f"    Targets Shape: {targets.shape}")

    assert features.dim() == 3, "Features should be 3D [Batch, Time, Dim]"
    assert (
        features.shape[2] == config.INPUT_DIM
    ), f"Feature dim mismatch. Expected {config.INPUT_DIM}, got {features.shape[2]}"
    assert targets.dim() == 2, "Targets should be 2D [Batch, Time]"

    # -------------------------------------------------------------------------
    # 4. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[4] Initializing DSR-CRCN Model and running forward pass...")

    device = utils.get_device()
    net = model.DSR_CRCN().to(device)

    # Move batch to device
    features = features.to(device)
    mask = mask.to(device)

    # Forward pass
    logits0, logits1, logits2 = net(features, mask)

    print(
        f"    Logits Output Shapes: Stage0={logits0.shape}, Stage1={logits1.shape}, Stage2={logits2.shape}"
    )

    # Validation
    B, T, C = logits2.shape
    assert (
        C == config.NUM_CLASSES
    ), f"Output classes mismatch. Expected {config.NUM_CLASSES}, got {C}"
    assert B == features.shape[0], "Batch size mismatch in output"
    assert T == features.shape[1], "Temporal length mismatch in output"

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Loop (2 Epochs)...")

    # Initialize Trainer
    trainer_instance = trainer.Trainer(net, train_loader, val_loader)

    # Run training
    trainer_instance.fit()

    # Verify checkpoint creation
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"    Success: Model checkpoint saved at {best_model_path}")
    else:
        raise AssertionError("Model checkpoint was not created!")

    # -------------------------------------------------------------------------
    # 6. Inference
    # -------------------------------------------------------------------------
    print("\n[6] Running Inference on Test Subset...")

    # Run inference (uses the best_model.pth generated above)
    # load_cached_data=False ensures it uses the test subset cache we just created
    inference.generate_submission(load_cached_data=True)

    # Verify submission file
    if os.path.exists(config.SUBMISSION_PATH):
        print(f"    Success: Submission file saved at {config.SUBMISSION_PATH}")

        # Check content
        with open(config.SUBMISSION_PATH, "r") as f:
            lines = f.readlines()
            print(f"    Generated {len(lines)} predictions.")
            if len(lines) > 0:
                print(f"    Sample prediction: {lines[0].strip()}")
    else:
        raise AssertionError("Submission file was not created!")

    # -------------------------------------------------------------------------
    # 7. Metric Logic Verification
    # -------------------------------------------------------------------------
    print("\n[7] Verifying Metric Logic (Levenshtein Distance)...")

    # Test Case:
    # Target: [1, 2, 3]
    # Prediction: [1, 3] (Deletion of '2') -> Distance = 1
    # Metric = Distance / Target_Length = 1 / 3 = 0.333...

    dummy_preds = [[1, 3]]
    dummy_targs = [[1, 2, 3]]

    score = utils.compute_levenshtein(dummy_preds, dummy_targs)
    expected_score = 1.0 / 3.0

    print(f"    Calculated Score: {score:.4f}")
    print(f"    Expected Score:   {expected_score:.4f}")

    assert abs(score - expected_score) < 1e-5, "Metric calculation is incorrect!"

    print("\n============================================================")
    print("       Demonstration Completed Successfully                 ")
    print("============================================================")


if __name__ == "__main__":
    run_demonstration()
