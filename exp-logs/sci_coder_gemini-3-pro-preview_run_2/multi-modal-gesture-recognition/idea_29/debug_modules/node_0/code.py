import os
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Import from the provided library
from library import config, utils, data_loader, model, losses, train, inference


def main():
    # 1. Setup & Configuration Override
    print("=== Setting up Demo Environment ===")

    # Define demo directories
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    METADATA_DIR = os.path.join(DEMO_DIR, "metadata")
    SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    CACHE_DIR = os.path.join(
        DEMO_DIR, "cache"
    )  # Although dataset uses config.WORKING_DIR

    os.makedirs(METADATA_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Create Mini Metadata (Subset of real data)
    # We take the first 10 training samples, 5 validation, and 5 test samples
    print("Creating mini metadata subsets...")

    try:
        train_full = pd.read_csv(config.TRAIN_METADATA_PATH)
        val_full = pd.read_csv(config.VAL_METADATA_PATH)
        test_full = pd.read_csv(config.TEST_METADATA_PATH)

        mini_train_path = os.path.join(METADATA_DIR, "train.csv")
        mini_val_path = os.path.join(METADATA_DIR, "val.csv")
        mini_test_path = os.path.join(METADATA_DIR, "test.csv")

        train_full.head(10).to_csv(mini_train_path, index=False)
        val_full.head(5).to_csv(mini_val_path, index=False)
        test_full.head(5).to_csv(mini_test_path, index=False)

        print(f"Mini metadata created at {METADATA_DIR}")

    except FileNotFoundError as e:
        print(f"Error reading original metadata: {e}")
        return

    # Monkey-patch config to use demo paths and settings
    print("Patching configuration for speed...")
    config.WORKING_DIR = DEMO_DIR
    config.SUBMISSION_DIR = SUBMISSION_DIR
    config.SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    config.TRAIN_METADATA_PATH = mini_train_path
    config.VAL_METADATA_PATH = mini_val_path
    config.TEST_METADATA_PATH = mini_test_path

    config.NUM_EPOCHS = 1
    config.BATCH_SIZE = 2
    config.NUM_WORKERS = 0  # Use main thread for demo stability
    config.EARLY_STOPPING_PATIENCE = 1

    # Set seeds
    utils.set_seed(42)

    # 2. Verify Utilities
    print("\n=== Verifying Utilities ===")
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 4]
    dist = utils.levenshtein_distance(seq1, seq2)
    print(f"Levenshtein distance between {seq1} and {seq2}: {dist}")
    assert dist == 1, "Levenshtein distance calculation incorrect"

    error_rate = utils.compute_levenshtein([seq1], [seq2])
    print(f"Error rate: {error_rate}")
    assert np.isclose(error_rate, 1 / 3), "Error rate calculation incorrect"

    # 3. Verify Data Loading
    print("\n=== Verifying Data Loader ===")
    # Initialize Dataset
    ds = data_loader.GestureDataset(
        config.TRAIN_METADATA_PATH, is_train=True, load_cached_data=False
    )
    print(f"Dataset length: {len(ds)}")
    assert len(ds) == 10, "Dataset length mismatch"

    # Check single item
    item = ds[0]
    features = item["features"]
    labels = item["labels"]
    print(f"Sample feature shape: {features.shape}")  # (T, 85)
    print(f"Sample label shape: {labels.shape}")  # (T,)

    assert (
        features.shape[1] == config.INPUT_DIM
    ), f"Feature dim mismatch. Expected {config.INPUT_DIM}, got {features.shape[1]}"
    assert (
        features.shape[0] == labels.shape[0]
    ), "Temporal dimension mismatch between features and labels"

    # Check Collate
    loader = torch.utils.data.DataLoader(
        ds, batch_size=config.BATCH_SIZE, collate_fn=data_loader.collate_fn
    )
    batch = next(iter(loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Batch features shape: {batch['features'].shape}")  # (B, T_max, D)
    print(f"Batch mask shape: {batch['mask'].shape}")  # (B, T_max)

    assert batch["features"].shape[0] == config.BATCH_SIZE
    assert batch["features"].shape[2] == config.INPUT_DIM

    # 4. Verify Model & Loss
    print("\n=== Verifying Model & Loss ===")
    device = torch.device("cpu")  # Force CPU for demo simplicity
    net = model.CASGCN().to(device)
    criterion = losses.MultiStageLoss(device=device)

    # Forward Pass
    outputs = net(batch["features"], batch["mask"], batch["lengths"])
    print("Model forward pass successful.")
    print(f"Output stages: {outputs.keys()}")

    assert "stage1" in outputs and "stage2" in outputs and "stage3" in outputs
    s3_cls, s3_bnd = outputs["stage3"]
    print(f"Stage 3 Class Logits: {s3_cls.shape}")  # (B, T, C)

    assert s3_cls.shape[2] == config.NUM_CLASSES

    # Loss Calculation
    loss, metrics = criterion(outputs, batch)
    print(f"Calculated Loss: {loss.item()}")
    print(f"Metrics: {metrics}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss > 0, "Loss should be positive"

    # 5. Run Training Loop
    print("\n=== Running Training Loop (1 Epoch) ===")
    # This calls the library function which uses our patched config
    train.run_training(
        num_epochs=config.NUM_EPOCHS,
        batch_size=config.BATCH_SIZE,
        learning_rate=1e-3,
        load_cached_data=True,  # Will use the cache generated by ds init above or create new
    )

    checkpoint_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(
        checkpoint_path
    ), "Training failed to produce 'best_model.pth'"
    print(f"Checkpoint created at {checkpoint_path}")

    # 6. Run Inference
    print("\n=== Running Inference ===")
    inference.run_inference(
        checkpoint_path=checkpoint_path,
        output_path=config.SUBMISSION_FILE_PATH,
        batch_size=config.BATCH_SIZE,
        load_cached_data=False,  # Force re-process for test set
        device="cpu",
    )

    assert os.path.exists(
        config.SUBMISSION_FILE_PATH
    ), "Inference failed to produce submission file"

    # 7. Verify Submission
    print("\n=== Verifying Submission File ===")
    with open(config.SUBMISSION_FILE_PATH, "r") as f:
        lines = f.readlines()

    print(f"Submission lines: {len(lines)}")
    # We expect 5 lines because we used head(5) for test set
    assert len(lines) == 5, f"Expected 5 predictions, got {len(lines)}"

    # Check format of first line
    sample_line = lines[0].strip()
    print(f"Sample prediction: {sample_line}")
    parts = sample_line.split(",")
    assert len(parts) >= 1, "Invalid submission format"
    # SessionID check
    assert parts[0].startswith("Sample"), "Invalid SessionID in submission"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    main()
