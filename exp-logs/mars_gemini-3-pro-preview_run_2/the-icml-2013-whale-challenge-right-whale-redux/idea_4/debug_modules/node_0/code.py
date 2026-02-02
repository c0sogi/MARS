import os
import pandas as pd
import numpy as np
import torch
import warnings

# Import from the provided library
from library import config, utils, model, dataset, train

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Library Usage Demonstration ===")

    # ---------------------------------------------------------
    # 1. Configuration & Setup for Fast Execution
    # ---------------------------------------------------------
    print("\n[Step 1] Configuring environment for rapid testing...")

    # Override config parameters to use a small subset and run quickly
    config.EPOCHS = 1
    config.BATCH_SIZE = 4
    config.NUM_WORKERS = 0  # Use main thread for simplicity/speed in demo
    config.PRETRAINED = (
        False  # Disable pretrained weights to avoid download overhead/errors
    )
    config.PATIENCE = 1
    config.SEED = 42

    # Define temporary paths for this demo
    demo_dir = "./working/demo_run"
    os.makedirs(demo_dir, exist_ok=True)

    # Override metadata paths in config
    config.TRAIN_METADATA = os.path.join(demo_dir, "train_subset.csv")
    config.VAL_METADATA = os.path.join(demo_dir, "val_subset.csv")
    config.TEST_METADATA = os.path.join(demo_dir, "test_subset.csv")

    # Override cache paths in config to avoid overwriting real training data
    config.TRAIN_DATA_CACHE = os.path.join(demo_dir, "train_data.npy")
    config.TRAIN_LABELS_CACHE = os.path.join(demo_dir, "train_labels.npy")
    config.VAL_DATA_CACHE = os.path.join(demo_dir, "val_data.npy")
    config.VAL_LABELS_CACHE = os.path.join(demo_dir, "val_labels.npy")
    config.TEST_DATA_CACHE = os.path.join(demo_dir, "test_data.npy")
    config.TEST_CLIPS_CACHE = os.path.join(demo_dir, "test_clips.npy")

    # Override submission path and working dir
    config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")
    config.WORKING_DIR = demo_dir

    # Set seed for reproducibility
    utils.set_seed(config.SEED)

    # ---------------------------------------------------------
    # 2. Create Data Subsets
    # ---------------------------------------------------------
    print("\n[Step 2] Creating small data subsets from original metadata...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Create subsets (10 samples each)
    subset_size = 10
    train_subset = orig_train.head(subset_size)
    val_subset = orig_val.head(subset_size)
    test_subset = orig_test.head(subset_size)

    # Save subsets to the overridden paths
    train_subset.to_csv(config.TRAIN_METADATA, index=False)
    val_subset.to_csv(config.VAL_METADATA, index=False)
    test_subset.to_csv(config.TEST_METADATA, index=False)

    print(f"Created subsets with {subset_size} samples each.")

    # ---------------------------------------------------------
    # 3. Demonstrate Data Processing & Dataset Class
    # ---------------------------------------------------------
    print("\n[Step 3] Demonstrating Data Processing and Dataset instantiation...")

    # Process data manually to show usage (this generates the .npy cache files)
    # We set load_cached=False to force processing from the new subset CSVs
    train_data, train_labels, _ = dataset.process_data(
        config.TRAIN_METADATA,
        config.TRAIN_DATA_CACHE,
        config.TRAIN_LABELS_CACHE,
        load_cached=False,
    )

    # Verify shapes
    print(f"Processed Train Data Shape: {train_data.shape}")
    print(f"Processed Train Labels Shape: {train_labels.shape}")

    assert train_data.shape[0] == subset_size, "Train data count mismatch"
    assert train_data.ndim == 4, "Train data should be 4D (N, C, F, T)"
    assert train_data.shape[1] == 1, "Should have 1 channel"

    # Instantiate Dataset
    ds = dataset.WhaleDataset(
        train_data, train_labels, transform=dataset.get_transforms("train")
    )
    sample_spec, sample_label = ds[0]

    print(f"Single Sample Shape: {sample_spec.shape}")
    assert sample_spec.ndim == 3, "Sample should be 3D (C, F, T)"
    assert isinstance(sample_spec, torch.Tensor), "Output should be a Tensor"

    # ---------------------------------------------------------
    # 4. Demonstrate Model Architecture
    # ---------------------------------------------------------
    print("\n[Step 4] Demonstrating Model Initialization and Forward Pass...")

    # Initialize model
    net = model.WhaleEfficientNet(pretrained=config.PRETRAINED)
    net.eval()

    # Create a dummy batch (Batch=2, Channel=1, Freq=128, Time=64)
    dummy_input = torch.randn(2, 1, config.N_MELS, 64)

    # Forward pass
    with torch.no_grad():
        output = net(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, 1), "Output shape should be (Batch_Size, Num_Classes)"

    # ---------------------------------------------------------
    # 5. Demonstrate Training Pipeline
    # ---------------------------------------------------------
    print("\n[Step 5] Running Training Pipeline (Train -> Val -> Submission)...")

    # We use the train module's run_training function which orchestrates the full workflow.
    # We pass load_cached_data=True. It will use the train cache we just created,
    # and it will automatically process and cache the Val and Test subsets since their caches don't exist yet.

    train.run_training(epochs=config.EPOCHS, load_cached_data=True)

    # ---------------------------------------------------------
    # 6. Verify Submission
    # ---------------------------------------------------------
    print("\n[Step 6] Verifying Submission File...")

    if os.path.exists(config.SUBMISSION_PATH):
        sub_df = pd.read_csv(config.SUBMISSION_PATH)
        print(f"Submission file found with {len(sub_df)} rows.")

        # Verify submission content
        assert (
            len(sub_df) == subset_size
        ), f"Expected {subset_size} rows, found {len(sub_df)}"
        assert list(sub_df.columns) == [
            "clip",
            "probability",
        ], "Incorrect columns in submission"
        assert (
            sub_df["probability"].dtype == float
            or sub_df["probability"].dtype == np.float64
        ), "Probability should be float"

        print("Submission verification passed.")
    else:
        raise FileNotFoundError(
            f"Submission file not generated at {config.SUBMISSION_PATH}"
        )

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
