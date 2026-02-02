import os
import shutil
import pandas as pd
import numpy as np
import torch
import sys

# Import library modules
import library.config as config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders, BraTSDataset
from library.model import AsymmetricEfficientNet
from library.train import run_training
from library.inference import run_inference


def setup_demo_environment():
    """
    Sets up a temporary environment for the demo by creating a subset of the data
    and overriding the global configuration paths to point to this subset.
    This ensures the demo runs quickly.
    """
    print("--- Setting up Demo Environment ---")

    # Define demo paths
    demo_base = "./working/demo_run"
    demo_metadata_dir = os.path.join(demo_base, "metadata")
    demo_working_dir = os.path.join(demo_base, "working")
    demo_submission_dir = os.path.join(demo_base, "submission")

    # Clean up previous runs if they exist
    if os.path.exists(demo_base):
        shutil.rmtree(demo_base)

    os.makedirs(demo_metadata_dir, exist_ok=True)
    os.makedirs(demo_working_dir, exist_ok=True)
    os.makedirs(demo_submission_dir, exist_ok=True)

    # 1. Create Subset Metadata
    # We read the original metadata and sample a small number of rows
    # to simulate a quick training run.

    # Load original
    orig_train = pd.read_csv(os.path.join(config.METADATA_DIR, "train.csv"))
    orig_val = pd.read_csv(os.path.join(config.METADATA_DIR, "val.csv"))
    orig_test = pd.read_csv(os.path.join(config.METADATA_DIR, "test.csv"))

    # Sample subsets (enough for 1 batch of 32)
    # We use n=32 for train to ensure drop_last=True works if batch_size=32
    subset_train = orig_train.head(32).copy()
    subset_val = orig_val.head(16).copy()
    subset_test = orig_test.head(5).copy()

    print(
        f"Created subset metadata: Train={len(subset_train)}, Val={len(subset_val)}, Test={len(subset_test)}"
    )

    # Save subsets
    subset_train.to_csv(os.path.join(demo_metadata_dir, "train.csv"), index=False)
    subset_val.to_csv(os.path.join(demo_metadata_dir, "val.csv"), index=False)
    subset_test.to_csv(os.path.join(demo_metadata_dir, "test.csv"), index=False)

    # 2. Monkey-Patch Configuration
    # We modify the config module variables at runtime to point to our demo directories.
    print("Overriding config paths...")
    config.METADATA_DIR = demo_metadata_dir
    config.WORKING_DIR = demo_working_dir
    config.SUBMISSION_DIR = demo_submission_dir
    config.SUBMISSION_PATH = os.path.join(demo_submission_dir, "demo_submission.csv")

    # Adjust hyperparameters for speed
    config.BATCH_SIZE = 8  # Smaller batch size for demo to reduce memory usage
    config.NUM_WORKERS = 2

    return subset_train, subset_val, subset_test


def verify_data_loading(train_df, val_df):
    """
    Demonstrates and verifies the Data Loading pipeline.
    """
    print("\n--- Verifying Data Loading ---")

    # Initialize DataLoaders
    # Note: This will compute ROI anchors for the subset, which might take a few seconds.
    print("Initializing DataLoaders (computing ROI anchors)...")
    train_loader, val_loader, _ = get_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=None,
        load_cached_data=False,  # Force re-compute for the demo subset
    )

    # Fetch one batch
    print("Fetching one training batch...")
    images, labels = next(iter(train_loader))

    # Assertions
    # Expected Shape: (Batch_Size, Channels, Height, Width)
    # Channels = 4 modalities * 5 slices = 20
    expected_channels = 20
    expected_size = config.IMG_SIZE

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    assert images.ndim == 4, "Images should be 4D tensors (B, C, H, W)"
    assert (
        images.shape[1] == expected_channels
    ), f"Expected {expected_channels} channels, got {images.shape[1]}"
    assert (
        images.shape[2] == expected_size and images.shape[3] == expected_size
    ), f"Expected resolution {expected_size}x{expected_size}, got {images.shape[2]}x{images.shape[3]}"
    assert labels.ndim == 1, "Labels should be 1D tensors"

    print("Data Loading verified successfully.")
    return images, labels


def verify_model_architecture(sample_batch):
    """
    Demonstrates and verifies the Model architecture.
    """
    print("\n--- Verifying Model Architecture ---")

    device = get_device()
    model = AsymmetricEfficientNet(model_name="efficientnet_b0", pretrained=False)
    model.to(device)
    model.eval()

    inputs = sample_batch.to(device)

    print("Running forward pass...")
    with torch.no_grad():
        logits = model(inputs)

    print(f"Output Logits Shape: {logits.shape}")

    # Assertions
    assert logits.ndim == 2, "Logits should be 2D (B, 1)"
    assert logits.shape[0] == inputs.shape[0], "Batch dimension mismatch"
    assert logits.shape[1] == 1, "Output dimension should be 1 (binary classification)"

    print("Model Architecture verified successfully.")


def demonstrate_training():
    """
    Demonstrates the training loop using the library function.
    """
    print("\n--- Demonstrating Training Loop ---")

    # Run training for 1 epoch
    # The run_training function uses the paths we overrode in config
    best_auc = run_training(load_cached_data=True, max_epochs=1)

    # Verify artifact creation
    model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(model_path), "Model checkpoint was not saved."

    print(f"Training demonstration complete. Best AUC: {best_auc:.4f}")


def demonstrate_inference():
    """
    Demonstrates the inference pipeline using the library function.
    """
    print("\n--- Demonstrating Inference Pipeline ---")

    # Run inference
    # This reads from config.METADATA_DIR/test.csv and saves to config.SUBMISSION_PATH
    run_inference(load_cached_data=True)

    # Verify submission file
    sub_path = config.SUBMISSION_PATH
    assert os.path.exists(sub_path), "Submission file was not created."

    df_sub = pd.read_csv(sub_path)
    print(f"Submission file loaded. Shape: {df_sub.shape}")

    # Check columns
    assert "BraTS21ID" in df_sub.columns, "BraTS21ID column missing"
    assert "MGMT_value" in df_sub.columns, "MGMT_value column missing"

    # Check values
    preds = df_sub["MGMT_value"]
    assert preds.min() >= 0.0 and preds.max() <= 1.0, "Predictions out of range [0, 1]"

    print("Inference demonstration complete.")
    print("Sample predictions:")
    print(df_sub.head())


if __name__ == "__main__":
    # 1. Reproducibility
    seed_everything(42)

    # 2. Setup environment (Subset data & Config override)
    try:
        train_df, val_df, test_df = setup_demo_environment()

        # 3. Verify Data Loading
        sample_images, _ = verify_data_loading(train_df, val_df)

        # 4. Verify Model
        verify_model_architecture(sample_images)

        # 5. Run Training Demo
        demonstrate_training()

        # 6. Run Inference Demo
        demonstrate_inference()

        print("\n=== All Demonstrations Passed Successfully ===")

    except AssertionError as e:
        print(f"\n!!! Validation Failed: {e} !!!")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! An unexpected error occurred: {e} !!!")
        import traceback

        traceback.print_exc()
        sys.exit(1)
