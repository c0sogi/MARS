import os
import shutil
import pandas as pd
import numpy as np
import torch
import warnings
import sys

# Import from the provided library files
from library.utils import seed_everything, get_data
from library.data_loader import get_dataloaders
from library.model import WideAsymmetricDCNResNet
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_mini_metadata(source_dir, dest_dir, sample_size=2000):
    """
    Creates a smaller version of the metadata parquet files for rapid demonstration.
    """
    print(f"Creating mini metadata in {dest_dir}...")
    os.makedirs(dest_dir, exist_ok=True)

    files = ["train.parquet", "val.parquet", "test.parquet"]

    for f in files:
        src_path = os.path.join(source_dir, f)
        dest_path = os.path.join(dest_dir, f)

        if os.path.exists(src_path):
            df = pd.read_parquet(src_path)
            # Sample to reduce size, ensuring we don't exceed length
            n = min(len(df), sample_size)

            # For training data, we want to try and keep class diversity if possible,
            # but simple sampling is sufficient for a functional code demo.
            df_mini = df.sample(n=n, random_state=42).reset_index(drop=True)

            df_mini.to_parquet(dest_path, index=False)
            print(f"  Saved mini {f}: {df_mini.shape}")
        else:
            print(f"  Warning: {f} not found in source.")


def test_data_pipeline(metadata_dir, cache_dir):
    """
    Verifies get_data and get_dataloaders functionality.
    """
    print("\n=== Testing Data Pipeline ===")

    # 1. Test get_data
    print("Testing library.utils.get_data...")
    data = get_data(
        load_cached_data=False,  # Force processing
        cache_dir=cache_dir,
        metadata_dir=metadata_dir,
    )

    expected_keys = ["train_X", "train_y", "val_X", "val_y", "test_X", "test_ids"]
    for k in expected_keys:
        assert k in data, f"Missing key {k} in data dictionary"
        assert isinstance(data[k], np.ndarray), f"{k} is not a numpy array"

    print("  get_data structure verified.")
    print(f"  Train X shape: {data['train_X'].shape}")
    print(f"  Train y shape: {data['train_y'].shape}")

    # 2. Test get_dataloaders
    print("Testing library.data_loader.get_dataloaders...")
    batch_size = 32
    train_loader, val_loader, test_loader, input_dim, num_classes = get_dataloaders(
        batch_size=batch_size,
        num_workers=0,  # Avoid multiprocessing overhead in simple demo
        load_cached_data=True,
        cache_dir=cache_dir,
        metadata_dir=metadata_dir,
    )

    assert input_dim == data["train_X"].shape[1], "Input dim mismatch"
    # num_classes depends on the max label in the mini-set
    print(f"  Input Dim: {input_dim}, Num Classes: {num_classes}")

    # Fetch one batch
    X_batch, y_batch = next(iter(train_loader))
    assert X_batch.shape == (
        batch_size,
        input_dim,
    ), f"Batch X shape mismatch: {X_batch.shape}"
    assert y_batch.shape == (batch_size,), f"Batch y shape mismatch: {y_batch.shape}"

    print("  DataLoaders verified.")
    return input_dim, num_classes


def test_model_architecture(input_dim, num_classes):
    """
    Verifies the model initialization and forward pass.
    """
    print("\n=== Testing Model Architecture ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WideAsymmetricDCNResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=128,  # Reduced for demo
        dropout=0.1,
    ).to(device)

    batch_size = 16
    dummy_input = torch.randn(batch_size, input_dim).to(device)

    print("Running forward pass...")
    output = model(dummy_input)

    assert output.shape == (
        batch_size,
        num_classes,
    ), f"Output shape mismatch. Expected {(batch_size, num_classes)}, got {output.shape}"

    print("  Model forward pass successful.")


def test_trainer_execution(metadata_dir, cache_dir, output_dir):
    """
    Verifies the Trainer class by running a full training cycle.
    """
    print("\n=== Testing Trainer Execution ===")

    trainer = Trainer(
        epochs=1,  # Single epoch for speed
        batch_size=64,
        learning_rate=1e-3,
        warmup_epochs=0,
        patience=1,
        hidden_dim=128,  # Small hidden dim for speed
        cache_dir=cache_dir,
        metadata_dir=metadata_dir,
        output_dir=output_dir,
        seed=42,
    )

    print("Starting Trainer.train()...")
    trainer.train()

    # Verify submission
    submission_path = os.path.join(output_dir, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    assert (
        "Id" in df_sub.columns and "Cover_Type" in df_sub.columns
    ), "Submission columns mismatch"

    assert len(df_sub) > 0, "Submission file is empty"

    # Check if predictions are valid integers (1-7 range expected for Cover_Type)
    # Note: In mini-set, we might not see all classes, but predictions should be valid.
    preds = df_sub["Cover_Type"]
    assert pd.api.types.is_integer_dtype(preds), "Predictions are not integers"
    print(f"  Submission generated at {submission_path}")
    print(f"  First 5 rows:\n{df_sub.head()}")


def main():
    # Setup paths
    BASE_DIR = "./working"
    METADATA_SOURCE = "./metadata"

    MINI_METADATA_DIR = os.path.join(BASE_DIR, "demo_metadata")
    CACHE_DIR = os.path.join(BASE_DIR, "demo_cache")
    OUTPUT_DIR = os.path.join(BASE_DIR, "demo_submission")

    # Ensure reproducibility
    seed_everything(42)

    try:
        # 1. Create Mini Metadata
        create_mini_metadata(METADATA_SOURCE, MINI_METADATA_DIR)

        # 2. Test Data Pipeline
        input_dim, num_classes = test_data_pipeline(MINI_METADATA_DIR, CACHE_DIR)

        # 3. Test Model
        test_model_architecture(input_dim, num_classes)

        # 4. Test Trainer
        test_trainer_execution(MINI_METADATA_DIR, CACHE_DIR, OUTPUT_DIR)

        print("\nAll demonstrations completed successfully.")

    finally:
        # Cleanup
        print("\nCleaning up temporary files...")
        for d in [MINI_METADATA_DIR, CACHE_DIR, OUTPUT_DIR]:
            if os.path.exists(d):
                shutil.rmtree(d)
        print("Cleanup done.")


if __name__ == "__main__":
    main()
