import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import patch

# Import provided library modules
import library.data_loader
from library.utils import seed_everything, get_device
from library.data_loader import get_processed_data, BraTSDataset
from library.model import VAMGNet, train_one_epoch, validate
from library.train import train, inference


def demo_utils():
    """
    Demonstrates utility functions for reproducibility and hardware selection.
    """
    print("=" * 40)
    print(" DEMO: UTILS")
    print("=" * 40)

    # 1. Seed Everything
    print("Setting random seeds...")
    seed_everything(42)

    # 2. Get Device
    device = get_device()
    print(f"Device detected: {device}")

    # Validation
    assert isinstance(
        device, torch.device
    ), "get_device should return a torch.device object"
    print("Utils demo passed.\n")


def demo_data_processing_and_loading():
    """
    Demonstrates the low-level data processing pipeline:
    Metadata -> DICOM Loading -> Preprocessing -> Dataset -> DataLoader
    """
    print("=" * 40)
    print(" DEMO: DATA PROCESSING & LOADING")
    print("=" * 40)

    # 1. Load Metadata
    meta_path = "./metadata/train.parquet"
    if not os.path.exists(meta_path):
        print(f"Metadata not found at {meta_path}. Skipping data demo.")
        return

    df = pd.read_parquet(meta_path)

    # 2. Create a small subset for demonstration (4 samples)
    subset_df = df.head(4).copy()
    print(f"Created subset with {len(subset_df)} samples.")

    # 3. Process Data
    # We use a custom split name 'demo_subset' to avoid conflict with main cache
    # This function loads DICOMs, resizes to 256x256, selects 16 slices, and normalizes
    print("Processing DICOM volumes (this may take a few seconds)...")
    X, y, ids = get_processed_data(
        subset_df, split_name="demo_subset", load_cached_data=False
    )

    # 4. Verify Shapes
    # Expected X shape: (N, Channels, H, W)
    # Channels = 4 modalities * 16 slices = 64
    print(f"Processed X shape: {X.shape}")
    print(f"Processed y shape: {y.shape}")

    assert X.shape == (
        4,
        64,
        256,
        256,
    ), f"Expected X shape (4, 64, 256, 256), got {X.shape}"
    assert y.shape == (4,), f"Expected y shape (4,), got {y.shape}"
    assert len(ids) == 4

    # 5. Create Dataset and DataLoader
    dataset = BraTSDataset(X, y, ids)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    # 6. Fetch a batch
    batch = next(iter(loader))
    images = batch["image"]
    targets = batch["target"]

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    assert images.shape == (2, 64, 256, 256)
    # Target is unsqueezed in BraTSDataset __getitem__ -> (1,)
    assert targets.shape == (2, 1)

    print("Data processing demo passed.\n")
    return loader


def demo_model_architecture(loader):
    """
    Demonstrates model instantiation, forward pass, and loss calculation.
    """
    print("=" * 40)
    print(" DEMO: MODEL ARCHITECTURE")
    print("=" * 40)

    device = get_device()

    # 1. Instantiate Model
    # VAMGNet uses EfficientNet-B0 backbone, modified for 64 input channels
    model = VAMGNet(num_classes=1).to(device)
    print("Model instantiated: VAMGNet (EfficientNet-B0 backbone)")

    # 2. Forward Pass
    batch = next(iter(loader))
    images = batch["image"].to(device)
    targets = batch["target"].to(device)

    outputs = model(images)
    print(f"Model Output Logits Shape: {outputs.shape}")

    assert outputs.shape == (2, 1), "Model output should be (Batch_Size, 1)"

    # 3. Loss Calculation
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(outputs, targets)
    print(f"Calculated Loss: {loss.item():.4f}")

    # 4. Verify Train Step Function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epoch_loss, epoch_auc = train_one_epoch(model, loader, criterion, optimizer, device)
    print(f"Single Epoch Train - Loss: {epoch_loss:.4f}, AUC: {epoch_auc:.4f}")

    print("Model architecture demo passed.\n")


def demo_full_pipeline_execution():
    """
    Demonstrates the high-level 'train' and 'inference' functions.
    Uses mocking to simulate a tiny dataset so the pipeline runs in seconds.
    """
    print("=" * 40)
    print(" DEMO: FULL PIPELINE (MOCKED)")
    print("=" * 40)

    # 1. Redirect Cache
    # We change the cache directory in the library to a temporary location.
    # This forces the data loader to re-process data (using our mocked subset)
    # instead of loading the full cached dataset from previous runs.
    demo_cache_dir = "./working/demo_cache/"
    if os.path.exists(demo_cache_dir):
        shutil.rmtree(demo_cache_dir)

    # Monkey-patch the global constant in the imported module
    library.data_loader.CACHE_DIR = demo_cache_dir
    print(f"Redirected cache to {demo_cache_dir}")

    # 2. Mock pandas.read_parquet
    # This ensures that when get_dataloaders() calls read_parquet(),
    # it gets a tiny DataFrame (4 rows) instead of the full dataset.
    real_read_parquet = pd.read_parquet

    def mock_read_parquet(path, *args, **kwargs):
        # Call the real function to get the structure/data
        df = real_read_parquet(path, *args, **kwargs)
        # Return only 4 rows to simulate a tiny dataset
        return df.head(4)

    print("Mocking dataset to 4 samples for rapid execution...")

    with patch("pandas.read_parquet", side_effect=mock_read_parquet):

        # 3. Run Training Loop
        print("\nStarting Training (1 Epoch)...")
        best_auc = train(
            epochs=1,
            batch_size=2,
            lr=1e-4,
            patience=1,
            save_path="./working/demo_best_model.pth",
            debug=False,
        )
        print(f"Training complete. Best AUC: {best_auc}")
        assert os.path.exists(
            "./working/demo_best_model.pth"
        ), "Model checkpoint not saved."

        # 4. Run Inference
        print("\nStarting Inference...")
        inference(
            model_path="./working/demo_best_model.pth",
            output_path="./working/demo_submission.csv",
            batch_size=2,
        )

        # 5. Verify Submission
        assert os.path.exists(
            "./working/demo_submission.csv"
        ), "Submission file not created."

        sub_df = pd.read_csv("./working/demo_submission.csv")
        print("\nGenerated Submission Head:")
        print(sub_df)

        assert (
            len(sub_df) == 4
        ), "Submission should have 4 rows (matching mocked test set)."
        assert "BraTS21ID" in sub_df.columns
        assert "MGMT_value" in sub_df.columns

        print("Full pipeline demo passed.")


if __name__ == "__main__":
    try:
        demo_utils()

        # We need the loader from the data demo to pass to the model demo
        loader = demo_data_processing_and_loading()

        if loader:
            demo_model_architecture(loader)

        demo_full_pipeline_execution()

        print("\n" + "=" * 40)
        print(" ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 40)

    except AssertionError as e:
        print(f"\nValidation Failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        exit(1)
