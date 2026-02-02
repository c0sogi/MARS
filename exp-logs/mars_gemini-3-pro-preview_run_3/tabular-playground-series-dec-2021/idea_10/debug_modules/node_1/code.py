import sys
import os
import torch
import numpy as np
import pandas as pd
import shutil

# Ensure the current directory is in the path to import from library
sys.path.append(os.getcwd())

from library.utils import seed_everything, get_device, ForestCoverDataset
from library.data_loader import get_dataloaders
from library.model import ParallelDCN_SE_ResNet, SEBlock, ResBlock, DCNv2Vector
from library.train import fit_model


def test_utils_and_dataset():
    print("=== Testing Utils and Dataset ===")

    # 1. Test Seeding
    seed_everything(42)
    rn1 = np.random.rand()
    seed_everything(42)
    rn2 = np.random.rand()
    assert rn1 == rn2, "seed_everything failed to produce reproducible numpy results"
    print("seed_everything: Verified")

    # 2. Test Device
    device = get_device()
    print(f"Device detected: {device}")
    assert isinstance(device, torch.device), "get_device did not return a torch.device"

    # 3. Test Dataset Class
    # Create dummy data
    X_dummy = np.random.randn(100, 10).astype(np.float32)
    y_dummy = np.random.randint(0, 7, size=(100,)).astype(np.int64)

    ds = ForestCoverDataset(X_dummy, y_dummy)
    assert len(ds) == 100, "Dataset length mismatch"

    sample_x, sample_y = ds[0]
    assert torch.is_tensor(sample_x), "Dataset __getitem__ X is not a tensor"
    assert torch.is_tensor(sample_y), "Dataset __getitem__ y is not a tensor"
    assert sample_x.shape == (10,), "Dataset sample X shape mismatch"

    # Test Test-set mode (no targets)
    ds_test = ForestCoverDataset(X_dummy)
    sample_x_test = ds_test[0]
    assert torch.is_tensor(sample_x_test), "Test Dataset X is not a tensor"
    assert not isinstance(sample_x_test, tuple), "Test Dataset should return only X"

    print("ForestCoverDataset: Verified")


def test_data_loader():
    print("\n=== Testing Data Loader ===")

    # Use quick_run=True to load a subset of data for speed
    # This triggers process_data internally
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=128,
        quick_run=True,
        cache_dir="./working/test_demo_cache/",  # Use a separate cache for this demo
    )

    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Val loader is empty"
    assert len(test_loader) > 0, "Test loader is empty"
    assert len(test_ids) > 0, "Test IDs are empty"

    # Check batch structure
    inputs, targets = next(iter(train_loader))
    assert inputs.dim() == 2, "Input batch should be 2D (Batch, Features)"
    assert targets.dim() == 1, "Target batch should be 1D (Batch,)"
    assert (
        inputs.shape[0] == targets.shape[0]
    ), "Batch size mismatch between inputs and targets"

    feature_dim = inputs.shape[1]
    print(f"Data Loaded successfully. Feature Dimension: {feature_dim}")
    print("get_dataloaders: Verified")

    return feature_dim


def test_model_architecture(input_dim):
    print("\n=== Testing Model Architecture ===")

    batch_size = 32
    num_classes = 7

    # 1. Test Components
    # SEBlock
    se = SEBlock(channels=64)
    dummy_feat = torch.randn(batch_size, 64)
    out_se = se(dummy_feat)
    assert out_se.shape == (batch_size, 64), "SEBlock output shape mismatch"

    # DCNv2Vector
    dcn = DCNv2Vector(input_dim=input_dim, num_layers=2)
    dummy_input = torch.randn(batch_size, input_dim)
    out_dcn = dcn(dummy_input)
    assert out_dcn.shape == (batch_size, input_dim), "DCNv2Vector output shape mismatch"

    # 2. Test Full Model
    model = ParallelDCN_SE_ResNet(input_dim=input_dim, num_classes=num_classes)
    model.eval()

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        batch_size,
        num_classes,
    ), f"Model output shape mismatch. Expected {(batch_size, num_classes)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    print("ParallelDCN_SE_ResNet: Verified")


def test_training_pipeline():
    print("\n=== Testing Training Pipeline ===")

    # Run the high-level fit_model function
    # Limiting to 1 epoch and quick_run for demonstration speed
    try:
        fit_model(epochs=1, batch_size=256, quick_run=True)
    except Exception as e:
        raise RuntimeError(f"Training pipeline failed: {e}")

    # Verify submission file generation
    submission_path = "./submission/submission.csv"
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    assert "Id" in df_sub.columns, "Submission missing Id column"
    assert "Cover_Type" in df_sub.columns, "Submission missing Cover_Type column"
    assert len(df_sub) > 0, "Submission file is empty"

    # Verify values are valid classes (1-7)
    # Note: quick_run subsets data, so we check the subset size logic in fit_model matches
    # In quick_run, test set is 2000 rows.
    assert (
        len(df_sub) == 2000
    ), f"Expected 2000 rows in quick_run submission, got {len(df_sub)}"
    assert df_sub["Cover_Type"].min() >= 1, "Invalid class label < 1 found"
    assert df_sub["Cover_Type"].max() <= 7, "Invalid class label > 7 found"

    print("fit_model: Verified")


if __name__ == "__main__":
    # Clean up any previous demo cache to ensure fresh run
    if os.path.exists("./working/test_demo_cache/"):
        shutil.rmtree("./working/test_demo_cache/")

    try:
        # 1. Utils & Dataset
        test_utils_and_dataset()

        # 2. Data Loader (returns input_dim for model test)
        feat_dim = test_data_loader()

        # 3. Model
        test_model_architecture(feat_dim)

        # 4. Full Training Loop
        test_training_pipeline()

        print("\nAll demonstrations and verifications passed successfully.")

    except AssertionError as e:
        print(f"\nVerification Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)
