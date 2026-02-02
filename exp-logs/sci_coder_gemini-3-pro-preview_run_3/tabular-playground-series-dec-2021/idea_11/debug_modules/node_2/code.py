import sys
import os
import torch
import numpy as np
import pandas as pd
import warnings

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.utils import seed_everything, get_device, get_data, save_submission
from library.data_processing import load_and_preprocess_data
from library.model import ParallelDCNResNeXt, CrossNetV2, ResNeXtBlock
from library.train import run_training


def demo_utils():
    """
    Demonstrates usage of utility functions: seeding, device selection,
    data loading (with caching/subsampling), and submission saving.
    """
    print("--- Demo: library.utils ---")

    # 1. Reproducibility
    seed_everything(42)

    # 2. Device Check
    device = get_device()
    print(f"Device selected: {device}")

    # 3. Data Loading with Subsampling
    # We use a specific directory for this demo to avoid conflicts
    base_dir = "./working/demo_utils"
    sample_n = 500
    print(f"Testing get_data with sample_size={sample_n}...")

    train_X, train_y, val_X, val_y, test_X, test_ids = get_data(
        load_cached_data=False, base_dir=base_dir, sample_size=sample_n
    )

    # Assertions to verify logic
    assert (
        len(train_X) == sample_n
    ), f"Expected {sample_n} train samples, got {len(train_X)}"
    assert len(train_y) == sample_n
    assert len(val_X) == sample_n
    assert len(val_y) == sample_n
    # Note: test_X is not subsampled by design in the library function

    print(f"Data shapes verified: Train {train_X.shape}, Val {val_X.shape}")

    # 4. Submission Saving
    dummy_ids = np.array([100, 101, 102])
    dummy_preds = np.array([1, 2, 1])
    sub_path = "./working/demo_utils/dummy_sub.csv"

    save_submission(dummy_ids, dummy_preds, output_path=sub_path)

    assert os.path.exists(sub_path), "Submission file not created"
    df = pd.read_csv(sub_path)
    assert df.shape == (3, 2), "Submission file has incorrect shape"
    assert list(df.columns) == ["Id", "Cover_Type"], "Submission columns mismatch"
    print("save_submission verified.")


def demo_data_processing():
    """
    Demonstrates creating DataLoaders and verifies batch shapes.
    Returns the input dimension for model configuration.
    """
    print("\n--- Demo: library.data_processing ---")
    base_dir = "./working/demo_data"
    batch_size = 32
    sample_n = 200

    print("Testing load_and_preprocess_data...")
    train_loader, val_loader, test_loader, test_ids = load_and_preprocess_data(
        batch_size=batch_size,
        load_cached_data=False,
        base_dir=base_dir,
        sample_size=sample_n,
    )

    # Verify Train Loader
    batch_X, batch_y = next(iter(train_loader))
    # Batch size might be smaller if it's the last batch, but here we check max
    assert batch_X.shape[0] <= batch_size
    assert batch_y.shape[0] <= batch_size

    # Determine input dimension (Original features + Engineered features)
    input_dim = batch_X.shape[1]
    print(f"DataLoader batch shape: {batch_X.shape}, Input Dim: {input_dim}")

    # Verify Test Loader
    batch_test_X = next(iter(test_loader))
    assert batch_test_X.shape[1] == input_dim

    print("DataLoaders verified.")
    return input_dim


def demo_model(input_dim):
    """
    Demonstrates instantiation and forward pass of the model components.
    """
    print("\n--- Demo: library.model ---")
    batch_size = 16
    num_classes = 7

    # 1. Test CrossNetV2 (DCN Branch)
    print("Testing CrossNetV2...")
    dcn = CrossNetV2(input_dim, num_layers=2)
    dummy_input = torch.randn(batch_size, input_dim)
    out_dcn = dcn(dummy_input)
    assert out_dcn.shape == (batch_size, input_dim), "CrossNetV2 output shape mismatch"

    # 2. Test ResNeXtBlock (Backbone Component)
    print("Testing ResNeXtBlock...")
    d_model = 64  # Must be divisible by cardinality (default 32)
    res_block = ResNeXtBlock(dim=d_model, cardinality=32)

    # ResNeXtBlock expects (Batch, Dim, 1) due to Conv1d usage
    dummy_res_input = torch.randn(batch_size, d_model, 1)
    out_res = res_block(dummy_res_input)
    assert out_res.shape == (
        batch_size,
        d_model,
        1,
    ), "ResNeXtBlock output shape mismatch"

    # 3. Test ParallelDCNResNeXt (Full Model)
    print("Testing ParallelDCNResNeXt...")
    model = ParallelDCNResNeXt(
        input_dim=input_dim,
        num_classes=num_classes,
        dcn_layers=2,
        resnext_layers=2,
        d_model=64,  # Reduced size for demo
        cardinality=32,
    )
    logits = model(dummy_input)

    # Output should be (Batch, NumClasses)
    assert logits.shape == (batch_size, num_classes), "Model output shape mismatch"
    print("Model architecture verified.")


def demo_training():
    """
    Demonstrates the full training pipeline using run_training.
    Uses minimal epochs and data to ensure speed.
    """
    print("\n--- Demo: library.train ---")

    print("Running training pipeline (fast mode)...")

    # run_training encapsulates data loading, model init, training loop, and prediction
    try:
        acc = run_training(
            batch_size=256,
            epochs=1,  # Only 1 epoch for demonstration
            learning_rate=1e-3,
            patience=1,
            base_dir="./working/demo_train",
            sample_size=1000,  # Small sample size for speed
        )
        print(f"Training completed. Validation Accuracy: {acc}")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e

    # Verify that the submission file was generated
    # The library function hardcodes the output to ./submission/submission.csv
    sub_path = "./submission/submission.csv"
    assert os.path.exists(sub_path), "Final submission file missing"

    # Verify content
    df = pd.read_csv(sub_path)
    assert not df.empty, "Submission file is empty"
    print(f"Submission found at {sub_path} with {len(df)} rows.")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    print("Starting Demonstration Script...")

    # 1. Utils
    demo_utils()

    # 2. Data Processing
    # We retrieve the input dimension dynamically to configure the model correctly
    dim = demo_data_processing()

    # 3. Model
    demo_model(dim)

    # 4. Training
    demo_training()

    print("\nAll demonstrations passed successfully.")
