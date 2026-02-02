import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import provided library modules
import library.config as config
import library.data_utils as data_utils
import library.dataset as dataset
import library.model as model
import library.train_eval as train_eval


def run_demo():
    print("=== Starting Demonstration ===")

    # ---------------------------------------------------------
    # 0. Setup & Configuration Overrides for Speed
    # ---------------------------------------------------------
    print("\n[Step 0] Configuring environment for rapid demonstration...")

    # Set seeds for reproducibility
    train_eval.set_seed(42)

    # Override config for speed
    config.EPOCHS = 2
    config.BATCH_SIZE = 32  # Smaller batch size for the small subset
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Define a demo working directory to avoid conflicts
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    config.WORKING_DIR = DEMO_DIR
    config.CACHE_DIR = DEMO_DIR
    config.SUBMISSION_DIR = DEMO_DIR
    config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Subset size for demonstration
    N_ROWS = 2000

    # ---------------------------------------------------------
    # 1. Data Loading & Feature Engineering Demo
    # ---------------------------------------------------------
    print(f"\n[Step 1] Loading and processing data (subset={N_ROWS})...")

    # Force processing from scratch (load_cached_data=False) to demonstrate engineering logic
    # Note: The library reads the full CSV first, then subsamples.
    train_df, val_df, test_df, vocab_sizes = data_utils.load_data(
        load_cached_data=False, nrows=N_ROWS
    )

    # Validation
    print("Validating data processing...")
    assert (
        len(train_df) == N_ROWS
    ), f"Train DataFrame should have {N_ROWS} rows, got {len(train_df)}"
    assert (
        len(val_df) == N_ROWS
    ), f"Val DataFrame should have {N_ROWS} rows, got {len(val_df)}"
    assert (
        len(test_df) == N_ROWS
    ), f"Test DataFrame should have {N_ROWS} rows, got {len(test_df)}"

    # Check if feature engineering happened (f_27 decomposed)
    # f_27 is length 10, so we expect f_27_0 to f_27_9
    expected_cols = [f"f_27_{i}" for i in range(10)]
    for col in expected_cols:
        assert col in train_df.columns, f"Decomposed column {col} missing from train_df"

    # Check if original f_27 is dropped
    assert "f_27" not in train_df.columns, "Original 'f_27' column should be dropped"

    # Check vocab_sizes
    # We have f_29, f_30, and 10 chars from f_27 = 12 categorical features
    assert (
        len(vocab_sizes) == 12
    ), f"Expected 12 categorical features, got {len(vocab_sizes)}"

    print("Data processing validation passed.")

    # ---------------------------------------------------------
    # 2. Dataset Class Demo
    # ---------------------------------------------------------
    print("\n[Step 2] Instantiating PyTorch Datasets...")

    train_ds = dataset.ManufacturingDataset(train_df, is_test=False)
    test_ds = dataset.ManufacturingDataset(test_df, is_test=True)

    # Validation
    print("Validating dataset...")
    sample = train_ds[0]

    # Check keys
    assert "continuous" in sample
    assert "categorical" in sample
    assert "target" in sample

    # Check types and shapes
    assert isinstance(sample["continuous"], torch.Tensor)
    assert isinstance(sample["categorical"], torch.Tensor)
    assert isinstance(sample["target"], torch.Tensor)

    # Continuous features: f_00..f_28 (excl f_27) + unique_count = 29 features
    assert (
        sample["continuous"].shape[0] == 29
    ), f"Expected 29 continuous features, got {sample['continuous'].shape[0]}"

    # Categorical features: 12
    assert (
        sample["categorical"].shape[0] == 12
    ), f"Expected 12 categorical features, got {sample['categorical'].shape[0]}"

    # Target shape
    assert sample["target"].shape == (
        1,
    ), f"Expected target shape (1,), got {sample['target'].shape}"

    print("Dataset validation passed.")

    # ---------------------------------------------------------
    # 3. Model Architecture Demo
    # ---------------------------------------------------------
    print("\n[Step 3] Initializing MRPFEModel...")

    num_continuous = len(config.ALL_CONTINUOUS_FEATURES)
    model_instance = model.MRPFEModel(vocab_sizes, num_continuous)
    model_instance.to(config.DEVICE)

    # Create a dummy batch
    batch_size = 4
    dummy_cont = torch.randn(batch_size, num_continuous).to(config.DEVICE)
    # Create dummy categorical indices within vocab range
    dummy_cat = torch.stack(
        [torch.randint(0, v, (batch_size,)) for v in vocab_sizes], dim=1
    ).to(config.DEVICE)

    # Forward pass
    print("Performing forward pass...")
    outputs = model_instance(dummy_cont, dummy_cat)

    # Validation
    # The model returns a list of outputs from 5 streams
    assert isinstance(outputs, list), "Model output should be a list"
    assert len(outputs) == 5, f"Expected 5 stream outputs, got {len(outputs)}"

    for i, out in enumerate(outputs):
        assert out.shape == (
            batch_size,
            1,
        ), f"Stream {i} output shape mismatch. Expected ({batch_size}, 1), got {out.shape}"

    print("Model architecture validation passed.")

    # ---------------------------------------------------------
    # 4. Training Loop Demo
    # ---------------------------------------------------------
    print("\n[Step 4] Running Training Loop (2 Epochs)...")

    # We use the provided run_training function which encapsulates the loop
    # It will use the cached data we generated in Step 1 if we don't clear it,
    # but run_training calls load_data internally.
    best_model_path, _, _ = train_eval.run_training(nrows=N_ROWS)

    # Validation
    assert os.path.exists(best_model_path), "Best model file was not saved."
    print(f"Training completed. Model saved to {best_model_path}")

    # ---------------------------------------------------------
    # 5. Submission Generation Demo
    # ---------------------------------------------------------
    print("\n[Step 5] Generating Submission...")

    train_eval.generate_submission(
        model_path=best_model_path,
        vocab_sizes=vocab_sizes,
        num_continuous=num_continuous,
        nrows=N_ROWS,  # Limit rows for speed in demo
    )

    # Validation
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not found."

    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    assert sub_df.shape == (
        N_ROWS,
        2,
    ), f"Submission shape mismatch. Expected ({N_ROWS}, 2), got {sub_df.shape}"
    assert list(sub_df.columns) == ["id", "target"], "Submission columns mismatch."

    # Check probabilities are valid
    assert (
        sub_df["target"].min() >= 0.0 and sub_df["target"].max() <= 1.0
    ), "Predictions out of probability range [0, 1]"

    print(f"Submission generated successfully at {config.SUBMISSION_PATH}")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
