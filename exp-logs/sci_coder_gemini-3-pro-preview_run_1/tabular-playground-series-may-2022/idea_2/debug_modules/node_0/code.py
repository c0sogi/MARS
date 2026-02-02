import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Import from the provided library files
from library.utils import seed_everything, get_device
from library.preprocessing import process_data
from library.dataset import ManufacturingDataset, get_datasets
from library.model import ManufacturingNet
from library.trainer import run_experiment


def main():
    print("Starting Library Usage Demonstration...")

    # 1. Setup and Configuration
    # ---------------------------------------------------------
    seed_everything(42)
    device = get_device()
    print(f"Device selected: {device}")

    # Define working paths
    WORKING_DIR = "./working/demo"
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 2. Verify Preprocessing Logic
    # ---------------------------------------------------------
    print("\n--- Verifying Preprocessing (process_data) ---")

    # Use a small sample size to speed up verification and force re-processing (bypassing cache)
    sample_size = 500

    # Note: process_data returns a tuple of 9 elements
    data_tuple = process_data(load_cached_data=False, sample_size=sample_size)

    (
        X_train_num,
        X_train_seq,
        y_train,
        X_val_num,
        X_val_seq,
        y_val,
        X_test_num,
        X_test_seq,
        ids_test,
    ) = data_tuple

    # Assertions to verify data shapes and types
    # Numerical features: 30 original + 1 engineered = 31
    assert X_train_num.shape == (
        sample_size,
        31,
    ), f"Expected X_train_num shape ({sample_size}, 31), got {X_train_num.shape}"
    assert X_train_seq.shape[0] == sample_size, "X_train_seq row count mismatch"
    assert y_train.shape == (sample_size,), "y_train shape mismatch"
    assert ids_test.shape == (sample_size,), "ids_test shape mismatch"

    # Verify data types
    assert X_train_num.dtype == np.float32, "Numerical data should be float32"
    assert X_train_seq.dtype == np.int32, "Sequence data should be int32"

    print("Preprocessing verification passed: Shapes and types are correct.")

    # 3. Verify Dataset Class
    # ---------------------------------------------------------
    print("\n--- Verifying Dataset (ManufacturingDataset) ---")

    # Initialize dataset with the processed data
    train_ds = ManufacturingDataset(X_train_num, X_train_seq, y_train)

    # Check length
    assert len(train_ds) == sample_size, "Dataset length mismatch"

    # Check item retrieval
    item = train_ds[0]
    required_keys = {"numerical", "sequence", "label"}
    assert required_keys.issubset(
        item.keys()
    ), f"Dataset item missing keys. Found: {item.keys()}"

    # Check tensor properties
    assert torch.is_tensor(item["numerical"]), "Numerical output is not a tensor"
    assert (
        item["numerical"].dtype == torch.float32
    ), "Numerical tensor should be float32"
    assert (
        item["sequence"].dtype == torch.long
    ), "Sequence tensor should be long (int64)"
    assert item["label"].dtype == torch.float32, "Label tensor should be float32"

    print("Dataset verification passed: Item structure and tensor types are correct.")

    # 4. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n--- Verifying Model (ManufacturingNet) ---")

    # Determine model parameters dynamically from data
    num_numerical = X_train_num.shape[1]
    seq_len = X_train_seq.shape[1]

    # Calculate vocab size (max index + 1 for padding/unknown)
    # We look at all splits to ensure we cover the full range
    max_idx = max(X_train_seq.max(), X_val_seq.max(), X_test_seq.max())
    vocab_size = int(max_idx) + 1

    embedding_dim = 8

    # Instantiate model
    model = ManufacturingNet(
        num_numerical_features=num_numerical,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        seq_len=seq_len,
        hidden_units=[32, 16],  # Small units for demo
        dropout_rate=0.1,
    ).to(device)

    # Create dummy batch
    batch_size = 4
    dummy_num = torch.randn(batch_size, num_numerical).to(device)
    dummy_seq = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_num, dummy_seq)

    # Assertions on output
    assert output.shape == (
        batch_size,
        1,
    ), f"Model output shape mismatch. Expected {(batch_size, 1)}, got {output.shape}"
    assert (output >= 0).all() and (
        output <= 1
    ).all(), "Model output not in probability range [0, 1]"

    print(
        "Model verification passed: Forward pass successful and output shape correct."
    )

    # 5. Verify Full Training Pipeline (Trainer)
    # ---------------------------------------------------------
    print("\n--- Verifying Full Pipeline (run_experiment) ---")

    # Run the experiment with minimal parameters for speed
    # This tests: get_datasets -> DataLoader -> Model Init -> Trainer -> Fit -> Predict
    run_experiment(
        epochs=2,
        batch_size=32,
        learning_rate=1e-3,
        embedding_dim=8,
        hidden_units=[32, 16],
        dropout_rate=0.1,
        patience=1,  # Strict early stopping
        sample_size=1000,  # Small subset for speed
        load_cached_data=False,  # Force reload to verify processing integration
        output_path=SUBMISSION_PATH,
    )

    # Verify submission file generation
    if not os.path.exists(SUBMISSION_PATH):
        raise FileNotFoundError(f"Submission file was not created at {SUBMISSION_PATH}")

    # Verify submission content format
    df_sub = pd.read_csv(SUBMISSION_PATH)
    assert list(df_sub.columns) == ["id", "target"], "Submission columns mismatch"
    assert len(df_sub) == 1000, f"Expected 1000 predictions, got {len(df_sub)}"
    assert df_sub["target"].between(0, 1).all(), "Predictions out of bounds"

    print(
        f"Pipeline verification passed: Submission generated at {SUBMISSION_PATH} with correct format."
    )
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
