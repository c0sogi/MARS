import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import set_seed, PIFEModel, NUM_STREAMS
from library.data_utils import process_data, get_dataloaders
from library.train_eval import train_and_evaluate


def main():
    print("=== Starting Library Demonstration ===")

    # 1. Setup and Configuration
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Demonstrate Data Processing
    print("\n[Step 1] Testing Data Processing...")
    # Load cached data if available, otherwise process from scratch.
    # This returns a dictionary containing numpy arrays for train/val/test.
    data = process_data(load_cached_data=True)

    # Validation: Check keys and basic properties
    expected_keys = [
        "X_train_cat",
        "X_train_cont",
        "y_train",
        "X_val_cat",
        "X_val_cont",
        "y_val",
        "X_test_cat",
        "X_test_cont",
        "test_ids",
        "vocab_sizes",
    ]
    for key in expected_keys:
        assert key in data, f"Missing key '{key}' in processed data."

    print(f"Data Loaded Successfully.")
    print(f"Train Set Size: {len(data['X_train_cat'])}")
    print(f"Vocab Sizes: {data['vocab_sizes']}")

    # 3. Demonstrate DataLoaders
    print("\n[Step 2] Testing DataLoaders...")
    batch_size = 128
    train_loader, val_loader, test_loader = get_dataloaders(data, batch_size=batch_size)

    # Fetch a single batch to verify shapes
    x_cat, x_cont, y = next(iter(train_loader))

    print(
        f"Batch Shapes -> Cat: {x_cat.shape}, Cont: {x_cont.shape}, Target: {y.shape}"
    )

    # Assertions for dimensions
    assert x_cat.shape[0] == batch_size, "Incorrect batch size for categorical data."
    assert x_cont.shape[0] == batch_size, "Incorrect batch size for continuous data."
    assert y.shape == (batch_size, 1), "Incorrect target shape."

    num_cat_features = x_cat.shape[1]
    num_cont_features = x_cont.shape[1]
    print(
        f"Features Detected -> Categorical: {num_cat_features}, Continuous: {num_cont_features}"
    )

    # 4. Demonstrate Model Initialization and Forward Pass
    print("\n[Step 3] Testing Model Architecture...")
    vocab_sizes = data["vocab_sizes"]

    # Instantiate the PIFE Model
    model = PIFEModel(vocab_sizes, num_cont_features).to(device)

    # Move batch to device
    x_cat_dev = x_cat.to(device)
    x_cont_dev = x_cont.to(device)

    # Perform forward pass
    with torch.no_grad():
        outputs = model(x_cat_dev, x_cont_dev)

    print(f"Model Output Shape: {outputs.shape}")

    # Assert output shape matches [batch_size, NUM_STREAMS]
    # The model outputs raw logits for each stream
    assert outputs.shape == (
        batch_size,
        NUM_STREAMS,
    ), f"Expected output shape {(batch_size, NUM_STREAMS)}, got {outputs.shape}"

    print("Model forward pass verified.")

    # 5. Demonstrate Training and Evaluation Pipeline
    print("\n[Step 4] Running Training Pipeline (Fast Mode)...")

    # We use max_samples to limit the training data for this demonstration,
    # ensuring the code runs quickly (within seconds/minutes).
    # We also set epochs=1.
    train_and_evaluate(
        epochs=1,
        batch_size=256,
        max_samples=2000,  # Limit to 2000 samples for speed
        load_cached_data=True,
    )

    # 6. Verify Submission File Generation
    print("\n[Step 5] Verifying Submission File...")
    submission_path = "./submission/submission.csv"

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file was not created at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"Submission File Loaded. Shape: {df_sub.shape}")

    # Validate submission content
    # The test set has 100,000 rows. Even if we train on a subset,
    # the prediction runs on the full test set provided in metadata.
    assert len(df_sub) == 100000, f"Expected 100,000 predictions, found {len(df_sub)}"
    assert "id" in df_sub.columns, "Column 'id' missing in submission."
    assert "target" in df_sub.columns, "Column 'target' missing in submission."

    # Check for valid probabilities
    assert df_sub["target"].min() >= 0.0, "Probabilities should be >= 0"
    assert df_sub["target"].max() <= 1.0, "Probabilities should be <= 1"

    print("Submission file verified successfully.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
