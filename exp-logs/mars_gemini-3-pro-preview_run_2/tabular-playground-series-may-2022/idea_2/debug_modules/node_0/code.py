import os
import torch
import pandas as pd
import numpy as np
import sys

# Import from the provided library
from library import config
from library.data_loader import get_dataloaders
from library.model import ParallelDCN
from library.trainer import Trainer
from library.inference import predict_and_submit
from library.utils import get_device


def main():
    print("=== Starting Library Usage Demonstration ===")

    # Ensure reproducibility
    torch.manual_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)

    device = get_device()
    print(f"Device detected: {device}")

    # --------------------------------------------------------------------------
    # 1. Data Loader Verification
    # --------------------------------------------------------------------------
    print("\n[1/4] Verifying Data Loaders...")

    # Load data (this will trigger preprocessing and caching if needed)
    # We use load_cached_data=True to utilize existing cache if available
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Fetch a single batch from the training loader
    batch = next(iter(train_loader))

    # Verify keys
    expected_keys = {"cat", "cont", "target"}
    assert expected_keys.issubset(
        batch.keys()
    ), f"Batch missing keys. Found: {batch.keys()}"

    # Verify shapes
    # Batch size is defined in config
    batch_size = config.BATCH_SIZE

    # Categorical: (Batch, 10)
    assert batch["cat"].shape == (
        batch_size,
        10,
    ), f"Incorrect categorical shape: {batch['cat'].shape}"

    # Continuous: (Batch, 30)
    assert batch["cont"].shape == (
        batch_size,
        30,
    ), f"Incorrect continuous shape: {batch['cont'].shape}"

    # Target: (Batch,) or (Batch, 1) - Library uses (Batch,) in dataset, unsqueezed in loop
    # In the dataset class, it returns a scalar tensor for target, so batching makes it (Batch,)
    assert (
        batch["target"].shape[0] == batch_size
    ), f"Incorrect target batch size: {batch['target'].shape}"

    print("Data Loader verification passed. Shapes are correct.")

    # --------------------------------------------------------------------------
    # 2. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n[2/4] Verifying Model Architecture...")

    model = ParallelDCN().to(device)

    # Move batch to device
    cat_x = batch["cat"].to(device)
    cont_x = batch["cont"].to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(cat_x, cont_x)

    # Verify output shape: (Batch, 1)
    assert logits.shape == (
        batch_size,
        1,
    ), f"Model output shape mismatch. Expected {(batch_size, 1)}, got {logits.shape}"

    print("Model architecture verification passed. Forward pass successful.")

    # --------------------------------------------------------------------------
    # 3. Training Loop Demonstration
    # --------------------------------------------------------------------------
    print("\n[3/4] Demonstrating Training Loop (1 Epoch)...")

    # Initialize Trainer
    trainer = Trainer(learning_rate=1e-3, weight_decay=1e-4)

    # Run fit
    # We override num_epochs to 1 for speed
    # This will train for 1 epoch, validate, and save the model if it's the best so far
    test_loader_returned = trainer.fit(num_epochs=1, patience=1, load_cached_data=True)

    # Verify model checkpoint exists
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model checkpoint was not created at {config.MODEL_SAVE_PATH}"
        )

    print(f"Training demonstration complete. Model saved to {config.MODEL_SAVE_PATH}")

    # --------------------------------------------------------------------------
    # 4. Inference and Submission Verification
    # --------------------------------------------------------------------------
    print("\n[4/4] Demonstrating Inference and Submission...")

    output_csv = "./working/demo_submission.csv"

    # Use the standalone inference function from library.inference
    # This simulates the final submission step
    predict_and_submit(
        load_cached_data=True, model_path=config.MODEL_SAVE_PATH, output_path=output_csv
    )

    # Verify the output file
    if not os.path.exists(output_csv):
        raise FileNotFoundError(f"Submission file not found at {output_csv}")

    df_sub = pd.read_csv(output_csv)

    # Check columns
    assert (
        "id" in df_sub.columns and "target" in df_sub.columns
    ), "Submission file missing required columns 'id' or 'target'"

    # Check length (Test set size is 100,000)
    assert (
        len(df_sub) == 100000
    ), f"Submission file has incorrect number of rows: {len(df_sub)}"

    # Check value range (probabilities should be between 0 and 1)
    assert (
        df_sub["target"].min() >= 0.0 and df_sub["target"].max() <= 1.0
    ), "Prediction values out of probability range [0, 1]"

    print(f"Inference verification passed. Submission generated at {output_csv}")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
