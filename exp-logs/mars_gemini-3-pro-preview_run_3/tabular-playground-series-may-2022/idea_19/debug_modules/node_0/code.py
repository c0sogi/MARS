import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.dataset import preprocess_data, ManufacturingDataset
from library.model import HPFEModel
from library.engine import train_model, predict_and_submit


def run_demo():
    print("=== Starting Demonstration of Manufacturing Control Pipeline ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment for rapid demonstration...")
    Config.set_seed(42)

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 1000  # Small subset for demo
    Config.MAX_EPOCHS = 2  # Minimal epochs to prove training loop works
    Config.BATCH_SIZE = 128  # Reasonable batch size for 1000 samples
    Config.PATIENCE = 1  # Aggressive early stopping

    # Ensure working directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Data Preprocessing
    # -------------------------------------------------------------------------
    print("\n[Step 2] Preprocessing data...")
    # This will load raw data from ./metadata, sample it, and apply feature engineering
    train_df, val_df, test_df, metadata = preprocess_data(
        load_cached_data=False, debug=True
    )

    # Validation Checks
    assert (
        len(train_df) == Config.DEBUG_SAMPLES
    ), f"Train set size mismatch: {len(train_df)}"
    assert len(val_df) == Config.DEBUG_SAMPLES, f"Val set size mismatch: {len(val_df)}"
    assert (
        len(test_df) == Config.DEBUG_SAMPLES
    ), f"Test set size mismatch: {len(test_df)}"
    assert "vocab_sizes" in metadata, "Metadata missing vocab_sizes"
    assert (
        "unique_character_count" in train_df.columns
    ), "Feature engineering failed (unique_character_count missing)"

    print(
        f"Data Loaded: Train={train_df.shape}, Val={val_df.shape}, Test={test_df.shape}"
    )

    # -------------------------------------------------------------------------
    # 3. Dataset and DataLoader Creation
    # -------------------------------------------------------------------------
    print("\n[Step 3] Creating Datasets and DataLoaders...")

    train_dataset = ManufacturingDataset(train_df, metadata, is_test=False)
    val_dataset = ManufacturingDataset(val_df, metadata, is_test=False)
    test_dataset = ManufacturingDataset(test_df, metadata, is_test=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Use 0 workers for simple script stability
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Verify a batch
    sample_batch = next(iter(train_loader))
    assert "cont" in sample_batch and "cat" in sample_batch and "target" in sample_batch
    print("DataLoader verification successful.")

    # -------------------------------------------------------------------------
    # 4. Model Initialization and Logic Check
    # -------------------------------------------------------------------------
    print("\n[Step 4] Initializing HPFE Model...")
    model = HPFEModel(metadata)
    model.to(Config.DEVICE)

    # Dry run forward pass
    print("Performing dry-run forward pass...")
    with torch.no_grad():
        cont_x = sample_batch["cont"].to(Config.DEVICE)
        cat_x = sample_batch["cat"].to(Config.DEVICE)
        logits_list = model(cont_x, cat_x)

    # Check output structure
    assert isinstance(logits_list, list), "Model output should be a list"
    assert len(logits_list) == 5, f"Expected 5 streams, got {len(logits_list)}"
    assert logits_list[0].shape == (
        cont_x.shape[0],
        1,
    ), f"Logit shape mismatch: {logits_list[0].shape}"
    print("Model architecture verification successful.")

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    print("\n[Step 5] Starting Training Loop...")
    trained_model = train_model(model, train_loader, val_loader)

    assert trained_model is not None, "Training function returned None"
    print("Training loop completed successfully.")

    # -------------------------------------------------------------------------
    # 6. Prediction and Submission
    # -------------------------------------------------------------------------
    print("\n[Step 6] Generating Predictions and Submission...")
    predict_and_submit(trained_model, test_loader)

    # -------------------------------------------------------------------------
    # 7. Final Validation of Submission File
    # -------------------------------------------------------------------------
    print("\n[Step 7] Validating Submission File...")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check shape
    expected_rows = Config.DEBUG_SAMPLES
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Check columns
    assert list(sub_df.columns) == [
        "id",
        "target",
    ], f"Invalid columns: {sub_df.columns}"

    # Check ID types
    assert pd.api.types.is_integer_dtype(sub_df["id"]), "ID column should be integer"

    # Check probability range
    preds = sub_df["target"]
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions out of probability range [0, 1]"

    print(f"Submission file validated: {Config.SUBMISSION_PATH}")
    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    run_demo()
