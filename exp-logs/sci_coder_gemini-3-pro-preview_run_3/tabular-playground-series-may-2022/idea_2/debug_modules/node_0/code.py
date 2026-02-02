import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.preprocessing import DataPreprocessor
from library.dataset import get_dataloaders, ManufacturingDataset
from library.model import DCNv2
from library.engine import run_training, Engine

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets fixed seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("Starting DCNv2 Pipeline Demonstration...")
    set_seed(42)

    # ==========================================
    # 1. Patch Configuration for Speed
    # ==========================================
    print("\n[1] Patching Config for rapid demonstration...")
    # Reduce training duration
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 256
    Config.PATIENCE = 1

    # Reduce model complexity for speed
    Config.HIDDEN_LAYERS = [64, 32]
    Config.EMBEDDING_DIM = 4

    # Ensure directories exist (Config.create_dirs handles this, but good to be explicit)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Config patched: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}")

    # ==========================================
    # 2. Data Preprocessing Verification
    # ==========================================
    print("\n[2] Verifying Data Preprocessing...")
    preprocessor = DataPreprocessor()

    # Use debug=True to load a small subset of data
    train_df, val_df, test_df = preprocessor.process_data(
        load_cached_data=False, debug=True
    )

    # Assertions to verify logic
    print("   Verifying DataFrame shapes and columns...")
    assert not train_df.empty, "Train DataFrame is empty."
    assert not val_df.empty, "Validation DataFrame is empty."
    assert not test_df.empty, "Test DataFrame is empty."

    # Verify Feature Engineering
    # f_27 should be removed, char_0...char_9 should exist
    assert "f_27" not in train_df.columns, "Column 'f_27' was not removed."
    expected_char_cols = [f"char_{i}" for i in range(10)]
    for col in expected_char_cols:
        assert col in train_df.columns, f"Expected engineered column {col} missing."

    assert (
        "unique_character_count" in train_df.columns
    ), "Engineered feature 'unique_character_count' missing."

    # Verify Scaling (Continuous columns should be roughly standard normal)
    # f_00 is a continuous column
    if "f_00" in train_df.columns:
        mean_val = train_df["f_00"].mean()
        std_val = train_df["f_00"].std()
        # In a small debug sample, it won't be exactly 0 and 1, but should be reasonable
        print(f"   Sample feature 'f_00' stats: Mean={mean_val:.4f}, Std={std_val:.4f}")

    print("   Data Preprocessing verified successfully.")

    # ==========================================
    # 3. DataLoader Verification
    # ==========================================
    print("\n[3] Verifying DataLoaders...")
    # Get dataloaders using the debug flag
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True
    )

    # Fetch one batch
    batch = next(iter(train_loader))

    continuous = batch["continuous"]
    categorical = batch["categorical"]
    targets = batch["target"]

    print(f"   Batch keys: {list(batch.keys())}")
    print(f"   Continuous shape: {continuous.shape}")
    print(f"   Categorical shape: {categorical.shape}")
    print(f"   Target shape: {targets.shape}")

    # Assertions
    assert (
        continuous.shape[0] == Config.BATCH_SIZE
    ), f"Batch size mismatch. Expected {Config.BATCH_SIZE}, got {continuous.shape[0]}"
    assert (
        categorical.shape[1] == Config.STR_LEN
    ), f"Categorical seq len mismatch. Expected {Config.STR_LEN}, got {categorical.shape[1]}"
    assert targets.shape[1] == 1, "Target should have shape (B, 1)"

    print("   DataLoaders verified successfully.")

    # ==========================================
    # 4. Model Verification
    # ==========================================
    print("\n[4] Verifying Model Architecture...")
    # Instantiate model
    model = DCNv2()

    # Move to CPU for this quick check
    model.to("cpu")

    # Forward pass
    with torch.no_grad():
        outputs = model(continuous, categorical)

    print(f"   Model Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch."
    assert (outputs >= 0).all() and (
        outputs <= 1
    ).all(), "Model outputs are not valid probabilities (0-1)."

    print("   Model architecture verified successfully.")

    # ==========================================
    # 5. Full Training Loop Execution
    # ==========================================
    print("\n[5] Executing Training Loop (Engine)...")

    # We use the run_training function which handles the loop, validation, and saving.
    # We pass the loaders created with debug=True to ensure this finishes quickly.
    run_training(model, train_loader, val_loader, test_loader)

    # ==========================================
    # 6. Submission Verification
    # ==========================================
    print("\n[6] Verifying Submission Output...")

    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    sub_df = pd.read_csv(submission_path)
    print(f"   Submission shape: {sub_df.shape}")
    print(f"   First few rows:\n{sub_df.head()}")

    # Assertions
    assert Config.ID_COL in sub_df.columns, f"Missing ID column {Config.ID_COL}"
    assert (
        Config.TARGET_COL in sub_df.columns
    ), f"Missing target column {Config.TARGET_COL}"
    assert len(sub_df) > 0, "Submission file is empty."

    # Check if values are probabilities
    preds = sub_df[Config.TARGET_COL]
    assert (
        preds.min() >= 0 and preds.max() <= 1
    ), "Predictions out of probability range."

    print("\nAll demonstrations and verifications passed successfully!")


if __name__ == "__main__":
    main()
