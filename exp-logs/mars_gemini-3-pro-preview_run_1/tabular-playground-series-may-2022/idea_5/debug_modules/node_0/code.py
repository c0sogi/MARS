import sys
import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import UnifiedTransformer
from library.engine import train_model, generate_submission


def main():
    print("Starting demonstration script...")

    # --------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed and Demonstration
    # --------------------------------------------------------------------------
    print("Configuring environment for fast execution...")

    # Set a specific project name for this demo to isolate outputs
    Config.PROJECT_NAME = "demo_execution"

    # Update paths based on new project name
    Config.WORKING_DIR = os.path.join("./working", Config.PROJECT_NAME)
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "model_checkpoint.pth")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Enable Debug Mode to use a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 2000  # Small enough for CPU/Fast execution

    # Reduce Training Parameters
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 32
    Config.PATIENCE = 2

    # Reduce Model Complexity for Speed
    Config.EMBED_DIM = 32
    Config.NUM_HEADS = 2
    Config.NUM_TRANSFORMER_LAYERS = 1
    Config.MLP_HIDDEN_LAYERS = [64, 32]

    # Re-run setup to create new directories
    Config.setup()

    # Set Random Seed
    seed_everything(Config.SEED)

    # --------------------------------------------------------------------------
    # 2. Data Loading and Verification
    # --------------------------------------------------------------------------
    print("Loading data (Debug Mode)...")

    # Force reload to ensure we process the small debug subset
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        load_cached_data=False,  # Force reprocessing for the demo subset
        num_workers=0,  # Avoid multiprocessing overhead for small demo
    )

    print("Verifying DataLoader structure...")
    # Fetch one batch to verify structure
    batch = next(iter(train_loader))

    # Check keys
    assert "numerical" in batch, "Batch missing 'numerical' key"
    assert "sequence" in batch, "Batch missing 'sequence' key"
    assert "target" in batch, "Batch missing 'target' key"

    # Check shapes
    # Numerical: (Batch, 30) - 30 features (f_00..f_30 excluding f_27)
    assert batch["numerical"].shape == (
        Config.BATCH_SIZE,
        30,
    ), f"Incorrect numerical shape: {batch['numerical'].shape}"

    # Sequence: (Batch, Max_Len)
    assert batch["sequence"].shape == (
        Config.BATCH_SIZE,
        Config.MAX_SEQ_LEN,
    ), f"Incorrect sequence shape: {batch['sequence'].shape}"

    # Target: (Batch,)
    assert batch["target"].shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect target shape: {batch['target'].shape}"

    print("Data structure verified successfully.")

    # --------------------------------------------------------------------------
    # 3. Model Instantiation and Logic Verification
    # --------------------------------------------------------------------------
    print("Instantiating model...")
    model = UnifiedTransformer()

    # Move to configured device
    device = torch.device(Config.DEVICE)
    model.to(device)

    print("Verifying model forward pass...")
    # Create dummy inputs on the correct device
    dummy_num = torch.randn(Config.BATCH_SIZE, 30).to(device)
    # Create dummy sequence (integers 0-26)
    dummy_seq = torch.randint(
        0, Config.VOCAB_SIZE, (Config.BATCH_SIZE, Config.MAX_SEQ_LEN)
    ).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_num, dummy_seq)

    # Verify output shape (Batch_Size,)
    assert output.shape == (
        Config.BATCH_SIZE,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE,)}, got {output.shape}"

    # Verify output range (Sigmoid should be 0-1)
    assert (
        output.min() >= 0.0 and output.max() <= 1.0
    ), "Model output values out of range [0, 1]"

    print("Model logic verified successfully.")

    # --------------------------------------------------------------------------
    # 4. Training Loop Execution
    # --------------------------------------------------------------------------
    print("Executing training loop...")

    # Run the training engine
    best_auc = train_model(model, train_loader, val_loader, device)

    # Verify a model file was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model checkpoint file was not created."
    print(f"Training finished. Best AUC: {best_auc:.4f}")

    # --------------------------------------------------------------------------
    # 5. Submission Generation
    # --------------------------------------------------------------------------
    print("Generating submission...")

    generate_submission(model, test_loader, device, output_path=Config.SUBMISSION_PATH)

    # --------------------------------------------------------------------------
    # 6. Final Output Verification
    # --------------------------------------------------------------------------
    print("Verifying submission file...")

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check shape
    # In debug mode, we might process fewer rows if the test set was sampled or if the loader dropped last.
    # However, get_dataloaders with DEBUG=True usually slices the input dataframe.
    # The test loader in `data.py` uses the sliced dataframe.
    # Let's verify against the actual test loader dataset size.
    expected_rows = len(test_loader.dataset)
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Check columns
    assert list(df_sub.columns) == [
        "id",
        "target",
    ], f"Submission columns mismatch. Expected ['id', 'target'], got {list(df_sub.columns)}"

    # Check value types
    assert pd.api.types.is_integer_dtype(df_sub["id"]), "ID column is not integer."
    assert pd.api.types.is_float_dtype(df_sub["target"]), "Target column is not float."

    print("Submission verified successfully.")
    print("Demonstration complete.")


if __name__ == "__main__":
    main()
