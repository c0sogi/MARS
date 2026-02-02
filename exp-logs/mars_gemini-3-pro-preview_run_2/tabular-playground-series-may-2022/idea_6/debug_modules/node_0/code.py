import os
import torch
import numpy as np
import pandas as pd
import shutil
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.dataset import get_datasets, CharTokenizer, ManufacturingDataset
from library.model import ResFunnelGLU
from library.trainer import Trainer, set_seed


def main():
    print("Starting Library Usage Demonstration...")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 2000  # Small subset for speed
    Config.MAX_EPOCHS = 2  # Minimal epochs to test loop
    Config.BATCH_SIZE = 64  # Smaller batch size for small subset
    Config.PATIENCE = 1

    # Redirect outputs to a demo directory to avoid overwriting production artifacts
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update dependent paths
    Config.PROCESSED_DATA_PATH = os.path.join(Config.WORKING_DIR, "processed_data.npz")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Set seeds for reproducibility
    set_seed(Config.SEED)
    print("Configuration updated. Random seeds set.")

    # --------------------------------------------------------------------------
    # 2. Tokenizer Verification
    # --------------------------------------------------------------------------
    print("\n[2] Verifying CharTokenizer logic...")
    tokenizer = CharTokenizer()

    # Test case: 'ABC' should map to [1, 2, 3]
    test_series = pd.Series(["ABC", "Z"])
    tokens = tokenizer.transform(test_series)

    # Check shape
    assert tokens.shape == (
        2,
        Config.F27_SEQ_LEN,
    ), f"Tokenizer output shape mismatch. Expected (2, {Config.F27_SEQ_LEN}), got {tokens.shape}"

    # Check values (A=1, B=2, C=3)
    # Note: The tokenizer pads/truncates based on string length, but here we just check the first chars
    assert tokens[0, 0] == 1, "Character 'A' did not map to 1"
    assert tokens[0, 1] == 2, "Character 'B' did not map to 2"
    assert tokens[0, 2] == 3, "Character 'C' did not map to 3"
    assert tokens[1, 0] == 26, "Character 'Z' did not map to 26"

    print("CharTokenizer verification passed.")

    # --------------------------------------------------------------------------
    # 3. Dataset Loading & Verification
    # --------------------------------------------------------------------------
    print("\n[3] Loading and verifying datasets...")

    # Force reprocessing to ensure pipeline works from scratch
    if os.path.exists(Config.PROCESSED_DATA_PATH):
        os.remove(Config.PROCESSED_DATA_PATH)

    train_ds, val_ds, test_ds = get_datasets(load_cached_data=False, debug=True)

    # Verify types
    assert isinstance(train_ds, ManufacturingDataset)
    assert isinstance(val_ds, ManufacturingDataset)
    assert isinstance(test_ds, ManufacturingDataset)

    # Verify lengths (should match DEBUG_SAMPLES)
    assert (
        len(train_ds) == Config.DEBUG_SAMPLES
    ), f"Train ds length {len(train_ds)} != {Config.DEBUG_SAMPLES}"
    assert (
        len(val_ds) == Config.DEBUG_SAMPLES
    ), f"Val ds length {len(val_ds)} != {Config.DEBUG_SAMPLES}"
    # Test ds might be smaller if the raw file is small, but usually it respects the limit
    assert len(test_ds) <= Config.DEBUG_SAMPLES

    # Verify data item structure
    sample = train_ds[0]
    assert "cont" in sample
    assert "cat" in sample
    assert "target" in sample

    # Check feature dimensions
    assert (
        sample["cont"].shape[0] == Config.NUM_CONT_FEATURES
    ), "Continuous feature dimension mismatch"
    assert (
        sample["cat"].shape[0] == Config.F27_SEQ_LEN
    ), "Categorical feature dimension mismatch"

    print(f"Dataset verification passed. Train size: {len(train_ds)}")

    # --------------------------------------------------------------------------
    # 4. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture (ResFunnelGLU)...")

    model = ResFunnelGLU().to(Config.DEVICE)

    # Create dummy batch
    batch_size = 4
    dummy_cont = torch.randn(batch_size, Config.NUM_CONT_FEATURES).to(Config.DEVICE)
    dummy_cat = torch.randint(
        0, Config.VOCAB_SIZE, (batch_size, Config.F27_SEQ_LEN)
    ).to(Config.DEVICE)

    # Forward pass
    output = model(dummy_cont, dummy_cat)

    # Check output shape (Batch, 1)
    assert output.shape == (
        batch_size,
        1,
    ), f"Model output shape mismatch. Expected ({batch_size}, 1), got {output.shape}"

    # Check output range [0, 1] (Sigmoid)
    assert (output >= 0).all() and (
        output <= 1
    ).all(), "Model output out of probability range [0, 1]"

    print("Model forward pass verification passed.")

    # --------------------------------------------------------------------------
    # 5. Training Loop Verification
    # --------------------------------------------------------------------------
    print("\n[5] Executing Training Loop (Trainer)...")

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Initialize Trainer
    trainer = Trainer(model, device=Config.DEVICE)

    # Run Fit
    best_auc = trainer.fit(
        train_loader, val_loader, epochs=Config.MAX_EPOCHS, patience=Config.PATIENCE
    )

    # Verify result
    assert isinstance(best_auc, float), "Trainer.fit did not return a float AUC score"
    assert 0 <= best_auc <= 1, f"AUC score {best_auc} is invalid"

    # Verify model save
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"

    print(f"Training loop verification passed. Best AUC: {best_auc:.4f}")

    # --------------------------------------------------------------------------
    # 6. Inference & Submission Verification
    # --------------------------------------------------------------------------
    print("\n[6] Generating Submission...")

    # To ensure submission generation works with the metadata logic, we need to make sure
    # the test_metadata.csv corresponds to the test dataset we loaded.
    # Since we are in DEBUG mode, the dataset is truncated.
    # The trainer.generate_submission loads test_metadata.csv to get IDs.
    # If len(preds) != len(metadata), it prints a warning but saves.
    # For this demo, we just want to ensure it runs without crashing and produces a file.

    trainer.generate_submission(test_loader)

    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Verify submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "id" in df_sub.columns and "target" in df_sub.columns
    ), "Submission missing required columns"
    assert len(df_sub) > 0, "Submission file is empty"

    # Check values
    assert (
        df_sub["target"].min() >= 0 and df_sub["target"].max() <= 1
    ), "Submission targets out of range"

    print(f"Submission verification passed. File saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {df_sub.shape}")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
