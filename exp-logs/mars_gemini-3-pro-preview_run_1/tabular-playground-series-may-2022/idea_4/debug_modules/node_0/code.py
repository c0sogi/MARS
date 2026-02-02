import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.data_utils import (
    CharTokenizer,
    ManufacturingDataset,
    get_dataloaders,
    set_seeds,
)
from library.model import HybridTransformer
from library.train_utils import run_training, predict


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Fast Demonstration
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast execution...")

    # Override Config to run a small, fast experiment
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "model_checkpoint.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Use a tiny subset of data
    Config.DEBUG_SAMPLE_SIZE = 200
    Config.BATCH_SIZE = 16
    Config.EPOCHS = 1

    # Force fresh processing to verify data pipeline logic
    Config.LOAD_CACHED_DATA = False

    # Ensure directories exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seeds for reproducibility
    set_seeds(Config.SEED)
    print("    Configuration updated: DEBUG_SAMPLE_SIZE=200, EPOCHS=1")

    # -------------------------------------------------------------------------
    # 2. Verify Tokenizer Logic
    # -------------------------------------------------------------------------
    print("\n[2] Verifying CharTokenizer...")
    tokenizer = CharTokenizer()
    sample_texts = ["ABC", "DE", "F"]
    tokenizer.fit(sample_texts)

    # Expected Vocab: A, B, C, D, E, F (sorted) -> indices 1 to 6. 0 is padding.
    # Transform with max_len=4
    seqs = tokenizer.transform(sample_texts, max_len=4)

    # Assertions
    assert isinstance(seqs, np.ndarray), "Tokenizer output should be numpy array"
    assert seqs.shape == (3, 4), f"Expected shape (3, 4), got {seqs.shape}"
    assert seqs[0, 0] != 0, "First character should not be 0 (padding)"
    assert seqs[1, 2] == 0, "Padding should be 0"

    print("    CharTokenizer verification passed.")

    # -------------------------------------------------------------------------
    # 3. Verify Dataset Logic
    # -------------------------------------------------------------------------
    print("\n[3] Verifying ManufacturingDataset...")
    # Create dummy data
    dummy_seq = np.random.randint(0, 10, (10, 5))
    dummy_num = np.random.randn(10, 3)
    dummy_target = np.random.randint(0, 2, (10,))

    ds = ManufacturingDataset(dummy_seq, dummy_num, dummy_target)

    # Check __getitem__
    s_item, n_item, t_item = ds[0]

    assert torch.is_tensor(s_item), "Sequence item should be a tensor"
    assert torch.is_tensor(n_item), "Numerical item should be a tensor"
    assert torch.is_tensor(t_item), "Target item should be a tensor"
    assert s_item.shape[0] == 5, "Sequence length mismatch"
    assert n_item.shape[0] == 3, "Numerical feature dimension mismatch"

    print("    ManufacturingDataset verification passed.")

    # -------------------------------------------------------------------------
    # 4. Data Loading & Processing
    # -------------------------------------------------------------------------
    print("\n[4] Running Data Pipeline (get_dataloaders)...")
    # This will load metadata, process features, update Config.VOCAB_SIZE, and return loaders
    train_loader, val_loader, test_loader = get_dataloaders()

    # Verify Train Loader
    batch_seq, batch_num, batch_target = next(iter(train_loader))
    assert batch_seq.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert (
        batch_seq.shape[1] == Config.MAX_SEQ_LEN
    ), "Sequence length mismatch in loader"

    # Verify Test Loader (no targets)
    test_batch = next(iter(test_loader))
    assert len(test_batch) == 2, "Test loader should return (seq, num)"

    print(f"    Data loaded successfully. Vocab Size determined: {Config.VOCAB_SIZE}")

    # -------------------------------------------------------------------------
    # 5. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[5] Verifying HybridTransformer Model...")
    model = HybridTransformer()

    # Move to CPU for simple verification
    model.to("cpu")

    # Create a dummy batch matching the config
    dummy_input_seq = torch.randint(
        0, Config.VOCAB_SIZE, (Config.BATCH_SIZE, Config.MAX_SEQ_LEN)
    )
    dummy_input_num = torch.randn(Config.BATCH_SIZE, len(Config.NUM_FEATURES))

    # Forward pass
    output = model(dummy_input_seq, dummy_input_num)

    # Assertions
    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {output.shape}"

    print("    Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 6. Run Training Loop
    # -------------------------------------------------------------------------
    print("\n[6] Executing Training Loop (run_training)...")

    # We use the imported run_training function which handles the loop, validation, and saving
    best_auc = run_training(train_loader, val_loader, epochs=Config.EPOCHS, patience=1)

    # Verify model file creation
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model file was not created at {Config.MODEL_SAVE_PATH}"
        )

    print(f"    Training completed. Best AUC: {best_auc:.4f}")
    print(f"    Model saved to: {Config.MODEL_SAVE_PATH}")

    # -------------------------------------------------------------------------
    # 7. Generate Predictions
    # -------------------------------------------------------------------------
    print("\n[7] Generating Submission (predict)...")

    # Run prediction using the trained model
    predict(test_loader)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    # Verify content format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission loaded. Shape: {df_sub.shape}")

    # Check columns
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert "target" in df_sub.columns, "Submission missing 'target' column"

    # Check row count (should match DEBUG_SAMPLE_SIZE)
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission row count {len(df_sub)} does not match debug size {Config.DEBUG_SAMPLE_SIZE}"

    print("    Submission verification passed.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
