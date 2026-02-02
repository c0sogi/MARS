import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.data_utils import get_dataloaders
from library.model import DSDN
from library.train_utils import run_training, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets fixed seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("Initializing Demonstration Script...")
    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed
    # -------------------------------------------------------------------------
    # We modify the Config class attributes directly to optimize for a quick demo run.
    print("Configuring hyperparameters for fast execution...")

    Config.DEBUG = True  # Use a small subset (1000 samples)
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 32  # Small batch size for debug

    # Reduce Model Complexity for speed
    Config.NUM_TRANSFORMER_LAYERS = 1
    Config.EMBED_DIM = 32
    Config.MLP_HIDDEN_DIM = 64
    Config.DIM_FEEDFORWARD = 64
    Config.NUM_HEADS = 2

    # Ensure working directories exist
    _ = Config(make_dirs=True)

    # -------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading Data...")

    # Load data using the library function.
    # debug=True ensures we only get 1000 samples, making this very fast.
    train_loader, val_loader, test_loader, vocab_size, ids_test = get_dataloaders(
        load_cached_data=True, debug=True
    )

    # Fetch a single batch to verify shapes and extract dimensions
    sample_batch = next(iter(train_loader))
    x_num_sample = sample_batch["numerical_features"]
    x_seq_sample = sample_batch["sequence_features"]
    y_sample = sample_batch["target"]

    num_features = x_num_sample.shape[1]
    seq_len = x_seq_sample.shape[1]

    print(f"  - Vocab Size: {vocab_size}")
    print(f"  - Numerical Features: {num_features}")
    print(f"  - Sequence Length: {seq_len}")
    print(f"  - Batch Size: {x_num_sample.shape[0]}")

    # Assertions to verify data integrity
    assert x_num_sample.dim() == 2, "Numerical features should be 2D (Batch, Features)"
    assert x_seq_sample.dim() == 2, "Sequence features should be 2D (Batch, Seq_Len)"
    assert y_sample.dim() == 1, "Targets should be 1D"
    print("  -> Data loading verified.")

    # -------------------------------------------------------------------------
    # 3. Model Instantiation & Forward Pass Verification
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = DSDN(num_features=num_features, vocab_size=vocab_size, seq_len=seq_len)
    model.to(device)
    model.train()  # Set to train mode to enable masking logic

    # Move sample batch to device
    x_num_dev = x_num_sample.to(device)
    x_seq_dev = x_seq_sample.to(device)

    # Perform forward pass
    outputs = model(x_num_dev, x_seq_dev)

    # Verify outputs
    logits = outputs["logits"]
    num_pred = outputs["num_pred"]
    seq_pred = outputs["seq_pred"]
    mask_indices = outputs["mask_indices"]

    # Check shapes
    assert logits.shape == (
        x_num_sample.shape[0],
        1,
    ), f"Logits shape mismatch: {logits.shape}"
    assert (
        num_pred.shape == x_num_sample.shape
    ), "Numerical reconstruction shape mismatch"
    assert seq_pred.shape == (
        x_num_sample.shape[0],
        seq_len,
        vocab_size,
    ), "Sequence reconstruction shape mismatch"
    assert mask_indices is not None, "Mask indices should be generated in training mode"

    print("  -> Model forward pass successful. Output shapes are correct.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[Step 3] Running Training Loop (2 Epochs)...")

    # run_training handles the optimizer, scheduler, and loop
    trained_model = run_training(
        train_loader, val_loader, vocab_size, num_features, seq_len
    )

    # Verify model checkpoint exists
    assert os.path.exists(
        Config.MODEL_PATH
    ), f"Model checkpoint not found at {Config.MODEL_PATH}"
    print(f"  -> Training complete. Best model saved to {Config.MODEL_PATH}")

    # -------------------------------------------------------------------------
    # 5. Submission Generation & Verification
    # -------------------------------------------------------------------------
    print("\n[Step 4] Generating Submission...")

    # Generate submission using the trained model
    generate_submission(test_loader, vocab_size, num_features, seq_len, ids_test)

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check submission format
    expected_cols = ["id", "target"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}"

    # Check length (should match test_loader size, which is 1000 in debug mode)
    assert len(df_sub) == len(
        ids_test
    ), f"Submission row count mismatch. Expected {len(ids_test)}, got {len(df_sub)}"

    # Check value range
    assert (
        df_sub["target"].min() >= 0.0 and df_sub["target"].max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    print(f"  -> Submission generated successfully with {len(df_sub)} rows.")
    print("\nAll demonstration steps completed successfully!")


if __name__ == "__main__":
    main()
