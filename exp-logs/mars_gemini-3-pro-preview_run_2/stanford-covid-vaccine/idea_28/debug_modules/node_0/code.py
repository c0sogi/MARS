import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library files
from library.config import Config, set_seed, device
from library.utils import parse_list_string, get_parsed_metadata
from library.data import get_dataset, RNADataset, process_data
from library.model import SRDN
from library.loss import MCRMSELoss
from library.train import train_model


def run_demo():
    # 1. Setup and Configuration Overrides
    print("=== Setting up Demo Configuration ===")

    # Override directories to keep demo isolated
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir)

    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir

    # Set seeds for reproducibility
    set_seed(42)
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {device}")

    # 2. Verify Utility Functions
    print("\n=== Verifying Utility Functions ===")

    # Test parse_list_string
    test_str = "[0.1, 0.2, 0.3]"
    parsed = parse_list_string(test_str)
    assert isinstance(parsed, np.ndarray), "Parsed object should be a numpy array"
    assert np.allclose(
        parsed, np.array([0.1, 0.2, 0.3], dtype=np.float32)
    ), "Parsing failed correctness check"
    print("parse_list_string: OK")

    # Test Metadata Loading (Small subset)
    # We load a tiny sample from the actual metadata file to ensure integration works
    df_meta = get_parsed_metadata(mode="train", sample_size=10)
    assert len(df_meta) == 10, "Metadata sampling failed"
    assert "reactivity" in df_meta.columns, "Metadata missing target columns"
    assert isinstance(
        df_meta.iloc[0]["reactivity"], np.ndarray
    ), "Metadata target parsing failed"
    print("get_parsed_metadata: OK")

    # 3. Verify Model Architecture
    print("\n=== Verifying Model Architecture (SRDN) ===")

    batch_size = 2
    seq_len = Config.SEQ_LENGTH
    # Input channels = 19 (static) + 5 (recycling) = 24
    # Note: The model expects 24 channels in the recycling loop, but the initial input
    # from dataset has 19. The recycling logic happens in the training loop/forward wrapper.
    # Here we simulate the input inside the training loop (concatenated).
    input_channels = Config.NUM_INPUT_CHANNELS  # 24

    # Create dummy input
    dummy_input = torch.randn(batch_size, seq_len, input_channels).to(device)

    # Create dummy partner indices (all unpaired = -1)
    dummy_pmap = torch.full((batch_size, seq_len), -1, dtype=torch.long).to(device)

    model = SRDN().to(device)
    model.eval()

    with torch.no_grad():
        output = model(dummy_input, dummy_pmap)

    # Check output shape: (Batch, Seq_Len, Num_Targets)
    expected_shape = (batch_size, seq_len, Config.NUM_TARGETS)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"
    print("SRDN Forward Pass: OK")

    # 4. Verify Loss Function
    print("\n=== Verifying Loss Function (MCRMSELoss) ===")

    criterion = MCRMSELoss()

    # Create dummy predictions and targets
    # Perfect prediction should yield 0 loss
    preds = torch.ones(batch_size, seq_len, Config.NUM_TARGETS).to(device)
    targets = torch.ones(batch_size, seq_len, Config.NUM_TARGETS).to(device)

    loss = criterion(preds, targets)
    assert torch.is_tensor(loss), "Loss should be a tensor"
    assert (
        loss.item() == 0.0
    ), f"Loss should be 0.0 for perfect predictions, got {loss.item()}"

    # Imperfect prediction
    preds_bad = preds + 1.0
    loss_bad = criterion(preds_bad, targets)
    # MCRMSE of difference 1.0 is 1.0
    assert np.isclose(
        loss_bad.item(), 1.0, atol=1e-5
    ), f"Loss should be 1.0, got {loss_bad.item()}"
    print("MCRMSELoss: OK")

    # 5. Run Full Training Pipeline (Demo Mode)
    print("\n=== Running Training Pipeline (Demo) ===")

    # We use a very small subset and 1 epoch to keep it fast
    debug_size = 32
    epochs = 2

    # This function handles data loading, model training, validation, and submission generation
    train_model(
        epochs=epochs,
        batch_size=8,
        patience=2,
        load_cached_data=False,  # Force processing to verify data pipeline
        debug_sample_size=debug_size,
    )

    # 6. Verify Submission Output
    print("\n=== Verifying Submission Output ===")

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)

    # Check columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    # Check row count
    # The test set has 240 samples. Each has 107 positions.
    # Total rows = 240 * 107 = 25680
    expected_rows = 240 * Config.SEQ_LENGTH
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Check for NaNs
    assert not df_sub.isnull().values.any(), "Submission contains NaN values"

    print(f"Submission verified. Shape: {df_sub.shape}")
    print("\nSUCCESS: All demo components executed correctly.")


if __name__ == "__main__":
    run_demo()
