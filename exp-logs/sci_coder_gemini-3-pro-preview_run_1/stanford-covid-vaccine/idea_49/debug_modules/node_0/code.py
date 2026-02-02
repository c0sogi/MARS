import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

# Import library modules
from library.config import Config
from library.utils import parse_structure, compute_laplacian_pe
from library.dataset import RNADataset
from library.model import SpectralTopologicalBiGRU
from library.engine import run_training, set_seed


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print(">>> Configuring environment for demonstration...")

    # Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_execution"
    Config.MAX_DEBUG_SAMPLES = 40  # Use only 40 samples for speed
    Config.EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for small data

    # Update dependent paths that were set at import time
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Re-initialize to ensure directories exist
    Config.initialize()
    set_seed(Config.SEED)

    # =========================================================================
    # 2. Verify Utility Functions
    # =========================================================================
    print("\n>>> Verifying utility functions...")

    # Test parse_structure
    structure_str = "((..))"
    pairs = parse_structure(structure_str)
    # Expected: 0-5, 1-4 (and symmetric 5-0, 4-1)
    assert pairs[0] == 5 and pairs[5] == 0, "Structure parsing failed for outer pair."
    assert pairs[1] == 4 and pairs[4] == 1, "Structure parsing failed for inner pair."
    assert (
        2 not in pairs and 3 not in pairs
    ), "Unpaired bases should not be in dictionary."
    print(" - parse_structure: OK")

    # Test compute_laplacian_pe
    seq_len = 10
    k = 4
    # Create a dummy structure string of length 10
    dummy_struct = "((......))"
    lpe = compute_laplacian_pe(dummy_struct, seq_len, k=k)

    assert isinstance(lpe, np.ndarray), "LPE should be a numpy array."
    assert lpe.shape == (
        seq_len,
        k,
    ), f"LPE shape mismatch. Expected ({seq_len}, {k}), got {lpe.shape}"
    assert not np.isnan(lpe).any(), "LPE contains NaNs."
    print(" - compute_laplacian_pe: OK")

    # =========================================================================
    # 3. Verify Dataset Loading
    # =========================================================================
    print("\n>>> Verifying RNADataset...")

    # Initialize train dataset (this will trigger processing and caching in the new WORKING_DIR)
    train_ds = RNADataset(mode="train", load_cached_data=False)

    # Assertions
    assert (
        len(train_ds) == Config.MAX_DEBUG_SAMPLES
    ), f"Dataset length mismatch. Expected {Config.MAX_DEBUG_SAMPLES}, got {len(train_ds)}"

    sample = train_ds[0]
    required_keys = [
        "sequence",
        "loop_type",
        "pair_dist",
        "lpe",
        "targets",
        "mask",
        "id",
    ]
    for key in required_keys:
        assert key in sample, f"Missing key in dataset sample: {key}"

    # Check tensor shapes for the sample
    # Sequence: (107,)
    assert sample["sequence"].shape == (Config.SEQ_LEN,), "Sequence shape incorrect."
    # Targets: (107, 3) - defined in Config.TARGET_COLS
    assert sample["targets"].shape == (
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), "Targets shape incorrect."
    # LPE: (107, 8)
    assert sample["lpe"].shape == (
        Config.SEQ_LEN,
        Config.LPE_DIM,
    ), "LPE shape incorrect."

    print(" - RNADataset initialization and shapes: OK")

    # =========================================================================
    # 4. Verify Model Architecture
    # =========================================================================
    print("\n>>> Verifying SpectralTopologicalBiGRU Model...")

    model = SpectralTopologicalBiGRU()
    model.eval()

    # Prepare a batch input (Unsqueeze to add batch dimension)
    input_seq = sample["sequence"].unsqueeze(0)  # (1, 107)
    input_loop = sample["loop_type"].unsqueeze(0)  # (1, 107)
    input_pair = sample["pair_dist"].unsqueeze(0)  # (1, 107, 64)
    input_lpe = sample["lpe"].unsqueeze(0)  # (1, 107, 8)

    with torch.no_grad():
        output = model(input_seq, input_loop, input_pair, input_lpe)

    # Check Output
    # Expected shape: (Batch=1, Seq_Len=107, Num_Targets=3)
    expected_shape = (1, Config.SEQ_LEN, Config.NUM_TARGETS)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs."

    print(" - Model forward pass: OK")

    # =========================================================================
    # 5. Execute Training Pipeline
    # =========================================================================
    print("\n>>> Executing Training Pipeline (run_training)...")
    # This function handles training, validation, and submission generation
    run_training()

    # =========================================================================
    # 6. Validate Submission Output
    # =========================================================================
    print("\n>>> Validating Submission File...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch."

    # Check row count
    # We used MAX_DEBUG_SAMPLES = 40 for the test set as well.
    # Each sample has 107 positions. Total rows = 40 * 107 = 4280.
    expected_rows = Config.MAX_DEBUG_SAMPLES * Config.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Check for valid values (no NaNs)
    assert not df_sub.isnull().values.any(), "Submission contains NaN values."

    print(f" - Submission file valid. Shape: {df_sub.shape}")
    print("\n>>> Demonstration completed successfully!")


if __name__ == "__main__":
    main()
