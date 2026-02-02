import sys
import os
import torch
import numpy as np
import pandas as pd
import warnings

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed
from library.data import parse_structure_to_distance
from library.model import RNAModel
from library.train import run_training

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def test_data_processing_logic():
    """
    Verifies the logic of the structure parsing function which converts
    dot-bracket notation into discrete topological distances.
    """
    print("1. Verifying Data Processing Logic...")

    # Test Case: Simple hairpin structure '((..))'
    # Indices:   012345
    # Pairs:     (0, 5) and (1, 4)
    # Unpaired:  2, 3
    #
    # Logic from library.data.parse_structure_to_distance:
    #   Clip = 32. Offset = 32.
    #   Unpaired (dist 0) -> 32
    #   Index 0 (paired to 5): dist = 5-0 = 5.   Token = 5 + 32 = 37
    #   Index 1 (paired to 4): dist = 4-1 = 3.   Token = 3 + 32 = 35
    #   Index 2 (unpaired):                      Token = 32
    #   Index 3 (unpaired):                      Token = 32
    #   Index 4 (paired to 1): dist = 1-4 = -3.  Token = -3 + 32 = 29
    #   Index 5 (paired to 0): dist = 0-5 = -5.  Token = -5 + 32 = 27

    structure = "((..))"
    seq_len = 6
    clip = 32
    expected_tokens = np.array([37, 35, 32, 32, 29, 27])

    # Execute function
    result = parse_structure_to_distance(structure, seq_len=seq_len, clip=clip)

    # Assertions
    assert len(result) == seq_len, f"Expected length {seq_len}, got {len(result)}"
    np.testing.assert_array_equal(
        result, expected_tokens, err_msg="Structure parsing logic mismatch"
    )

    print("   [Passed] Structure parsing logic verified.")


def test_model_logic():
    """
    Verifies the model instantiation and forward pass shapes.
    """
    print("2. Verifying Model Architecture...")

    # Use CPU for this quick logic check
    device = torch.device("cpu")
    model = RNAModel(Config).to(device)
    model.eval()

    # Create dummy batch
    batch_size = 4
    seq_len = Config.SEQ_LENGTH  # 107

    # Generate random inputs within vocabulary ranges
    dummy_seq = torch.randint(0, Config.VOCAB_SIZE_SEQ, (batch_size, seq_len)).to(
        device
    )
    dummy_loop = torch.randint(0, Config.VOCAB_SIZE_LOOP, (batch_size, seq_len)).to(
        device
    )
    dummy_dist = torch.randint(0, Config.VOCAB_SIZE_DIST, (batch_size, seq_len)).to(
        device
    )

    # Perform forward pass
    with torch.no_grad():
        outputs = model(dummy_seq, dummy_loop, dummy_dist)

    # Check Output Shape: (Batch, Seq_Len, Num_Targets)
    # Num_Targets is 3 (reactivity, deg_Mg_pH10, deg_Mg_50C)
    expected_shape = (batch_size, seq_len, Config.NUM_TARGETS)

    assert (
        outputs.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {outputs.shape}"

    print(f"   [Passed] Model forward pass successful. Output shape: {outputs.shape}")


def run_pipeline_demo():
    """
    Runs the full training and submission pipeline using the provided library.
    Uses 'debug=True' to subset data for speed.
    """
    print("3. Running Full Training Pipeline (Demo Mode)...")

    # Configuration overrides for speed
    DEMO_EPOCHS = 2
    DEMO_BATCH_SIZE = 16
    DEMO_DEBUG = True  # Subsets data to 100 samples

    # Run the training loop
    # This function handles: Data Loading -> Training -> Validation -> Saving Best Model -> Generating Submission
    run_training(epochs=DEMO_EPOCHS, batch_size=DEMO_BATCH_SIZE, debug=DEMO_DEBUG)

    # Verify the output submission file
    submission_path = Config.SUBMISSION_PATH

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file was not created at {submission_path}")

    df_sub = pd.read_csv(submission_path)

    # Calculate expected rows: 100 samples (debug subset) * 107 positions
    expected_rows = Config.DEBUG_SUBSET_SIZE * Config.SEQ_LENGTH

    print(f"   Submission generated at: {submission_path}")
    print(f"   Submission dimensions: {df_sub.shape}")

    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Verify required columns
    required_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert (
        list(df_sub.columns) == required_cols
    ), f"Submission columns mismatch. Expected {required_cols}, got {list(df_sub.columns)}"

    print("   [Passed] Pipeline execution and submission verification successful.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # Execute demonstrations
    test_data_processing_logic()
    test_model_logic()
    run_pipeline_demo()

    print("\nAll tasks completed successfully.")
