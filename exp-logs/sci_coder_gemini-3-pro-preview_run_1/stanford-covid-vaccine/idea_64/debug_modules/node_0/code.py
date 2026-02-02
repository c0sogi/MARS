import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

# Import from provided library
from library.config import Config
from library.utils import set_seed
from library.features import (
    Tokenizer,
    compute_pair_distance,
    get_sinusoidal_encoding,
    load_data,
)
from library.data import RNADataset, get_dataloaders
from library.model import StructureInjectedWideResBiLSTM
from library.engine import train_model


def main():
    print("Initializing demonstration script...")

    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE

    # 2. Verify Feature Engineering Components
    print("\n--- Verifying Feature Engineering ---")

    # Test Tokenizer
    seq_vocab = {"A": 0, "G": 1, "C": 2, "U": 3}
    tokenizer = Tokenizer(seq_vocab)
    test_seq = ["ACGU"]
    tokens = tokenizer.transform(test_seq)
    expected_tokens = np.array([[0, 2, 1, 3]])
    np.testing.assert_array_equal(
        tokens, expected_tokens, err_msg="Tokenizer failed mapping."
    )
    print("Tokenizer verification passed.")

    # Test Pair Distance Computation
    # Structure: ((..)) -> Indices: 012345
    # Pair (1,4): dist at 1 is 3, at 4 is -3
    # Pair (0,5): dist at 0 is 5, at 5 is -5
    # Unpaired 2,3: dist 0
    test_struct = ["((..))"]
    dists = compute_pair_distance(test_struct)
    expected_dists = np.array([[5, 3, 0, 0, -3, -5]])
    np.testing.assert_array_equal(
        dists, expected_dists, err_msg="Pair distance computation failed."
    )
    print("Structure distance verification passed.")

    # Test Sinusoidal Encoding
    dist_tensor = torch.tensor(dists)
    embed_dim = 64
    enc = get_sinusoidal_encoding(dist_tensor, embed_dim)
    assert enc.shape == (1, 6, embed_dim), f"Encoding shape mismatch: {enc.shape}"
    print("Sinusoidal encoding verification passed.")

    # 3. Verify Data Loading and Dataset
    print("\n--- Verifying Data Loading ---")

    # Load a tiny subset of training data (using cache if available or processing raw)
    # We force max_samples in get_dataloaders, but let's test load_data raw first
    raw_data = load_data(split="train", load_cached_data=True)
    assert "seq" in raw_data and "targets" in raw_data, "Missing keys in loaded data."
    print(f"Loaded raw data with {len(raw_data['seq'])} samples.")

    # Create a DataLoader with a small subset
    batch_size = 4
    train_loader = get_dataloaders(
        split="train", batch_size=batch_size, max_samples=20, shuffle=False
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    seq, loop, dist, targets = (
        batch["seq"],
        batch["loop"],
        batch["dist"],
        batch["targets"],
    )

    # Assert shapes
    # Seq: (B, 107)
    assert seq.shape == (batch_size, Config.SEQ_LEN), f"Seq shape error: {seq.shape}"
    # Dist: (B, 107)
    assert dist.shape == (batch_size, Config.SEQ_LEN), f"Dist shape error: {dist.shape}"
    # Targets: (B, 68, 3)
    assert targets.shape == (
        batch_size,
        Config.PRED_LEN,
        Config.NUM_CLASSES,
    ), f"Target shape error: {targets.shape}"

    print("DataLoader and Dataset verification passed.")

    # 4. Verify Model Architecture
    print("\n--- Verifying Model Architecture ---")
    model = StructureInjectedWideResBiLSTM()
    model.to(device)

    # Move batch to device
    seq = seq.to(device)
    loop = loop.to(device)
    dist = dist.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(seq, loop, dist)

    # Output shape should be (B, 107, 3) - model outputs full sequence length
    expected_shape = (batch_size, Config.SEQ_LEN, Config.NUM_CLASSES)
    assert (
        outputs.shape == expected_shape
    ), f"Model output shape mismatch. Got {outputs.shape}, expected {expected_shape}"
    print("Model forward pass verification passed.")

    # 5. Run Training Pipeline (Integration Test)
    print("\n--- Running Training Pipeline (Fast Mode) ---")
    # We use a very small number of samples and epochs to ensure this finishes within seconds/minutes
    max_samples_run = 50
    epochs_run = 2

    # This function handles training loop, validation, saving checkpoint, and generating submission
    train_model(max_epochs=epochs_run, max_samples=max_samples_run)

    # 6. Verify Outputs
    print("\n--- Verifying Pipeline Outputs ---")

    # Check Model Checkpoint
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Checkpoint found at: {best_model_path}")
        # Verify we can load it
        ckpt = torch.load(best_model_path, map_location="cpu")
        assert ckpt is not None, "Checkpoint file is empty or invalid."
    else:
        raise FileNotFoundError(f"Checkpoint not found at {best_model_path}")

    # Check Submission File
    submission_path = "./submission/submission.csv"
    if os.path.exists(submission_path):
        print(f"Submission found at: {submission_path}")
        df_sub = pd.read_csv(submission_path)

        # Verify columns
        expected_cols = [
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]
        assert (
            list(df_sub.columns) == expected_cols
        ), f"Submission columns mismatch. Got {list(df_sub.columns)}"

        # Verify rows
        # We processed the full test set in generate_submission (240 samples * 107 length)
        # Note: train_model calls generate_submission which loads the TEST set.
        # The test set has 240 samples. 240 * 107 = 25680 rows.
        expected_rows = 240 * 107
        assert (
            len(df_sub) == expected_rows
        ), f"Submission row count mismatch. Got {len(df_sub)}, expected {expected_rows}"
        print("Submission file content verification passed.")
    else:
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    print("\nAll demonstrations and verifications completed successfully.")


if __name__ == "__main__":
    main()
