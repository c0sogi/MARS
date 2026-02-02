import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config, seed_everything
from library.data import get_structure_distance, load_data, RNADataset
from library.model import RNAModel, SinusoidalPositionalEmbedding
from library.train import run_training
from library.utils import mcrmse


def run_demo():
    print("=== Starting RNA Degradation Library Demo ===\n")

    # 1. Setup Configuration for Demo
    # We override paths to keep demo artifacts separate and ensure speed
    print("[Setup] Configuring environment...")
    Config.CACHE_DIR = "./working/demo_run/"
    Config.SUBMISSION_DIR = "./working/demo_run/submission/"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Logic Verification: Structure Distance
    print("\n[Verification] Testing get_structure_distance logic...")
    # Test case: "((..))"
    # Index 0 '(' pairs with 5 ')': dist 5
    # Index 1 '(' pairs with 4 ')': dist 3
    # Index 2 '.' unpaired: dist 0
    # Index 3 '.' unpaired: dist 0
    # Index 4 ')' pairs with 1 '(': dist -3
    # Index 5 ')' pairs with 0 '(': dist -5
    test_structure = "((..))"
    expected_dist = np.array([5, 3, 0, 0, -3, -5], dtype=np.int32)
    calculated_dist = get_structure_distance(test_structure)

    np.testing.assert_array_equal(calculated_dist, expected_dist)
    print("  -> Structure distance calculation verified.")

    # 3. Component Verification: Model
    print("\n[Verification] Testing Model Components...")

    # 3a. Sinusoidal Embeddings
    d_model = 64
    pos_emb_layer = SinusoidalPositionalEmbedding(d_model).to(device)
    dummy_dists = torch.tensor([5.0, -5.0, 0.0], device=device)  # Shape (3,)
    emb_out = pos_emb_layer(dummy_dists)  # Shape (3, 64)

    assert emb_out.shape == (3, d_model), f"Embedding shape mismatch: {emb_out.shape}"
    # Check property: sin(-x) = -sin(x), cos(-x) = cos(x)
    # The embedding concatenates [sin, cos].
    # So for 5 vs -5: first half should be negated, second half identical.
    half_dim = d_model // 2
    # Check sin part (negated)
    assert torch.allclose(
        emb_out[0, :half_dim], -emb_out[1, :half_dim], atol=1e-5
    ), "Sin component symmetry failed"
    # Check cos part (identical)
    assert torch.allclose(
        emb_out[0, half_dim:], emb_out[1, half_dim:], atol=1e-5
    ), "Cos component symmetry failed"
    print("  -> Sinusoidal embeddings verified.")

    # 3b. Full Model Forward Pass
    model = RNAModel(Config).to(device)
    batch_size = 2
    seq_len = Config.SEQ_LENGTH

    # Create dummy inputs
    dummy_seq = torch.randint(0, 4, (batch_size, seq_len)).to(device)
    dummy_loop = torch.randint(0, 7, (batch_size, seq_len)).to(device)
    dummy_dist = torch.randn(batch_size, seq_len).to(device)

    output = model(dummy_seq, dummy_loop, dummy_dist)

    expected_shape = (batch_size, seq_len, 3)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Got {output.shape}, expected {expected_shape}"
    print("  -> Model forward pass verified.")

    # 4. Integration Test: Full Training Pipeline (Debug Mode)
    print("\n[Integration] Running Training Pipeline (Debug Mode)...")
    # run_training with debug=True uses a subset of 100 samples
    run_training(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=True)

    # 5. Output Verification
    print("\n[Verification] Validating Submission Output...")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  -> Submission loaded. Shape: {sub_df.shape}")

    # Expected rows: In debug mode, we load 'test' data.
    # However, run_training loads the *full* test set even in debug mode for inference
    # (debug slicing usually applies to train/val to speed up training).
    # Let's check the test data size to confirm.
    test_ids, _, _, _ = load_data("test")
    expected_rows = len(test_ids) * Config.SEQ_LENGTH

    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Check columns
    expected_cols = ["id_seqpos"] + Config.ALL_PRED_COLS
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    # Check that unscored columns are 0.0 (deg_pH10, deg_50C)
    # These are indices 3 and 5 in the CSV (0-based: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C)
    # Names: deg_pH10, deg_50C
    assert (sub_df["deg_pH10"] == 0.0).all(), "deg_pH10 column should be all zeros"
    assert (sub_df["deg_50C"] == 0.0).all(), "deg_50C column should be all zeros"

    print("  -> Submission format verified.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
