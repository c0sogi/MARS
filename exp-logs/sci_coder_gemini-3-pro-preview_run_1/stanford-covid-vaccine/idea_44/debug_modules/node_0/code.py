import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, mcrmse
from library.features import (
    tokenize_sequence,
    tokenize_loop,
    get_couples,
    get_signed_distance_vector,
    sinusoidal_encoding,
)
from library.dataset import load_data, RNADataset
from library.model import DualStreamBiGRU
from library.engine import run_training, generate_submission


def main():
    # 1. Setup and Configuration
    warnings.filterwarnings("ignore")
    seed_everything(42)
    print("=== RNA Degradation Prediction Pipeline Demo ===\n")

    print("1. Configuring for Demo Speed...")
    # Override Config for fast execution
    Config.EPOCHS = 1
    Config.BACKBONE_LAYERS = 1  # Reduce model depth
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo

    # Ensure working directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    print(f"   Epochs set to: {Config.EPOCHS}")
    print(f"   Backbone Layers set to: {Config.BACKBONE_LAYERS}")

    # 2. Verify Metric (MCRMSE)
    print("\n2. Verifying Metric (MCRMSE)...")
    # Test Case: 2 samples, 3 targets
    # Target 1 RMSE: 0.1, Target 2 RMSE: 0.2, Target 3 RMSE: 0.0 -> Mean: 0.1
    y_true = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    y_pred = np.array([[1.1, 2.2, 3.0], [1.1, 2.2, 3.0]])

    score = mcrmse(y_true, y_pred)
    expected_score = 0.1
    assert np.isclose(
        score, expected_score
    ), f"MCRMSE calculation error. Got {score}, expected {expected_score}"

    # Test with PyTorch Tensors
    score_t = mcrmse(torch.tensor(y_true), torch.tensor(y_pred))
    assert np.isclose(score_t, expected_score), "MCRMSE tensor calculation error."
    print("   MCRMSE verification passed.")

    # 3. Verify Feature Engineering
    print("\n3. Verifying Feature Engineering...")

    # Test Structure Parsing: "((..))"
    # Indices: 0,1 (open), 2,3 (unpaired), 4,5 (close)
    # Pairs: 0-5, 1-4
    structure = "((..))"
    pairs = get_couples(structure)
    expected_pairs = np.array([5, 4, -1, -1, 1, 0])
    assert np.array_equal(pairs, expected_pairs), f"get_couples failed. Got {pairs}"

    # Test Signed Distance: Partner Index - Current Index
    # 0: 5-0=5, 1: 4-1=3, 2: 0, 3: 0, 4: 1-4=-3, 5: 0-5=-5
    dists = get_signed_distance_vector(structure)
    expected_dists = np.array([5, 3, 0, 0, -3, -5])
    assert np.array_equal(
        dists, expected_dists
    ), f"get_signed_distance_vector failed. Got {dists}"

    # Test Sinusoidal Encoding
    dim = 4
    emb = sinusoidal_encoding(dists, dim)
    assert emb.shape == (len(structure), dim), f"Embedding shape mismatch: {emb.shape}"
    # Unpaired (dist=0) should yield sin(0)=0, cos(0)=1 -> [0, 1, 0, 1]
    expected_zero_emb = np.array([0.0, 1.0, 0.0, 1.0])
    assert np.allclose(emb[2], expected_zero_emb), "Zero distance embedding incorrect."
    print("   Feature engineering verification passed.")

    # 4. Verify Dataset Loading
    print("\n4. Verifying Dataset Loading (Debug Mode)...")
    # debug=True loads only 100 samples
    train_ds = load_data("train", debug=True)
    assert (
        len(train_ds) == 100
    ), f"Expected 100 samples in debug mode, got {len(train_ds)}"

    sample = train_ds[0]
    required_keys = {"seq", "loop", "dist", "targets", "id"}
    assert required_keys.issubset(
        sample.keys()
    ), "Dataset sample missing required keys."

    # Verify Tensor Shapes
    # Seq: (107,), Dist: (107, 64), Targets: (107, 3)
    assert sample["seq"].shape == (
        Config.SEQ_LEN,
    ), f"Seq shape error: {sample['seq'].shape}"
    assert sample["dist"].shape == (
        Config.SEQ_LEN,
        Config.EMB_DIM_DIST,
    ), f"Dist shape error: {sample['dist'].shape}"
    assert sample["targets"].shape == (
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Targets shape error: {sample['targets'].shape}"
    print("   Dataset loading verification passed.")

    # 5. Verify Model Architecture
    print("\n5. Verifying Model Architecture...")
    model = DualStreamBiGRU()
    model.eval()

    # Create a dummy batch
    batch_size = 2
    dummy_input = {
        "seq": torch.zeros((batch_size, Config.SEQ_LEN), dtype=torch.long),
        "loop": torch.zeros((batch_size, Config.SEQ_LEN), dtype=torch.long),
        "dist": torch.zeros(
            (batch_size, Config.SEQ_LEN, Config.EMB_DIM_DIST), dtype=torch.float32
        ),
    }

    with torch.no_grad():
        output = model(dummy_input)

    # Output should be (Batch, Seq_Len, Num_Targets)
    expected_shape = (batch_size, Config.SEQ_LEN, Config.NUM_TARGETS)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Got {output.shape}, expected {expected_shape}"
    print("   Model architecture verification passed.")

    # 6. Run Training Pipeline
    print("\n6. Running Training Pipeline (Debug Mode)...")
    # Executes the training loop defined in library.engine
    best_model_path = run_training(debug=True)

    assert os.path.exists(best_model_path), "Best model file was not created."
    print(f"   Training complete. Model saved to: {best_model_path}")

    # 7. Generate Submission
    print("\n7. Generating Submission...")
    # Generates predictions on the test set (debug mode = 100 samples)
    generate_submission(best_model_path, debug=True)

    assert os.path.exists(Config.SUBMISSION_FILE_PATH), "Submission file not created."

    # Verify Submission Content
    sub_df = pd.read_csv(Config.SUBMISSION_FILE_PATH)

    # Expected rows: 100 samples * 107 positions = 10700
    expected_rows = 100 * 107
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Got {len(sub_df)}, expected {expected_rows}"

    # Check Columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch."

    # Check that unscored columns are strictly 0.0 as per logic
    assert (sub_df["deg_pH10"] == 0).all(), "deg_pH10 column should be 0.0"
    assert (sub_df["deg_50C"] == 0).all(), "deg_50C column should be 0.0"

    print("   Submission generation verification passed.")
    print("\n=== All demonstrations and verifications completed successfully. ===")


if __name__ == "__main__":
    main()
