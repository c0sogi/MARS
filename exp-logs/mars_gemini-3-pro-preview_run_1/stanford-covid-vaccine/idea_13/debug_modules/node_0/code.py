import os
import shutil
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import library components
from library.config import Config
from library.utils import seed_everything, mcrmse
from library.data import (
    sequence_to_kmers,
    get_pair_distances,
    encode_loop_types,
    get_dataloaders,
    process_dataframe,
)
from library.model import KmerBiGRU, SinusoidalDistanceEmbedding
from library.engine import fit, generate_submission, masked_mse_loss


def create_subset_data(source_path, dest_path, n_samples=50):
    """Creates a small subset of the parquet data for demonstration."""
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file {source_path} not found.")

    df = pd.read_parquet(source_path)
    # Take a small subset
    df_subset = df.head(n_samples).copy()
    df_subset.to_parquet(dest_path, index=False)
    return len(df_subset)


def test_data_utils():
    """Validates the low-level data processing functions."""
    print("Testing data utility functions...")

    # 1. Test Sequence to K-mers
    seq = "AGGGU"
    # k=3.
    # i=0 (A): pad-A-G -> invalid/pad? implementation handles boundaries.
    # Let's check the implementation logic in library/data.py:
    # It uses a centered window.
    # For "AGGGU", len=5.
    # i=0: window -1 to 2 -> "padAG" -> contains pad -> 0
    # i=1: window 0 to 3 -> "AGG" -> valid
    # i=2: window 1 to 4 -> "GGG" -> valid
    # i=3: window 2 to 5 -> "GGU" -> valid
    # i=4: window 3 to 6 -> "GUpad" -> 0
    kmers = sequence_to_kmers(seq, k=3)
    assert len(kmers) == 5
    assert kmers[0] == 0
    assert kmers[-1] == 0
    assert kmers[1] != 0  # Should be valid index

    # 2. Test Pair Distances
    # Structure: .((..))
    # Indices:   0123456
    # Pairs: (1, 6), (2, 5). 3,4 are unpaired. 0 is unpaired.
    # Dist at 1: 6 - 1 = 5
    # Dist at 6: 1 - 6 = -5
    # Dist at 2: 5 - 2 = 3
    # Dist at 5: 2 - 5 = -3
    structure = ".((..))"
    dists = get_pair_distances(structure)
    assert len(dists) == 7
    assert dists[0] == 0
    assert dists[1] == 5.0
    assert dists[6] == -5.0
    assert dists[2] == 3.0
    assert dists[3] == 0

    # 3. Test Loop Types
    loop_str = "EEESSSM"
    encoded = encode_loop_types(loop_str)
    assert len(encoded) == 7
    # E->2, S->6, M->5 based on library mapping
    assert encoded[0] == 2
    assert encoded[3] == 6
    assert encoded[6] == 5

    print("Data utility functions verified.")


def main():
    # 1. Setup
    print("Initializing demonstration...")
    seed_everything(42)

    # Define working directory for this demo
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # 2. Override Config for Speed and Path Redirection
    print("Configuring environment...")
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Point to subset files we are about to create
    Config.TRAIN_DATA_PATH = os.path.join(demo_dir, "train_subset.parquet")
    Config.VAL_DATA_PATH = os.path.join(demo_dir, "val_subset.parquet")
    Config.TEST_DATA_PATH = os.path.join(demo_dir, "test_subset.parquet")

    # Reduce training load
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.HIDDEN_DIM = 64  # Reduce model size for speed
    Config.NUM_LAYERS = 2

    # 3. Create Data Subsets
    print("Creating data subsets...")
    # Use original metadata paths to source data
    orig_train_path = "./metadata/train.parquet"
    orig_val_path = "./metadata/val.parquet"
    orig_test_path = "./metadata/test.parquet"

    n_train = create_subset_data(orig_train_path, Config.TRAIN_DATA_PATH, n_samples=32)
    n_val = create_subset_data(orig_val_path, Config.VAL_DATA_PATH, n_samples=16)
    n_test = create_subset_data(orig_test_path, Config.TEST_DATA_PATH, n_samples=16)

    print(f"Subsets created: Train={n_train}, Val={n_val}, Test={n_test}")

    # 4. Run Unit Tests
    test_data_utils()

    # 5. Data Loading
    print("Loading data via DataLoaders...")
    # Force reprocessing by ensuring cache dir is fresh (handled by rmtree above)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify Batch Structure
    batch = next(iter(train_loader))
    seq_inputs = batch["seq_inputs"]
    pair_dists = batch["pair_dists"]
    targets = batch["targets"]

    assert seq_inputs.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH)
    assert pair_dists.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH)
    assert targets.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH, Config.NUM_TARGETS)
    print(f"Batch shapes verified: Inputs {seq_inputs.shape}, Targets {targets.shape}")

    # 6. Model Initialization & Forward Pass
    print("Initializing Model...")
    device = Config.DEVICE
    model = KmerBiGRU().to(device)

    # Forward pass check
    print("Running forward pass check...")
    with torch.no_grad():
        preds = model(
            seq_inputs.to(device), pair_dists.to(device), batch["loop_types"].to(device)
        )

    assert preds.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH, Config.NUM_TARGETS)
    print("Forward pass successful.")

    # 7. Training Loop
    print("Starting Training Loop (2 Epochs)...")
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    trained_model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.NUM_EPOCHS,
    )

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    print("Training complete.")

    # 8. Metric Verification
    print("Verifying Metric Calculation...")
    # Create dummy tensors
    y_true = torch.randn(4, 107, 3)
    y_pred = y_true.clone()  # Perfect prediction
    score = mcrmse(y_true, y_pred, num_scored=68)
    assert score < 1e-6, f"Perfect prediction should have near-zero error, got {score}"

    y_pred_off = y_true + 1.0
    # RMSE of 1.0 everywhere -> MCRMSE should be 1.0
    score_off = mcrmse(y_true, y_pred_off, num_scored=68)
    assert abs(score_off - 1.0) < 1e-5, f"Expected 1.0, got {score_off}"
    print("Metric calculation verified.")

    # 9. Inference & Submission
    print("Generating Submission...")
    generate_submission(
        model=trained_model,
        loader=test_loader,
        device=device,
        output_path=Config.SUBMISSION_PATH,
    )

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Expected rows: n_test_samples * seq_length
    expected_rows = n_test * Config.SEQ_LENGTH
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(sub_df)}"

    # Expected columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert (
        list(sub_df.columns) == expected_cols
    ), "Submission columns do not match requirements."

    # Check if dummy columns are 0
    assert (sub_df["deg_pH10"] == 0).all(), "deg_pH10 should be 0"
    assert (sub_df["deg_50C"] == 0).all(), "deg_50C should be 0"

    print("Submission generated and verified successfully.")
    print("\nAll demonstration steps completed successfully!")


if __name__ == "__main__":
    main()
