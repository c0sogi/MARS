import os
import shutil
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.features import extract_features
from library.dataset import RNADataset
from library.model import StabilizedWideBiGRU
from library.loss_metric import masked_mse_loss, mcrmse
import library.runner as runner


def run_demo():
    print("=== Starting RNA Degradation Prediction Demo ===")

    # 1. Setup and Configuration Override
    # We modify the Config class attributes directly to ensure the demo runs quickly
    # and uses a separate working directory for this execution.
    print("\n[1] Configuring environment...")

    runner.set_seed(42)

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 32  # Use only 32 samples
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size

    # Redirect outputs to a demo folder
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up any previous demo run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # 2. Feature Extraction Verification
    print("\n[2] Verifying Feature Extraction Logic...")

    # Load a tiny slice of raw metadata manually to test the feature extractor
    raw_df = (
        pd.read_parquet(os.path.join(Config.METADATA_DIR, "train.parquet"))
        .iloc[:5]
        .reset_index(drop=True)
    )

    # Run extraction
    features = extract_features(raw_df, "demo_test", load_cached=False)

    # Verify shapes
    # RWPE shape: (N, Seq_Len, n_steps)
    expected_rwpe_shape = (5, Config.SEQ_LENGTH, len(Config.RWPE_STEPS))
    assert (
        features["rwpe"].shape == expected_rwpe_shape
    ), f"RWPE shape mismatch. Expected {expected_rwpe_shape}, got {features['rwpe'].shape}"

    # Pair encoding shape: (N, Seq_Len, embed_dim)
    expected_pair_shape = (5, Config.SEQ_LENGTH, Config.EMBED_DIM_PAIR)
    assert (
        features["pair_enc"].shape == expected_pair_shape
    ), f"Pair encoding shape mismatch. Expected {expected_pair_shape}, got {features['pair_enc'].shape}"

    print("    Feature extraction shapes verified.")

    # 3. Dataset Verification
    print("\n[3] Verifying Dataset Logic...")

    # Initialize dataset (this will use the Config.DEBUG settings)
    train_dataset = RNADataset(split="train", load_cached=False)

    assert (
        len(train_dataset) == Config.DEBUG_SUBSET_SIZE
    ), f"Dataset length mismatch. Expected {Config.DEBUG_SUBSET_SIZE}, got {len(train_dataset)}"

    # Get a sample
    sample = train_dataset[0]

    # Verify sample keys
    required_keys = ["seq", "loop", "rwpe", "pair_enc", "targets", "mask", "id"]
    for k in required_keys:
        assert k in sample, f"Missing key {k} in dataset sample"

    # Verify tensor types and shapes for a single sample
    assert sample["seq"].shape == (Config.SEQ_LENGTH,), "Sequence ID shape incorrect"
    assert sample["targets"].shape == (
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), "Targets shape incorrect"
    assert sample["mask"].shape == (Config.SEQ_LENGTH,), "Mask shape incorrect"

    print("    Dataset structure and shapes verified.")

    # 4. Model and Metric Verification
    print("\n[4] Verifying Model and Metric Logic...")

    device = torch.device("cpu")  # Use CPU for simple logic check
    model = StabilizedWideBiGRU().to(device)

    # Create a small batch
    loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    seq = batch["seq"].to(device)
    loop = batch["loop"].to(device)
    rwpe = batch["rwpe"].to(device)
    pair_enc = batch["pair_enc"].to(device)
    targets = batch["targets"].to(device)
    mask = batch["mask"].to(device)

    # Forward pass
    preds = model(seq, loop, rwpe, pair_enc)

    # Check output shape: (Batch, Seq, Num_Targets)
    assert preds.shape == (
        4,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), f"Model output shape mismatch. Got {preds.shape}"

    # Compute Loss
    loss = masked_mse_loss(preds, targets, mask)
    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"

    # Compute Metric
    metric = mcrmse(preds, targets, mask)
    assert metric.dim() == 0, "Metric should be a scalar"
    assert metric.item() >= 0, "Metric should be non-negative"

    print(
        f"    Forward pass successful. Loss: {loss.item():.4f}, MCRMSE: {metric.item():.4f}"
    )

    # 5. Full Pipeline Execution
    print("\n[5] Executing Full Training Pipeline (Demo)...")

    # runner.train_model() uses the Config we modified globally
    runner.train_model()

    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model file was not created."
    print("    Training complete. Model saved.")

    print("\n[6] Executing Inference Pipeline (Demo)...")

    # runner.predict_and_submit() uses the Config we modified globally
    runner.predict_and_submit()

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created."

    # Verify submission format
    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"

    # Check row count: N_samples * Seq_Length
    # Note: Dataset is sliced in DEBUG mode, so test set is also sliced
    expected_rows = Config.DEBUG_SUBSET_SIZE * Config.SEQ_LENGTH
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    print(f"    Inference complete. Submission generated at {Config.SUBMISSION_FILE}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
