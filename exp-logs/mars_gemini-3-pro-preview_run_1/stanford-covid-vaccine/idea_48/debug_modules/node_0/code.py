import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import library components
from library.config import Config
from library.utils import seed_everything, mcrmse
from library.features import compute_adjacency, compute_rwpe, compute_signed_distance
from library.dataset import process_dataframe, RNADataset
from library.model import TopologicalWideResBiLSTM
from library.train import run_training

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def demo_features():
    print("\n[1] Demonstrating Feature Engineering...")

    # Dummy RNA data (length 10)
    # Structure: Hairpin loop (....)
    seq_len = 10
    structure = "((....)).."

    # 1. Adjacency Matrix
    adj = compute_adjacency(structure, seq_len)
    print(f"  Adjacency Matrix Shape: {adj.shape}")

    # Validation: Matrix should be symmetric and binary
    assert adj.shape == (seq_len, seq_len), "Adjacency shape mismatch"
    assert np.allclose(adj, adj.T), "Adjacency matrix is not symmetric"
    assert np.all(np.isin(adj, [0, 1])), "Adjacency matrix is not binary"

    # 2. Random Walk Positional Encoding (RWPE)
    steps = [1, 2, 4]
    rwpe = compute_rwpe(adj, steps=steps)
    print(f"  RWPE Feature Shape: {rwpe.shape}")

    # Validation: Shape should be (seq_len, len(steps))
    assert rwpe.shape == (seq_len, len(steps)), "RWPE shape mismatch"

    # 3. Signed Distance
    dist = compute_signed_distance(structure, seq_len)
    print(f"  Signed Distance Shape: {dist.shape}")

    # Validation: Paired bases should have non-zero distance
    # Base 0 '(' is paired with Base 7 ')' -> dist should be 7 - 0 = 7
    # Base 7 ')' is paired with Base 0 '(' -> dist should be 0 - 7 = -7
    assert dist.shape == (seq_len,), "Distance shape mismatch"
    assert dist[0] == 7.0, f"Expected distance 7.0 at index 0, got {dist[0]}"
    assert dist[7] == -7.0, f"Expected distance -7.0 at index 7, got {dist[7]}"

    print("  Feature engineering validation passed.")


def demo_dataset_and_model():
    print("\n[2] Demonstrating Dataset and Model...")

    # Initialize Config
    config = Config(debug=True, batch_size=2)
    seed_everything(config.seed)

    # Load a tiny subset of training data
    if not os.path.exists(config.train_file):
        raise FileNotFoundError(f"Training file not found at {config.train_file}")

    df_sample = pd.read_parquet(config.train_file).head(4)
    print(f"  Loaded {len(df_sample)} samples from metadata.")

    # Process DataFrame
    data_dict = process_dataframe(df_sample, config, is_test=False)

    # Instantiate Dataset
    dataset = RNADataset(data_dict)

    # Instantiate DataLoader
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False
    )

    # Get one batch
    batch = next(iter(loader))

    # Validate Batch Shapes
    # Sequence: (B, 107)
    assert batch["sequence"].shape == (config.batch_size, config.seq_len)
    # RWPE: (B, 107, 5) -> 5 is default len(rwpe_steps)
    assert batch["rwpe"].shape == (
        config.batch_size,
        config.seq_len,
        len(config.rwpe_steps),
    )
    # Targets: (B, 68, 3) -> 3 targets
    assert batch["targets"].shape == (config.batch_size, config.pred_len, 3)

    print("  Batch shapes validated.")

    # Instantiate Model
    model = TopologicalWideResBiLSTM(config)
    model.to(config.device)
    model.eval()

    # Forward Pass
    with torch.no_grad():
        sequence = batch["sequence"].to(config.device)
        loop_type = batch["loop_type"].to(config.device)
        rwpe = batch["rwpe"].to(config.device)
        distance = batch["distance"].to(config.device)

        logits = model(sequence, loop_type, rwpe, distance)

    print(f"  Model Output Shape: {logits.shape}")

    # Validation: Output should be (B, 107, 3)
    # Note: Model outputs for full sequence length (107), slicing happens in loss/eval
    assert logits.shape == (
        config.batch_size,
        config.seq_len,
        3,
    ), "Model output shape mismatch"

    print("  Model forward pass validation passed.")


def demo_metric():
    print("\n[3] Demonstrating Metric (MCRMSE)...")

    # Create dummy data
    # Shape: (N_samples, N_targets)
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.1, 1.9], [3.2, 3.8]])

    # Manual Calculation:
    # Col 1: (1.0-1.1)^2 = 0.01, (3.0-3.2)^2 = 0.04 -> Mean=0.025 -> RMSE=sqrt(0.025) ~= 0.1581
    # Col 2: (2.0-1.9)^2 = 0.01, (4.0-3.8)^2 = 0.04 -> Mean=0.025 -> RMSE=sqrt(0.025) ~= 0.1581
    # MCRMSE = (0.1581 + 0.1581) / 2 = 0.1581

    score = mcrmse(y_true, y_pred)
    print(f"  Calculated MCRMSE: {score:.4f}")

    expected_rmse = np.sqrt(0.025)
    assert np.isclose(
        score, expected_rmse
    ), f"Metric calculation failed. Got {score}, expected {expected_rmse}"

    print("  Metric validation passed.")


def demo_full_training_run():
    print("\n[4] Demonstrating Full Training Pipeline (Debug Mode)...")

    # Run the training function provided in library.train
    # We use debug=True to run for limited epochs (2) and small batch size
    # This tests the integration of all components
    try:
        run_training(debug=True, epochs=1, batch_size=4)
        print("  Full training pipeline executed successfully.")
    except Exception as e:
        print(f"  Training pipeline failed with error: {e}")
        raise e


if __name__ == "__main__":
    seed_everything(42)

    # 1. Test Feature Generation logic
    demo_features()

    # 2. Test Data Loading and Model Inference
    demo_dataset_and_model()

    # 3. Test Metric Calculation
    demo_metric()

    # 4. Test End-to-End Training Pipeline
    demo_full_training_run()

    print("\nAll demonstrations completed successfully.")
