import sys
import os
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, mcrmse_loss, parse_structure_pairs
from library.data import get_dataloaders
from library.model import DISR_BiGRU, generate_submission
from library.train import run_training


def main():
    print("==== RNA Degradation Prediction Demo ====")

    # 1. Setup Configuration
    # We use debug=True to trigger subsetting in data.py and reduced epochs
    print("\n[1] Initializing Configuration...")
    config = Config(debug=True)

    # Override paths and parameters for a self-contained demo run
    config.working_dir = "./working/demo_execution"
    os.makedirs(config.working_dir, exist_ok=True)

    # Use distinct cache files for the demo to avoid conflicts
    config.train_cache_path = os.path.join(config.working_dir, "train_cache.npy")
    config.val_cache_path = os.path.join(config.working_dir, "val_cache.npy")
    config.test_cache_path = os.path.join(config.working_dir, "test_cache.npy")
    config.model_save_path = os.path.join(config.working_dir, "best_model.pth")
    config.submission_path = os.path.join(config.working_dir, "submission.csv")

    # Optimization for speed
    config.epochs = 2
    config.batch_size = 8
    config.subset_fraction = 0.05  # Use only 5% of data
    config.num_workers = 0  # Main process only to avoid overhead

    set_seed(config.seed)
    print("    Configuration configured for fast demo execution.")

    # 2. Verify Utility Functions
    print("\n[2] Verifying Utility Logic...")

    # Test Structure Parsing
    # Structure: ((..)) -> Indices: 0 pairs 5, 1 pairs 4. 2,3 unpaired.
    struct_str = "((..))"
    pairs = parse_structure_pairs(struct_str)
    expected_pairs = np.array([5, 4, -1, -1, 1, 0], dtype=np.int32)
    assert np.array_equal(
        pairs, expected_pairs
    ), f"Structure parsing failed. Got {pairs}, expected {expected_pairs}"
    print("    Structure parsing verified.")

    # Test MCRMSE Loss
    # Perfect prediction should yield 0 loss
    y_true = torch.tensor([[[1.0, 0.5], [0.2, 0.8]]])  # (1, 2, 2)
    y_pred = torch.tensor([[[1.0, 0.5], [0.2, 0.8]]])
    loss = mcrmse_loss(y_pred, y_true)
    assert torch.isclose(
        loss, torch.tensor(0.0)
    ), "Loss calculation failed for perfect match."
    print("    MCRMSE loss function verified.")

    # 3. Data Loading
    print("\n[3] Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Fetch a single batch to verify shapes
    batch = next(iter(train_loader))
    features = batch["features"]  # (B, 107, 14)
    pair_indices = batch["pair_indices"]  # (B, 107)
    targets = batch["targets"]  # (B, 107, 5)
    mask = batch["mask"]  # (B, 107)

    print(f"    Batch Size: {features.shape[0]}")
    print(f"    Feature Shape: {features.shape}")

    # Assertions for data integrity
    assert features.shape == (
        config.batch_size,
        107,
        14,
    ), "Feature tensor shape mismatch."
    assert pair_indices.shape == (
        config.batch_size,
        107,
    ), "Pair indices shape mismatch."
    assert targets.shape == (config.batch_size, 107, 5), "Target tensor shape mismatch."
    assert mask.shape == (config.batch_size, 107), "Mask tensor shape mismatch."
    print("    Data shapes verified.")

    # 4. Model Initialization & Forward Pass
    print("\n[4] Initializing Model...")
    model = DISR_BiGRU(config).to(config.device)

    # Run forward pass
    with torch.no_grad():
        output = model(features.to(config.device), pair_indices.to(config.device))

    assert output.shape == (
        config.batch_size,
        107,
        5,
    ), f"Model output shape mismatch. Got {output.shape}"
    print("    Model initialized and forward pass successful.")

    # 5. Training Loop
    print("\n[5] Running Training Loop...")
    # run_training handles the full loop: train -> validate -> save best model
    run_training(config)

    # Verify model artifact creation
    if not os.path.exists(config.model_save_path):
        raise FileNotFoundError(
            f"Training failed to save model at {config.model_save_path}"
        )
    print("    Training completed. Best model saved.")

    # 6. Submission Generation
    print("\n[6] Generating Submission...")
    generate_submission(config)

    # Verify submission file
    if not os.path.exists(config.submission_path):
        raise FileNotFoundError(
            f"Submission file not found at {config.submission_path}"
        )

    sub_df = pd.read_csv(config.submission_path)

    # Calculate expected rows: num_test_samples * seq_len (107)
    # Note: test_loader is also subsetted because debug=True
    num_test_samples = len(test_loader.dataset)
    expected_rows = num_test_samples * 107

    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    expected_cols = ["id_seqpos"] + config.target_cols
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(sub_df.columns)}"

    print(f"    Submission saved to {config.submission_path}")
    print(f"    Submission shape: {sub_df.shape}")
    print("\n==== Demo Execution Completed Successfully ====")


if __name__ == "__main__":
    main()
