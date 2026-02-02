import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, MCRMSELoss, compute_mcrmse
from library.data import get_dataloaders, RNADataset
from library.model import RNAGRUModel
from library.train import train_model


def main():
    print("=== Starting RNA Degradation Prediction Demo ===\n")

    # 1. Setup Configuration
    # We use debug=True for speed (fewer epochs, smaller data subset)
    print("1. Initializing Configuration...")
    config = Config(debug=True, epochs=2, batch_size=8)

    # Override working directory for this demo to keep things clean
    config.working_dir = "./working/demo_execution"
    os.makedirs(config.working_dir, exist_ok=True)

    # Update paths in config based on new working dir
    config.train_cache_path = os.path.join(config.working_dir, "train_cache.npy")
    config.val_cache_path = os.path.join(config.working_dir, "val_cache.npy")
    config.test_cache_path = os.path.join(config.working_dir, "test_cache.npy")
    config.model_save_path = os.path.join(config.working_dir, "best_model.pth")
    config.submission_path = os.path.join(config.working_dir, "submission.csv")

    print(config.get_config_info())
    set_seed(config.seed)
    print("Configuration initialized.\n")

    # 2. Data Pipeline Verification
    print("2. Verifying Data Pipeline...")
    # get_dataloaders handles preprocessing and caching
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=False
    )

    # Fetch one batch from train loader
    features, targets = next(iter(train_loader))

    print(f"  Feature Batch Shape: {features.shape}")
    print(f"  Target Batch Shape: {targets.shape}")

    # Assertions for dimensions
    # Features: (Batch, Seq_Len=107, Channels=14)
    assert features.shape == (
        config.batch_size,
        107,
        14,
    ), f"Expected feature shape ({config.batch_size}, 107, 14), got {features.shape}"

    # Targets: (Batch, Seq_Scored=68, Num_Targets=5)
    assert targets.shape == (
        config.batch_size,
        68,
        5,
    ), f"Expected target shape ({config.batch_size}, 68, 5), got {targets.shape}"

    print("Data Pipeline verification passed.\n")

    # 3. Model Architecture Verification
    print("3. Verifying Model Architecture...")
    device = config.device if torch.cuda.is_available() else "cpu"
    model = RNAGRUModel(config).to(device)

    # Create dummy input
    dummy_input = torch.randn(config.batch_size, 107, 14).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Model Output Shape: {output.shape}")

    # Assertions for output dimensions
    # Model outputs full sequence length predictions: (Batch, Seq_Len=107, Num_Targets=5)
    assert output.shape == (
        config.batch_size,
        107,
        5,
    ), f"Expected output shape ({config.batch_size}, 107, 5), got {output.shape}"

    print("Model Architecture verification passed.\n")

    # 4. Loss Function Verification
    print("4. Verifying Loss Function (MCRMSE)...")
    criterion = MCRMSELoss()

    # Create dummy predictions and targets (flattened as per training loop logic)
    # Simulating a batch of predictions for the scored sequence length (68)
    # Shape: (Batch * Seq_Scored, Num_Targets)
    n_samples = config.batch_size * 68
    dummy_preds = torch.ones(n_samples, 5) * 0.5
    dummy_targets = torch.zeros(n_samples, 5)

    loss = criterion(dummy_preds, dummy_targets)
    print(f"  Calculated Loss: {loss.item():.4f}")

    # Expected MCRMSE for constant 0.5 error: sqrt(0.5^2) = 0.5
    expected_loss = 0.5
    assert (
        abs(loss.item() - expected_loss) < 1e-5
    ), f"Expected loss ~{expected_loss}, got {loss.item()}"

    print("Loss Function verification passed.\n")

    # 5. Full Training Loop Integration
    print("5. Running Full Training Loop (Debug Mode)...")
    # This runs training, validation, and generates submission
    train_model(config)

    # Verify artifacts exist
    assert os.path.exists(config.model_save_path), "Model file was not saved."
    assert os.path.exists(config.submission_path), "Submission file was not generated."
    print("Training loop completed successfully.\n")

    # 6. Submission File Verification
    print("6. Verifying Submission File...")
    sub_df = pd.read_csv(config.submission_path)
    print(f"  Submission Shape: {sub_df.shape}")
    print(f"  Submission Columns: {list(sub_df.columns)}")

    # Check required columns
    required_cols = ["id_seqpos"] + config.target_cols
    for col in required_cols:
        assert col in sub_df.columns, f"Missing column in submission: {col}"

    # Check number of rows
    # In debug mode, we process a subset of data.
    # The get_dataloaders function with debug=True loads head(100) of test data.
    # However, the test.parquet has 240 rows.
    # The library code `get_dataloaders` -> `load_or_process` calls `df.head(100)` if config.debug is True.
    # So we expect 100 samples * 68 predictions = 6800 rows.
    # Let's verify based on the actual test loader size used in generation.

    # Note: The provided `generate_submission` loads IDs from `config.test_metadata_path`.
    # If `train_model` used the debug data loader, but `generate_submission` reads the full parquet
    # to get IDs, there might be a mismatch if not handled carefully in the library.
    # Looking at `generate_submission` in `library/train.py`:
    #   test_df = pd.read_parquet(config.test_metadata_path)
    #   ids = test_df["id"].values
    # It reads the FULL metadata file.
    # But `test_loader` was created with debug data (100 samples).
    # This would cause an index error in `generate_submission` if the loop over IDs exceeds predictions.
    # However, since we are running this code to demonstrate usage, if the library code has this potential bug
    # in debug mode, we will see it crash.
    # Wait, `generate_submission` iterates: `for i, sample_id in enumerate(ids):`.
    # `all_preds` comes from `test_loader`.
    # If `len(ids)` (240) > `len(all_preds)` (100), `all_preds[i]` will raise IndexError.

    # To fix this for the demo without modifying library code, we must ensure the test metadata
    # read inside `generate_submission` matches the loader.
    # Since we cannot modify `library/train.py`, we rely on the fact that `train_model` calls `get_dataloaders`.
    # If `config.debug` is True, `get_dataloaders` subsets the data.
    # The `generate_submission` function reads the original parquet file.
    # This implies `config.debug=True` might fail in `generate_submission` unless we patch the metadata path
    # or if the library handles it (it doesn't seem to).

    # Workaround for the demo:
    # We will create a subset test parquet file and point the config to it before running training.
    print("  (Adjusting test metadata for debug consistency...)")
    full_test_df = pd.read_parquet(config.test_metadata_path)
    debug_test_df = full_test_df.head(100)  # Match the debug subset size in data.py
    debug_test_meta_path = os.path.join(config.working_dir, "test_subset.parquet")
    debug_test_df.to_parquet(debug_test_meta_path)
    config.test_metadata_path = debug_test_meta_path

    # We also need to do this for train/val to be safe, though `train_model` doesn't reload them from parquet for IDs.
    # But let's be consistent.
    full_train_df = pd.read_parquet(config.train_metadata_path)
    debug_train_df = full_train_df.head(100)
    debug_train_meta_path = os.path.join(config.working_dir, "train_subset.parquet")
    debug_train_df.to_parquet(debug_train_meta_path)
    config.train_metadata_path = debug_train_meta_path

    full_val_df = pd.read_parquet(config.val_metadata_path)
    debug_val_df = full_val_df.head(100)
    debug_val_meta_path = os.path.join(config.working_dir, "val_subset.parquet")
    debug_val_df.to_parquet(debug_val_meta_path)
    config.val_metadata_path = debug_val_meta_path

    # Re-run training with consistent metadata
    print("  (Re-running training with consistent debug metadata...)")
    train_model(config)

    # Now verify submission again
    sub_df = pd.read_csv(config.submission_path)
    expected_rows = 100 * 68
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(sub_df)}"

    print("Submission verification passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
