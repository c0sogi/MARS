import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import config
from library.utils import seed_everything, calculate_mcrmse
from library.data import load_or_process_data, RNADataset
from library.model import HC_BD_BiGRU
from library.train import Trainer
from torch.utils.data import DataLoader


def run_demo():
    # 1. Setup and Configuration Overrides for Speed
    print("Setting up demo configuration...")
    seed_everything(42)

    # Create a separate directory for demo outputs
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override config paths
    config.WORKING_DIR = demo_dir
    config.MODEL_SAVE_PATH = os.path.join(demo_dir, "demo_model.pth")
    config.SUBMISSION_PATH = os.path.join(demo_dir, "submission_demo.csv")

    # Override hyperparameters for fast execution (Tiny Model)
    config.HIDDEN_DIM = 32
    config.STEM_FILTERS = 16
    config.BOTTLENECK_DIM = 8
    config.NUM_LAYERS = 2
    config.EPOCHS = 1
    config.BATCH_SIZE = 4
    config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo

    print(f"Working directory: {config.WORKING_DIR}")
    print(f"Device: {config.DEVICE}")

    # 2. Data Loading and Slicing (Subset)
    print("\nLoading and slicing data...")

    # Load raw dictionaries (this handles the parquet reading and processing)
    # We force load_cached_data=False to ensure we test the processing logic at least once,
    # or rely on the library's caching mechanism.
    train_data_full = load_or_process_data(
        "train", config.TRAIN_PATH, load_cached_data=True
    )
    val_data_full = load_or_process_data("val", config.VAL_PATH, load_cached_data=True)
    test_data_full = load_or_process_data(
        "test", config.TEST_PATH, load_cached_data=True
    )

    # Helper to slice dictionary arrays
    def slice_data_dict(data_dict, num_samples):
        sliced = {}
        for k, v in data_dict.items():
            if (
                v is not None
                and hasattr(v, "__len__")
                and len(v) == len(data_dict["ids"])
            ):
                sliced[k] = v[:num_samples]
            else:
                sliced[k] = v
        return sliced

    # Keep only 20 samples for this demo
    SUBSET_SIZE = 20
    train_data_subset = slice_data_dict(train_data_full, SUBSET_SIZE)
    val_data_subset = slice_data_dict(val_data_full, SUBSET_SIZE)
    test_data_subset = slice_data_dict(test_data_full, SUBSET_SIZE)

    print(f"Train subset size: {len(train_data_subset['ids'])}")

    # Create Datasets
    train_dataset = RNADataset(train_data_subset)
    val_dataset = RNADataset(val_data_subset)
    test_dataset = RNADataset(test_data_subset)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 3. Model Verification
    print("\nVerifying model architecture...")
    model = HC_BD_BiGRU()
    model.to(config.DEVICE)

    # Get a batch
    batch = next(iter(train_loader))
    seq = batch["sequence"].to(config.DEVICE)
    pidx = batch["pair_indices"].to(config.DEVICE)
    pmask = batch["pair_mask"].to(config.DEVICE)

    # Forward pass
    output = model(seq, pidx, pmask)

    # Check output shape: (Batch, SeqLen, NumTargets) -> (4, 107, 5)
    expected_shape = (config.BATCH_SIZE, config.SEQ_LEN, config.NUM_TARGETS)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"
    print("Model forward pass successful. Output shape verified.")

    # 4. Metric Verification
    print("\nVerifying metric calculation (MCRMSE)...")
    # Create dummy predictions and targets
    # Shape: (Batch=2, Seq=107, Targets=5)
    # Target Shape: (Batch=2, SeqScored=68, Targets=5)

    dummy_preds = np.zeros((2, 107, 5))
    dummy_targets = np.zeros((2, 68, 5))

    # Let's set a specific error for the scored columns
    # Scored cols indices in [reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
    # are [0, 1, 3] corresponding to [reactivity, deg_Mg_pH10, deg_Mg_50C]
    scored_indices = [0, 1, 3]

    # Set prediction to 1.0 and target to 0.0 for scored columns
    # Error = 1.0, Squared Error = 1.0, MSE = 1.0, RMSE = 1.0
    for idx in scored_indices:
        dummy_preds[:, :68, idx] = 1.0

    # Set prediction to 100.0 for UNSCORED columns (indices 2, 4)
    # These should NOT affect the metric
    unscored_indices = [2, 4]
    for idx in unscored_indices:
        dummy_preds[:, :68, idx] = 100.0

    score = calculate_mcrmse(dummy_preds, dummy_targets)

    # Expected: RMSE of 1.0 for scored cols, averaged. Result should be 1.0.
    assert np.isclose(
        score, 1.0
    ), f"Metric calculation failed. Expected 1.0, got {score}"
    print("Metric logic verified (correctly ignores unscored columns).")

    # 5. Training Loop Demonstration
    print("\nStarting training loop (1 Epoch)...")
    trainer = Trainer(model, train_loader, val_loader, test_loader)
    trainer.fit()

    # Verify model file creation
    assert os.path.exists(config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    print("Training complete and model saved.")

    # 6. Inference and Submission
    print("\nGenerating submission...")
    trainer.generate_submission()

    # Verify submission file
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {df_sub.shape}")

    # Check rows: 20 samples * 107 positions = 2140 rows
    expected_rows = SUBSET_SIZE * config.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Check columns
    expected_cols = ["id_seqpos"] + config.TARGET_COLS
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch."

    print("Submission verified.")
    print("\nDemo execution completed successfully!")


if __name__ == "__main__":
    run_demo()
