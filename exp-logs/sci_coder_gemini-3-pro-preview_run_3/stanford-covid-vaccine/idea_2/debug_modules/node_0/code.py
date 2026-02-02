import os
import shutil
import torch
import pandas as pd
import numpy as np

# Import components from the provided library files
from library.config import Config
from library.utils import set_seed, format_submission
from library.data import get_dataloaders
from library.model import HybridGNN
from library.train import run_training


def main():
    print("=== RNA Degradation Prediction Pipeline Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Override
    # -------------------------------------------------------------------------
    # We modify the Config class attributes directly to optimize for a fast demo run.
    # This ensures we don't consume too much time or resources during this check.
    print(">>> Configuring environment for fast demonstration...")

    Config.NUM_EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 16  # Small batch size
    Config.GNN_HIDDEN_DIM = 32  # Reduced model size for speed
    Config.LSTM_HIDDEN_DIM = 32
    Config.WORKING_DIR = "./working/demo_run"  # Isolate demo files
    Config.CACHE_DIR = Config.WORKING_DIR
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Clean up any previous demo run to ensure a fresh start
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(42)
    print(f"Working directory set to: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loading and Verification
    # -------------------------------------------------------------------------
    print("\n>>> Initializing DataLoaders...")

    # We load data. The first run will process the Parquet files into Graph objects.
    # Subsequent calls (like in run_training) will use the cache generated here.
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=0,  # Use 0 workers to avoid multiprocessing overhead in demo
        load_cached_data=False,
    )

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))
    print(f"Batch loaded. Graphs in batch: {batch.num_graphs}")

    # Verify Node Features
    # Expected: (Total Nodes, 14 features)
    # 14 = 4 (Seq) + 7 (Loop) + 3 (Struct)
    assert batch.x.size(1) == 14, f"Expected 14 node features, got {batch.x.size(1)}"

    # Verify Targets
    # PyG stacks targets. Expected rows = Batch_Size * Seq_Length (107)
    expected_rows = batch.num_graphs * Config.SEQ_LENGTH
    assert (
        batch.y.size(0) == expected_rows
    ), f"Expected {expected_rows} target rows, got {batch.y.size(0)}"
    assert batch.y.size(1) == 5, f"Expected 5 target columns, got {batch.y.size(1)}"

    print("Data integrity check passed.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = HybridGNN().to(device)
    batch = batch.to(device)

    # Perform a forward pass
    with torch.no_grad():
        preds = model(batch)

    # Expected Output: (Batch_Size, Seq_Length, Num_Targets)
    print(f"Prediction shape: {preds.shape}")

    assert preds.size(0) == batch.num_graphs
    assert preds.size(1) == Config.SEQ_LENGTH
    assert preds.size(2) == Config.NUM_TARGETS

    print("Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n>>> Executing Training Loop...")

    # run_training encapsulates the Trainer initialization and fitting loop
    best_val_loss = run_training(
        num_epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        patience=1,  # Strict early stopping for demo
        load_cached_data=True,  # Use the cache we just created
    )

    print(f"Training finished. Best Validation MCRMSE: {best_val_loss:.4f}")

    # Verify model checkpoint exists
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved!"

    # -------------------------------------------------------------------------
    # 5. Inference and Submission Generation
    # -------------------------------------------------------------------------
    print("\n>>> Generating Submission...")

    # Load the best saved model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    all_ids = []
    all_preds = []

    # Inference on Test Set
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch)  # Shape: (B, 107, 5)

            all_preds.append(preds.cpu())
            all_ids.extend(batch.id)

    full_preds = torch.cat(all_preds, dim=0)

    # Verify inference counts
    n_test_samples = len(test_loader.dataset)
    assert len(all_ids) == n_test_samples
    assert full_preds.size(0) == n_test_samples

    # Format using the utility function
    sub_df = format_submission(all_ids, full_preds, save_path=Config.SUBMISSION_PATH)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    # Verify DataFrame structure
    # Total rows should be Num_Samples * Seq_Length (107)
    expected_sub_rows = n_test_samples * Config.SEQ_LENGTH
    assert (
        len(sub_df) == expected_sub_rows
    ), f"Submission has {len(sub_df)} rows, expected {expected_sub_rows}"

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("Head of submission:")
    print(sub_df.head())

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
