import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.model import RNAModel
from library.data import get_dataloaders
from library.utils import set_seed, mcrmse
from library.train import train_one_epoch, validate, generate_submission


def run_demo():
    print(">>> Starting Demo Execution")

    # =========================================================================
    # 1. Configuration Overrides for Demo
    # =========================================================================
    print(">>> Configuring environment...")

    # Modify Config to use a demo-specific working directory
    # This prevents conflicts with the main training artifacts
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config attributes
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_FILE = os.path.join(DEMO_DIR, "submission.csv")
    Config.NUM_EPOCHS = 1  # Run only 1 epoch for speed
    Config.BATCH_SIZE = 32  # Reasonable batch size
    Config.HIDDEN_DIM = 64  # Smaller model for faster demo execution
    Config.EMBED_DIM_SEQ = 32
    Config.EMBED_DIM_LOOP = 16
    Config.EMBED_DIM_DIST = 16
    Config.N_LAYERS = 2  # Shallower network

    # Set seed for reproducibility
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================================================================
    # 2. Verify Metric Logic (MCRMSE)
    # =========================================================================
    print(">>> Verifying MCRMSE metric...")

    # Case 1: Perfect prediction
    y_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    score = mcrmse(y_true, y_pred)
    assert torch.isclose(score, torch.tensor(0.0)), f"Expected 0.0, got {score}"

    # Case 2: Known error
    # Col 1: |1-0|=1, |3-2|=1 -> MSE=1 -> RMSE=1
    # Col 2: |2-0|=2, |4-2|=2 -> MSE=4 -> RMSE=2
    # Avg RMSE = (1 + 2) / 2 = 1.5
    y_true_err = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_pred_err = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    score_err = mcrmse(y_true_err, y_pred_err)
    assert torch.isclose(score_err, torch.tensor(1.5)), f"Expected 1.5, got {score_err}"

    print("MCRMSE check passed.")

    # =========================================================================
    # 3. Verify Model Architecture
    # =========================================================================
    print(">>> Verifying Model Forward Pass...")

    model = RNAModel(config=Config).to(device)

    # Create dummy batch
    dummy_batch_size = 4
    seq_len = Config.SEQ_LEN

    dummy_seq = torch.randint(0, 4, (dummy_batch_size, seq_len)).to(device)
    dummy_loop = torch.randint(0, 7, (dummy_batch_size, seq_len)).to(device)
    dummy_dist = torch.randint(-10, 10, (dummy_batch_size, seq_len)).to(device)

    with torch.no_grad():
        output = model(dummy_seq, dummy_loop, dummy_dist)

    # Check output shape: (Batch, Seq_Len, N_Outputs)
    expected_shape = (dummy_batch_size, seq_len, Config.N_OUTPUTS)
    assert (
        output.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {output.shape}"

    print(f"Model forward pass successful. Output shape: {output.shape}")

    # =========================================================================
    # 4. Data Loading
    # =========================================================================
    print(">>> Loading DataLoaders...")
    # This will trigger processing of parquet files and caching to DEMO_DIR
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, batch_size=Config.BATCH_SIZE
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # =========================================================================
    # 5. Training Loop Execution
    # =========================================================================
    print(">>> Running Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)
    criterion = torch.nn.MSELoss()

    # Train
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device, Config.CLIP_GRAD
    )
    print(f"Epoch 1 Train Loss: {train_loss:.6f}")

    # Validate
    val_score = validate(model, val_loader, device)
    print(f"Epoch 1 Val MCRMSE: {val_score:.6f}")

    # =========================================================================
    # 6. Submission Generation
    # =========================================================================
    print(">>> Generating Submission...")

    generate_submission(model, test_loader, device, Config.SUBMISSION_FILE)

    # Verify file existence
    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError("Submission file was not created.")

    # Verify file content format
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission shape: {df_sub.shape}")

    # Check columns
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
    ), f"Columns mismatch. Got {list(df_sub.columns)}"

    # Check row count
    # Test set has 240 samples, each length 107 -> 240 * 107 = 25680 rows
    # Note: The provided sample_submission.csv has 25680 rows.
    # Let's check against the test loader size or hardcoded expectation.
    # The metadata test.parquet has 240 rows.
    expected_rows = 240 * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Check values in unscored columns (should be 0.0 as per generate_submission logic)
    assert (df_sub["deg_pH10"] == 0.0).all(), "deg_pH10 should be 0.0"
    assert (df_sub["deg_50C"] == 0.0).all(), "deg_50C should be 0.0"

    print(">>> Demo Execution Completed Successfully.")


if __name__ == "__main__":
    run_demo()
