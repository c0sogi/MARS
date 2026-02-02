import sys
import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Add current directory to path to ensure library imports work correctly
sys.path.append(os.getcwd())

# Import library modules
from library.config import Config
from library.utils import seed_everything, MCRMSELoss
from library.data import preprocess_dataframe
from library.model import GCDARN
from library.train import run_training

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Initializing Demonstration...")

    # 1. Setup Environment and Config Overrides
    # We define a specific working directory for this demo to avoid clutter
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Setting up configuration in {demo_dir}...")

    # Override Config paths to point to the demo directory
    Config.WORK_DIR = demo_dir
    Config.CACHE_FILE = os.path.join(demo_dir, "debug_cache.npz")
    Config.MODEL_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "demo_submission.csv")

    # Optimize hyperparameters for a fast demonstration run
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 16  # Reasonable batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for this check

    # Set fixed seed for reproducibility
    seed_everything(Config.SEED)

    # 2. Verify Data Processing Logic
    print("\nVerifying Data Processing Logic...")

    # Create a dummy single-row DataFrame to test preprocessing
    # Sequence length must be 107
    dummy_seq = "AGCU" * 26 + "AGC"  # 104 + 3 = 107 chars
    # Structure: ".(..)" pattern repeated.
    # Index 0: ., Index 1: (, Index 2: ., Index 3: ., Index 4: )
    # This implies 1 pairs with 4.
    dummy_struct = ".(..)" * 21 + ".."
    dummy_loop = "EESSS" * 21 + "EE"
    # Targets: Stringified list of 68 floats
    dummy_react = str([0.1] * 68)

    dummy_data = {
        "id": "id_dummy",
        "sequence": dummy_seq,
        "structure": dummy_struct,
        "predicted_loop_type": dummy_loop,
        "reactivity": dummy_react,
        "deg_Mg_pH10": dummy_react,
        "deg_pH10": dummy_react,
        "deg_Mg_50C": dummy_react,
        "deg_50C": dummy_react,
    }
    df_dummy = pd.DataFrame([dummy_data])

    # Execute preprocessing
    feats, pidx, tgts, ids = preprocess_dataframe(df_dummy, is_test=False)

    # Assertions
    # Feature shape: (Num_Samples, Seq_Len, Feature_Dim) -> (1, 107, 18)
    assert feats.shape == (1, 107, 18), f"Feature shape mismatch: {feats.shape}"
    # Partner indices shape: (1, 107)
    assert pidx.shape == (1, 107), f"Partner indices shape mismatch: {pidx.shape}"
    # Targets shape: (1, 107, 5)
    assert tgts.shape == (1, 107, 5), f"Targets shape mismatch: {tgts.shape}"

    # Verify pairing logic: In ".(..)", index 1 is '(' and index 4 is ')'
    # They should be paired.
    assert (
        pidx[0, 1] == 4
    ), f"Partner logic failed: index 1 should pair with 4, got {pidx[0, 1]}"
    assert (
        pidx[0, 4] == 1
    ), f"Partner logic failed: index 4 should pair with 1, got {pidx[0, 4]}"

    print("Data Processing Verification Passed.")

    # 3. Verify Model Architecture
    print("\nVerifying Model Architecture...")
    device = torch.device("cpu")  # Use CPU for basic logic verification
    model = GCDARN().to(device)
    model.eval()

    # Convert dummy data to tensors
    dummy_feats_t = torch.tensor(feats, dtype=torch.float32).to(device)
    dummy_pidx_t = torch.tensor(pidx, dtype=torch.long).to(device)

    # Run forward pass
    with torch.no_grad():
        out1, out2 = model(dummy_feats_t, dummy_pidx_t)

    # Check output shapes: (Batch, Seq, 5)
    assert out1.shape == (1, 107, 5), f"Output 1 shape mismatch: {out1.shape}"
    assert out2.shape == (1, 107, 5), f"Output 2 shape mismatch: {out2.shape}"
    print("Model Architecture Verification Passed.")

    # 4. Verify Loss Function
    print("\nVerifying Loss Function...")
    criterion = MCRMSELoss()

    # Create synthetic predictions and targets
    # Scored columns in Config are indices: 0 (reactivity), 1 (deg_Mg_pH10), 3 (deg_Mg_50C)
    # We set targets to 1.0 for these columns and preds to 0.0
    # MSE = (0-1)^2 = 1. RMSE = 1. Mean RMSE = 1.

    t_preds = torch.zeros((1, 107, 5), dtype=torch.float32)
    t_targets = torch.zeros((1, 107, 5), dtype=torch.float32)

    scored_indices = [0, 1, 3]
    for idx in scored_indices:
        # Only the first 68 positions are scored
        t_targets[:, :68, idx] = 1.0

    loss_val = criterion(t_preds, t_targets)

    assert (
        abs(loss_val.item() - 1.0) < 1e-5
    ), f"Loss calculation incorrect. Expected 1.0, got {loss_val.item()}"
    print("Loss Function Verification Passed.")

    # 5. Run Training Pipeline (Integration Test)
    print("\nRunning Training Pipeline (1 Epoch)...")

    # This function will:
    # 1. Load data from ./metadata (using the logic in library/data.py)
    # 2. Process features and save to our demo cache
    # 3. Train the model for 1 epoch
    # 4. Run inference on test set
    # 5. Generate submission file
    try:
        run_training(load_cached_data=False, num_epochs=Config.EPOCHS)
    except Exception as e:
        print(f"Training pipeline failed: {e}")
        raise e

    # 6. Verify Submission Output
    print("\nVerifying Submission Output...")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")

    # Expected rows: 240 test samples * 107 sequence length = 25680
    expected_rows = 240 * 107
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Expected columns: id_seqpos + 5 targets
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(sub_df.columns)}"

    # Check for NaNs
    assert not sub_df.isnull().values.any(), "Submission contains NaN values."

    print("Submission Verification Passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
