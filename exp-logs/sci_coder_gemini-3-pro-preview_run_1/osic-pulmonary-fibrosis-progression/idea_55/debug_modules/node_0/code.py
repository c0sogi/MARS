import sys
import os
import warnings
import torch
import pandas as pd
import numpy as np

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Ensure library can be imported from current directory
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, metric_score
from library.data import get_dataloaders
from library.model import NSLHN
from library.train import (
    LaplaceLogLikelihoodLoss,
    train_epoch,
    valid_epoch,
    generate_submission,
)


def prepare_debug_metadata(sample_size=6):
    """
    Creates a consistent subset of metadata files in the working directory.
    This ensures that the dataloaders and the generate_submission function
    operate on the same number of samples, preventing length mismatch errors
    during the demo execution.
    """
    debug_meta_dir = os.path.join(Config.WORKING_DIR, "debug_metadata")
    os.makedirs(debug_meta_dir, exist_ok=True)

    # Map original config paths to new debug filenames
    files = {
        "train": (Config.TRAIN_CSV, "train.csv"),
        "val": (Config.VAL_CSV, "val.csv"),
        "test": (Config.TEST_CSV, "test.csv"),
    }

    new_paths = {}

    for key, (src_path, filename) in files.items():
        # Read original metadata
        df = pd.read_csv(src_path)
        # Create a small subset
        df_subset = df.head(sample_size).copy()
        # Save to working directory
        dst_path = os.path.join(debug_meta_dir, filename)
        df_subset.to_csv(dst_path, index=False)
        new_paths[key] = dst_path

    return new_paths


def run_demo():
    print("=== Starting NSL-HN Demonstration ===\n")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("[1] Configuring Environment...")

    # Override Config for fast demonstration
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Disable multiprocessing for stability in demo

    # Prepare Debug Data
    print("    Creating debug metadata subsets (N=6)...")
    new_paths = prepare_debug_metadata(sample_size=6)

    # Update Config to point to debug metadata
    Config.TRAIN_CSV = new_paths["train"]
    Config.VAL_CSV = new_paths["val"]
    Config.TEST_CSV = new_paths["test"]

    # Initialize Environment (Creates directories, sets seeds)
    Config.setup()
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print("    Configuration Complete.")

    # ---------------------------------------------------------
    # 2. Data Loading Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Data Loading...")
    # We use debug=False here because we manually created the debug files
    # and updated the Config paths above.
    train_loader, val_loader, test_loader = get_dataloaders(debug=False)

    # Fetch a single batch to verify structure
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError(
            "DataLoader returned no batches. Check dataset size vs batch size."
        )

    # Verify Batch Keys
    required_keys = [
        "patient_id",
        "axial",
        "coronal",
        "tabular",
        "delta_week",
        "baseline_fvc",
        "target",
    ]
    for k in required_keys:
        assert k in batch, f"Missing key '{k}' in batch"

    # Verify Tensor Shapes
    # Axial/Coronal: (Batch, 3, 224, 224)
    assert batch["axial"].shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), f"Axial shape incorrect: {batch['axial'].shape}"
    assert batch["coronal"].shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), f"Coronal shape incorrect: {batch['coronal'].shape}"
    # Tabular: (Batch, 6)
    assert batch["tabular"].shape == (
        Config.BATCH_SIZE,
        6,
    ), f"Tabular shape incorrect: {batch['tabular'].shape}"
    # Scalars: (Batch,)
    assert batch["target"].shape == (Config.BATCH_SIZE,)

    print(f"    Batch Loaded Successfully. Image Shape: {batch['axial'].shape}")

    # ---------------------------------------------------------
    # 3. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")
    model = NSLHN().to(device)

    # Prepare inputs
    axial = batch["axial"].to(device)
    coronal = batch["coronal"].to(device)
    tabular = batch["tabular"].to(device)

    # Forward Pass
    alpha, sigma_base, sigma_growth = model(axial, coronal, tabular)

    # Verify Output Shapes
    assert alpha.shape == (Config.BATCH_SIZE,)
    assert sigma_base.shape == (Config.BATCH_SIZE,)
    assert sigma_growth.shape == (Config.BATCH_SIZE,)

    # Verify Constraints (Sigma must be positive via Softplus)
    assert (sigma_base > 0).all().item(), "Sigma Base must be positive"
    assert (sigma_growth > 0).all().item(), "Sigma Growth must be positive"

    print("    Forward Pass Successful.")
    print(
        f"    Sample Output - Alpha: {alpha[0].item():.4f}, SigmaBase: {sigma_base[0].item():.4f}"
    )

    # ---------------------------------------------------------
    # 4. Loss Function Verification
    # ---------------------------------------------------------
    print("\n[4] Verifying Loss Function...")
    criterion = LaplaceLogLikelihoodLoss()

    target = batch["target"].to(device)
    delta_week = batch["delta_week"].to(device)
    baseline_fvc = batch["baseline_fvc"].to(device)

    loss = criterion(alpha, sigma_base, sigma_growth, target, delta_week, baseline_fvc)

    # Loss should be a scalar
    assert loss.dim() == 0, "Loss must be a scalar"
    assert not torch.isnan(loss).item(), "Loss contains NaN values"

    print(f"    Loss Calculated: {loss.item():.6f}")

    # ---------------------------------------------------------
    # 5. Training Loop Verification
    # ---------------------------------------------------------
    print("\n[5] Verifying Training & Validation Loop...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)

    # Execute one training epoch
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    print(f"    Train Loss: {train_loss:.6f}")

    # Execute one validation epoch
    val_loss, val_score = valid_epoch(model, val_loader, criterion, device)
    print(f"    Val Loss: {val_loss:.6f} | Val Metric: {val_score:.6f}")

    assert isinstance(train_loss, float)
    assert isinstance(val_score, float)
    print("    Training Pipeline Verified.")

    # ---------------------------------------------------------
    # 6. Submission Generation Verification
    # ---------------------------------------------------------
    print("\n[6] Verifying Submission Generation...")
    # This function generates predictions for the test set and saves to file
    generate_submission(model, test_loader, device)

    # Verify file existence and content
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created"

    sub_df = pd.read_csv(Config.SUBMISSION_FILE)

    # Check dimensions (should match our debug sample size of 6)
    assert (
        len(sub_df) == 6
    ), f"Submission length mismatch. Expected 6, got {len(sub_df)}"
    # Check columns
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch: {sub_df.columns}"

    print(f"    Submission saved to {Config.SUBMISSION_FILE}")
    print("    Submission Verification Complete.")

    # ---------------------------------------------------------
    # 7. Metric Logic Verification
    # ---------------------------------------------------------
    print("\n[7] Verifying Metric Logic...")
    # Manual calculation check
    # Metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    # Case 1: Perfect prediction, Sigma=70 (Minimum clipped)
    # Delta = 0, Sigma_Clipped = 70
    # Expected = 0 - ln(sqrt(2) * 70)
    score_perfect = metric_score(np.array([2000]), np.array([2000]), np.array([70]))
    expected_perfect = -np.log(np.sqrt(2) * 70)
    assert np.isclose(score_perfect, expected_perfect, atol=1e-5)

    # Case 2: Large Error (clipped at 1000), Low Sigma (clipped at 70)
    # Delta = 1000, Sigma_Clipped = 70
    score_bad = metric_score(np.array([3000]), np.array([2000]), np.array([10]))
    term1 = -(np.sqrt(2) * 1000) / 70
    term2 = -np.log(np.sqrt(2) * 70)
    expected_bad = term1 + term2
    assert np.isclose(score_bad, expected_bad, atol=1e-5)

    print("    Metric Logic Verified.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
