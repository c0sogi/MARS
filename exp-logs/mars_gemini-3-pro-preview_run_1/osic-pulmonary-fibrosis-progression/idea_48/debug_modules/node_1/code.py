import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihoodLoss, calculate_metric
from library.data import get_dataloaders
import library.data  # Imported to check for pydicom availability
from library.model import SLHDAN
from library.train import run_training, build_baseline_lookup, get_baseline_tensors

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== SLH-DAN Pipeline Demonstration ===")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[1] Configuring Environment...")

    # Override Config defaults for a fast demonstration run
    Config.DEBUG_SUBSET_SIZE = 20  # Use only 20 samples for training/val
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple debug

    # Set a custom cache directory for this demo to avoid conflicts
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    Config.setup()  # Create directories

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")

    # Monkeypatch pydicom if missing (allows demo to run without raw DICOM processing)
    if library.data.pydicom is None:
        print(
            "    ! pydicom not found. Monkeypatching read_dicom_volume for demonstration."
        )

        def dummy_read_dicom_volume(path):
            # Return a random 3D volume (Depth, Height, Width)
            # Simulates a normalized CT scan
            return np.random.rand(20, 512, 512).astype(np.float32)

        library.data.read_dicom_volume = dummy_read_dicom_volume
    else:
        print("    pydicom is available. Using real DICOM processing.")

    # ---------------------------------------------------------
    # 2. Data Loading Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    print(f"    Train Batches: {len(train_loader)}")

    # Fetch a single batch to inspect
    batch = next(iter(train_loader))

    # Check required keys
    required_keys = [
        "image_ax",
        "image_cor",
        "tabular",
        "target",
        "weeks",
        "patient_id",
    ]
    for k in required_keys:
        assert k in batch, f"Batch missing key: {k}"

    # Inspect shapes
    img_ax = batch["image_ax"]
    img_cor = batch["image_cor"]
    tabular = batch["tabular"]
    targets = batch["target"]

    print(
        f"    Shapes -> Axial: {img_ax.shape}, Tabular: {tabular.shape}, Target: {targets.shape}"
    )

    # Assertions for correctness
    assert img_ax.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), "Incorrect Axial Image Shape"
    assert img_cor.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), "Incorrect Coronal Image Shape"
    assert tabular.shape == (Config.BATCH_SIZE, 4), "Incorrect Tabular Shape"

    # ---------------------------------------------------------
    # 3. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Architecture (SLH-DAN)...")

    model = SLHDAN().to(device)

    # Move inputs to device
    img_ax = img_ax.to(device)
    img_cor = img_cor.to(device)
    tabular = tabular.to(device)

    # Test A: Static Parameter Prediction (No temporal info provided)
    # Output should be (B, 3) -> [alpha, sigma_base, sigma_growth]
    static_out = model(img_ax, img_cor, tabular)
    print(f"    Static Output Shape: {static_out.shape}")
    assert static_out.shape == (Config.BATCH_SIZE, 3), "Static output should be (B, 3)"

    # Test B: Trajectory Prediction (With temporal anchors)
    # We need to simulate the baseline lookup performed during training
    train_lookup = build_baseline_lookup(train_loader.dataset)
    patient_ids = batch["patient_id"]

    # Retrieve baseline tensors
    base_fvc, base_week = get_baseline_tensors(patient_ids, train_lookup, device)
    current_weeks = batch["weeks"].to(device)

    # Output should be (B, 2) -> [Pred_FVC, Pred_Sigma]
    traj_out = model(
        img_ax,
        img_cor,
        tabular,
        base_fvc=base_fvc,
        base_week=base_week,
        current_week=current_weeks,
    )
    print(f"    Trajectory Output Shape: {traj_out.shape}")
    assert traj_out.shape == (
        Config.BATCH_SIZE,
        2,
    ), "Trajectory output should be (B, 2)"

    # ---------------------------------------------------------
    # 4. Loss Function Verification
    # ---------------------------------------------------------
    print("\n[4] Verifying Metric & Loss...")

    criterion = LaplaceLogLikelihoodLoss().to(device)
    targets = targets.to(device)

    # Calculate Loss
    loss = criterion(traj_out, targets)
    print(f"    Loss Value: {loss.item():.4f}")
    assert torch.isfinite(loss), "Loss is not finite"

    # Calculate Metric (Should be roughly negative loss)
    metric = calculate_metric(traj_out, targets)
    print(f"    Metric Value: {metric:.4f}")

    # ---------------------------------------------------------
    # 5. Full Pipeline Execution
    # ---------------------------------------------------------
    print("\n[5] Executing Full Training Loop (Short Run)...")

    # run_training() handles the entire loop: Data -> Train -> Val -> Inference -> Save
    # It uses the Config settings we modified at the start.
    run_training()

    # ---------------------------------------------------------
    # 6. Submission Verification
    # ---------------------------------------------------------
    print("\n[6] Verifying Submission File...")

    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"    Submission saved to: {Config.SUBMISSION_PATH}")
        print(f"    Rows: {len(sub_df)}")
        print(sub_df.head(3))

        expected_cols = ["Patient_Week", "FVC", "Confidence"]
        assert all(
            c in sub_df.columns for c in expected_cols
        ), "Missing columns in submission"
        assert len(sub_df) > 0, "Submission file is empty"
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
