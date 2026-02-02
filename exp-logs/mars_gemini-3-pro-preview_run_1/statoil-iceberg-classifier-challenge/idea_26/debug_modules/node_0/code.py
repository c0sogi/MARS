import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import process_json, IcebergDataset, get_transforms
from library.model import IcebergResNet18
from library.calibration import run_calibration_phase
from library.production import run_production_phase

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Ship vs Iceberg Demo Execution ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup & Overrides
    # -------------------------------------------------------------------------
    print("[1/6] Configuring environment...")

    # Redirect working directory to a demo folder
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    Config.setup()

    # Override hyperparameters for speed
    Config.P1_MAX_EPOCHS = 1  # Run only 1 epoch for calibration
    Config.P1_PATIENCE = 1  # minimal patience
    Config.SWA_EPOCHS = 1  # Run only 1 SWA epoch
    Config.N_FOLDS = 2  # Use only 2 folds instead of 5
    Config.BATCH_SIZE = 8  # Small batch size for the small subset
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print("    Configuration overrides applied for fast execution.")

    # -------------------------------------------------------------------------
    # 2. Data Subsetting (Optimization Strategy)
    # -------------------------------------------------------------------------
    print("[2/6] Preparing data subset for demonstration...")

    # We load the full data once, slice it, and save it to the cache.
    # Subsequent calls in the library will pick up this cached subset.

    # Process Train Data
    full_train = process_json(
        Config.TRAIN_JSON, "temp_train.npz", load_cached_data=False
    )
    subset_size = 50  # Use 50 samples for demo

    train_subset = {
        "ids": full_train["ids"][:subset_size],
        "band_1": full_train["band_1"][:subset_size],
        "band_2": full_train["band_2"][:subset_size],
        "inc_angles": full_train["inc_angles"][:subset_size],
        "labels": full_train["labels"][:subset_size],
    }

    # Save to the actual cache location expected by library
    np.savez(os.path.join(Config.CACHE_DIR, "train_processed.npz"), **train_subset)

    # Process Test Data
    full_test = process_json(Config.TEST_JSON, "temp_test.npz", load_cached_data=False)
    test_subset = {
        "ids": full_test["ids"][:subset_size],
        "band_1": full_test["band_1"][:subset_size],
        "band_2": full_test["band_2"][:subset_size],
        "inc_angles": full_test["inc_angles"][:subset_size],
    }
    np.savez(os.path.join(Config.CACHE_DIR, "test_processed.npz"), **test_subset)

    # Clean up temp files
    if os.path.exists(os.path.join(Config.CACHE_DIR, "temp_train.npz")):
        os.remove(os.path.join(Config.CACHE_DIR, "temp_train.npz"))
    if os.path.exists(os.path.join(Config.CACHE_DIR, "temp_test.npz")):
        os.remove(os.path.join(Config.CACHE_DIR, "temp_test.npz"))

    print(f"    Cached subset of {subset_size} samples created.")

    # -------------------------------------------------------------------------
    # 3. Verify Data Loading Logic
    # -------------------------------------------------------------------------
    print("[3/6] Verifying Data Loading logic...")

    # Load back using the library function to ensure it reads cache correctly
    data_loaded = process_json(
        Config.TRAIN_JSON, "train_processed.npz", load_cached_data=True
    )

    # Assertions
    assert len(data_loaded["ids"]) == subset_size, "Subset size mismatch"
    assert data_loaded["band_1"].shape == (subset_size, 75, 75), "Band 1 shape mismatch"
    assert data_loaded["band_2"].shape == (subset_size, 75, 75), "Band 2 shape mismatch"
    assert "labels" in data_loaded, "Labels missing in train data"

    # Verify Dataset Class
    ds = IcebergDataset(
        data_loaded, np.arange(subset_size), transform=get_transforms("train")
    )
    sample_img, sample_angle, sample_label = ds[0]

    assert sample_img.shape == (
        3,
        224,
        224,
    ), f"Unexpected image shape: {sample_img.shape}"
    assert isinstance(sample_angle, torch.Tensor), "Angle should be a tensor"
    assert isinstance(sample_label, torch.Tensor), "Label should be a tensor"
    print("    Data loading and processing verified.")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("[4/6] Verifying Model Architecture...")

    model = IcebergResNet18().to(Config.DEVICE)
    model.eval()

    # Create dummy input batch
    dummy_img = torch.randn(4, 3, 224, 224).to(Config.DEVICE)
    dummy_angle = torch.tensor([35.0, 40.0, 30.0, 45.0]).to(Config.DEVICE)

    with torch.no_grad():
        output = model(dummy_img, dummy_angle)

    assert output.shape == (4, 1), f"Model output shape mismatch: {output.shape}"
    print("    Model forward pass verified.")

    # Free memory
    del model, dummy_img, dummy_angle
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 5. Execute Pipeline: Phase 1 (Calibration)
    # -------------------------------------------------------------------------
    print("[5/6] Executing Phase 1: Calibration...")

    # This will run 2 folds (Config.N_FOLDS) for 1 epoch (Config.P1_MAX_EPOCHS)
    # on the 50-sample subset.
    e_conv = run_calibration_phase(load_cached_data=True)

    print(f"    Calibration complete. Optimal Epochs (e_conv): {e_conv}")

    # Sanity check
    assert isinstance(e_conv, int) and e_conv > 0, "Invalid e_conv returned"

    # -------------------------------------------------------------------------
    # 6. Execute Pipeline: Phase 2 (Production)
    # -------------------------------------------------------------------------
    print("[6/6] Executing Phase 2: Production...")

    # This will train 5 ensemble models for e_conv epochs + 1 SWA epoch
    # on the 50-sample subset.
    run_production_phase(e_conv=e_conv, load_cached_data=True)

    # Validate Submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not generated.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission generated at: {Config.SUBMISSION_PATH}")
    print("    First 5 rows:")
    print(df_sub.head())

    # Validate format
    assert list(df_sub.columns) == ["id", "is_iceberg"], "Incorrect submission columns"
    assert (
        len(df_sub) == subset_size
    ), f"Submission length {len(df_sub)} != test subset size {subset_size}"
    assert (
        df_sub["is_iceberg"].min() >= 0 and df_sub["is_iceberg"].max() <= 1
    ), "Probabilities out of range"

    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
