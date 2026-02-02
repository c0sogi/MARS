import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood
from library.data import get_data
from library.engine import Engine
from library.model import VaryingCoeffNet


def run_demo():
    # 1. Setup and Configuration Override for Speed
    print("--- 1. Setup and Configuration ---")
    seed_everything(42)

    # Override Config defaults to ensure the demo runs quickly (within minutes)
    # We use a very small subset of data and minimal epochs.
    Config.DEBUG = True
    Config.DEBUG_SIZE = 10  # Use only 10 patients/samples
    Config.EPOCHS = 2  # Run only 2 epochs per phase
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    print(f"Configured for speed: DEBUG={Config.DEBUG}, EPOCHS={Config.EPOCHS}")
    print(f"Device: {Config.DEVICE}")

    # 2. Data Loading and Processing
    print("\n--- 2. Data Loading and Processing ---")
    # We force load_cached_data=False to demonstrate the raw data processing pipeline
    # (reading DICOMs, extracting features, processing tabular data).
    train_ds, val_ds, test_ds = get_data(load_cached_data=False)

    print(f"Train Dataset Size: {len(train_ds)}")
    print(f"Val Dataset Size: {len(val_ds)}")
    print(f"Test Dataset Size: {len(test_ds)}")

    # Verify Data Shapes (Image embedding dim + Tabular dim + Week + Target/ID)
    # Image features should be (12, 1280) from EfficientNet-B0
    sample_img, sample_tab, sample_week, sample_target = train_ds[0]
    assert sample_img.shape == (
        12,
        1280,
    ), f"Unexpected image feature shape: {sample_img.shape}"
    assert sample_tab.shape == (
        8,
    ), f"Unexpected tabular feature shape: {sample_tab.shape}"  # 3 cont + 2 sex + 3 smoke

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # 3. Model Training (Two-Phase Strategy)
    print("\n--- 3. Model Training ---")
    engine = Engine(device=Config.DEVICE)

    # run_training handles Phase 1 (FVC Trajectory) and Phase 2 (Uncertainty)
    trained_model = engine.run_training(train_loader, val_loader)

    # Verify model returns expected shapes
    # Create a dummy batch
    dummy_img = torch.randn(2, 12, 1280).to(Config.DEVICE)
    dummy_tab = torch.randn(2, 8).to(Config.DEVICE)
    dummy_weeks = torch.tensor([[10.0], [20.0]]).to(Config.DEVICE)

    trained_model.eval()
    with torch.no_grad():
        fvc_pred, delta_pred = trained_model(dummy_img, dummy_tab, dummy_weeks)

    assert fvc_pred.shape == (
        2,
        1,
    ), f"Expected FVC pred shape (2, 1), got {fvc_pred.shape}"
    assert delta_pred.shape == (
        2,
        1,
    ), f"Expected Delta pred shape (2, 1), got {delta_pred.shape}"
    print("Model inference shape verification passed.")

    # 4. Submission Generation
    print("\n--- 4. Submission Generation ---")
    engine.generate_submission(trained_model, test_loader)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission file loaded successfully.")
    print(sub_df.head())

    # Check columns
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert list(sub_df.columns) == expected_cols, f"Invalid columns: {sub_df.columns}"

    # Check values
    assert not sub_df.isnull().values.any(), "Submission contains NaNs"
    assert len(sub_df) == len(
        test_ds
    ), f"Submission row count mismatch. Expected {len(test_ds)}, got {len(sub_df)}"

    # 5. Metric Logic Verification
    print("\n--- 5. Metric Verification ---")
    # Test the Laplace Log Likelihood function with known values
    # Case 1: Perfect prediction
    # Delta = 0, Sigma = 70 (clipped from anything < 70)
    # Metric = - (sqrt(2) * 0) / 70 - ln(sqrt(2) * 70)
    #        = - ln(98.99) approx -4.595
    val_perfect = laplace_log_likelihood([2000], [2000], [50])
    expected_perfect = -np.log(np.sqrt(2) * 70)
    assert np.isclose(
        val_perfect, expected_perfect, atol=1e-4
    ), f"Metric calculation failed for perfect case. Got {val_perfect}, expected {expected_perfect}"

    # Case 2: Large Error
    # Delta = 1000 (clipped), Sigma = 100
    # Metric = - (sqrt(2) * 1000) / 100 - ln(sqrt(2) * 100)
    #        = - 14.142 - 4.95 = -19.09
    val_bad = laplace_log_likelihood([3000], [1000], [100])
    expected_bad = -(np.sqrt(2) * 1000) / 100 - np.log(np.sqrt(2) * 100)
    assert np.isclose(
        val_bad, expected_bad, atol=1e-4
    ), f"Metric calculation failed for bad case. Got {val_bad}, expected {expected_bad}"

    print("Metric verification passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
