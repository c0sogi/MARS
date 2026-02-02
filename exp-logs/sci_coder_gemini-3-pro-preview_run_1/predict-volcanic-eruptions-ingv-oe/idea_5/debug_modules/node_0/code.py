import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.data_processing import DataManager
from library.models import VolcanoEfficientNet
from library.dataset import VolcanoSpectrogramDataset
from library.train_tabular import run_tabular_training
from library.train_vision import run_vision_training
from library.stacking import train_meta_learner


def main():
    print("=== Volcano Eruption Prediction Pipeline Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Fast Demonstration
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Enable Debug mode to use small data subsets (20 samples per split)
    Config.DEBUG = True

    # Reduce Cross-Validation folds to 2 for speed
    Config.N_FOLDS = 2

    # Optimize LightGBM for tiny dataset (prevent errors with small sample size)
    Config.LGBM_PARAMS["n_estimators"] = 20
    Config.LGBM_PARAMS["min_child_samples"] = 2
    Config.LGBM_PARAMS["num_leaves"] = 5
    Config.LGBM_PARAMS["learning_rate"] = 0.1

    # Optimize CNN for speed (1 epoch, small batch)
    Config.CNN_PARAMS["epochs"] = 1
    Config.CNN_PARAMS["batch_size"] = 4

    # Disable multiprocessing for data loading to avoid overhead in this small demo
    Config.NUM_WORKERS = 0

    # Set a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print("Configuration updated.")

    # -------------------------------------------------------------------------
    # 2. Data Processing & Loading Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying DataManager and Data Loading...")

    dm = DataManager()

    # Load training data in debug mode
    # This should load 20 samples from train.csv (as per DataManager debug logic)
    print("Loading debug train data...")
    # We force load_cached_data=False to ensure the processing logic runs
    X_tab, X_spec, y = dm.get_data("train", load_cached_data=False, debug=True)

    # Verify Tabular Data
    print(f"Tabular Data Shape: {X_tab.shape}")
    assert isinstance(X_tab, pd.DataFrame)
    assert len(X_tab) == 20, "Debug mode should load exactly 20 samples"
    assert "sensor_1_mean" in X_tab.columns, "Feature extraction failed"

    # Verify Spectrogram Data
    print(f"Spectrogram Data Shape: {X_spec.shape}")
    assert isinstance(X_spec, np.ndarray)
    assert X_spec.shape[0] == 20
    assert X_spec.shape[1] == 10, "Should have 10 channels for 10 sensors"
    # Spectrogram shape is (N, 10, Freq, Time)

    # Verify Targets
    print(f"Target Data Shape: {y.shape}")
    assert len(y) == 20

    print("Data loading verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Vision Model Architecture...")

    # Instantiate model (pretrained=False to avoid download attempts during this check)
    model = VolcanoEfficientNet(pretrained=False)
    model.eval()

    # Create dummy input based on loaded spectrogram shape
    # Shape: (Batch, Channels, Freq, Time)
    freq_bins = X_spec.shape[2]
    time_steps = X_spec.shape[3]
    dummy_batch_size = 2
    dummy_input = torch.randn(dummy_batch_size, 10, freq_bins, time_steps)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")

    assert output.shape == (
        dummy_batch_size,
        1,
    ), "Model output shape should be (Batch, 1)"
    print("Model architecture verification passed.")

    # -------------------------------------------------------------------------
    # 4. Dataset Class Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying VolcanoSpectrogramDataset...")

    dataset = VolcanoSpectrogramDataset(X_spec, y, log_target=True)
    spec_tensor, target_tensor = dataset[0]

    print(f"Sample Tensor Shape: {spec_tensor.shape}")
    print(f"Target Tensor Value: {target_tensor.item()}")

    assert spec_tensor.shape == (10, freq_bins, time_steps)
    assert target_tensor.shape == (1,)
    # Check log transform: log1p(y)
    expected = np.log1p(y[0])
    assert np.isclose(target_tensor.item(), expected, atol=1e-5)

    print("Dataset verification passed.")

    # -------------------------------------------------------------------------
    # 5. Full Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n[5] Executing Full Pipeline (Tabular + Vision + Stacking)...")

    # A. Tabular Branch
    print("\n--- Running Tabular Training ---")
    # debug=True loads 20 train and 20 val samples -> Total 40 samples for CV
    df_oof_tab, df_test_tab = run_tabular_training(debug=True)

    # Expected OOF size: 20 (train debug) + 20 (val debug) = 40
    assert (
        len(df_oof_tab) == 40
    ), f"Tabular OOF size mismatch. Expected 40, got {len(df_oof_tab)}"
    assert (
        len(df_test_tab) == 20
    ), f"Tabular Test size mismatch. Expected 20, got {len(df_test_tab)}"

    # B. Vision Branch
    print("\n--- Running Vision Training ---")
    df_oof_vis, df_test_vis = run_vision_training(debug=True)

    assert (
        len(df_oof_vis) == 40
    ), f"Vision OOF size mismatch. Expected 40, got {len(df_oof_vis)}"
    assert (
        len(df_test_vis) == 20
    ), f"Vision Test size mismatch. Expected 20, got {len(df_test_vis)}"

    # C. Meta-Learner (Stacking)
    print("\n--- Running Meta-Learner ---")
    train_meta_learner(df_oof_tab, df_test_tab, df_oof_vis, df_test_vis)

    print("Pipeline execution completed.")

    # -------------------------------------------------------------------------
    # 6. Submission Verification
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Submission File...")

    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission File Found: {Config.SUBMISSION_PATH}")
        print(f"Submission Shape: {df_sub.shape}")
        print("First 5 rows:")
        print(df_sub.head())

        # Verify format
        assert list(df_sub.columns) == ["segment_id", "time_to_eruption"]
        assert (
            len(df_sub) == 20
        ), "Submission should correspond to debug test set (20 samples)"
        assert df_sub["time_to_eruption"].dtype == float
        assert (
            df_sub["time_to_eruption"] >= 0
        ).all(), "Predictions must be non-negative"

        print("Submission verification passed.")
    else:
        raise FileNotFoundError(
            f"Submission file not generated at {Config.SUBMISSION_PATH}"
        )

    print("\n=== Demonstration Finished Successfully ===")


if __name__ == "__main__":
    main()
