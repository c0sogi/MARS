import os
import sys
import numpy as np
import pandas as pd
import shutil

# Ensure the library modules can be imported
sys.path.append(".")

from library import config
from library import signal_processing
from library import feature_extraction
from library import data_manager
from library import model_trainer


def setup_demo_config():
    """
    Modifies the configuration to ensure the demo runs quickly.
    """
    print("--- Configuring Environment for Fast Demonstration ---")

    # Reduce data size for debug mode
    config.DEBUG_SAMPLE_SIZE = 20
    print(f"Set DEBUG_SAMPLE_SIZE to {config.DEBUG_SAMPLE_SIZE}")

    # Reduce Cross-Validation folds
    config.N_FOLDS = 2
    print(f"Set N_FOLDS to {config.N_FOLDS}")

    # Reduce LightGBM complexity
    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["min_child_samples"] = 2  # Allow splits on small data
    config.EARLY_STOPPING_ROUNDS = 5
    print(f"Set LGBM n_estimators to {config.LGBM_PARAMS['n_estimators']}")

    # Ensure clean state for working directory features to force generation logic
    # (Optional, but good for verifying the generation code works)
    features_files = [
        os.path.join(config.WORKING_DIR, "train_features_debug.parquet"),
        os.path.join(config.WORKING_DIR, "val_features_debug.parquet"),
        os.path.join(config.WORKING_DIR, "test_features_debug.parquet"),
    ]
    for f in features_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"Removed cached file: {f}")


def test_signal_processing():
    """
    Validates the signal processing functions with synthetic data.
    """
    print("\n--- Testing Signal Processing Module ---")

    # Generate synthetic signal: 1000 samples, 10 Hz sine wave
    fs = 100
    t = np.linspace(0, 10, 1000, endpoint=False)
    signal = np.sin(2 * np.pi * 10 * t)

    # 1. Test fill_missing_values
    signal_with_nan = signal.copy()
    signal_with_nan[10:20] = np.nan
    filled_signal = signal_processing.fill_missing_values(signal_with_nan)
    assert not np.isnan(
        filled_signal
    ).any(), "fill_missing_values failed to remove NaNs"
    assert len(filled_signal) == len(signal), "Signal length changed after filling NaNs"
    print("fill_missing_values: OK")

    # 2. Test apply_savitzky_golay
    trend = signal_processing.apply_savitzky_golay(
        signal, window_length=11, polyorder=2
    )
    assert len(trend) == len(signal), "Trend signal length mismatch"
    print("apply_savitzky_golay: OK")

    # 3. Test compute_derivatives
    vel, acc = signal_processing.compute_derivatives(trend)
    assert len(vel) == len(signal), "Velocity length mismatch"
    assert len(acc) == len(signal), "Acceleration length mismatch"
    print("compute_derivatives: OK")

    # 4. Test compute_welch_psd
    freqs, psd = signal_processing.compute_welch_psd(signal, fs=fs)
    assert len(freqs) == len(psd), "Frequency and PSD array length mismatch"
    assert np.all(psd >= 0), "PSD contains negative values"
    print("compute_welch_psd: OK")

    # 5. Test compute_band_powers
    bands = [(1, 5), (5, 15), (15, 50)]
    powers = signal_processing.compute_band_powers(freqs, psd, bands)
    assert isinstance(powers, dict), "Band powers should return a dictionary"
    assert len(powers) == 3, "Incorrect number of bands computed"
    # The 10Hz signal should have significant power in the 5-15Hz band
    assert powers["band_5_15"] > powers["band_1_5"], "Spectral power logic check failed"
    print("compute_band_powers: OK")

    # 6. Test compute_wavelet_energy
    energy = signal_processing.compute_wavelet_energy(signal)
    assert (
        "energy_detail" in energy and "energy_approx" in energy
    ), "Missing wavelet energy keys"
    print("compute_wavelet_energy: OK")

    # 7. Test get_signal_stats
    stats = signal_processing.get_signal_stats(signal)
    required_stats = ["mean", "std", "min", "max", "rms"]
    assert all(k in stats for k in required_stats), "Missing signal statistics"
    print("get_signal_stats: OK")


def test_feature_extraction():
    """
    Validates the feature extraction module using a dummy DataFrame.
    """
    print("\n--- Testing Feature Extraction Module ---")

    # Create dummy sensor data
    rows = 200
    data = {
        "sensor_1": np.random.randn(rows).astype(np.float32),
        "sensor_2": np.random.randn(rows).astype(np.float32),
    }
    df = pd.DataFrame(data)

    # Extract features
    features = feature_extraction.extract_segment_features(df)

    assert isinstance(features, dict), "Output should be a dictionary"

    # Check for expected keys for sensor_1
    # We expect keys like 'sensor_1_trend_vel_mean', 'sensor_1_spec_spectral_centroid', etc.
    keys = features.keys()

    has_trend = any("sensor_1_trend" in k for k in keys)
    has_spec = any("sensor_1_spec" in k for k in keys)
    has_temp = any("sensor_1_temp" in k for k in keys)

    assert has_trend, "Missing trend features for sensor_1"
    assert has_spec, "Missing spectral features for sensor_1"
    assert has_temp, "Missing temporal features for sensor_1"

    # Check that sensor_2 is also processed
    assert any("sensor_2" in k for k in keys), "Missing features for sensor_2"

    # Check that missing sensors (e.g., sensor_3) are not in keys
    assert not any("sensor_3" in k for k in keys), "Found features for missing sensor_3"

    print(f"Extracted {len(features)} features from dummy segment.")
    print("extract_segment_features: OK")


def test_pipeline_execution():
    """
    Validates the Data Manager and Model Trainer by running a complete training and inference cycle
    on a small subset of data (Debug Mode).
    """
    print("\n--- Testing Full Pipeline (Data Manager + Model Trainer) ---")

    trainer = model_trainer.ModelTrainer()

    # 1. Train Cross-Validation
    # This triggers data_manager.generate_feature_matrix internally
    print("Running training in debug mode...")
    trainer.train_cross_validation(debug=True)

    assert (
        len(trainer.models) == config.N_FOLDS
    ), f"Expected {config.N_FOLDS} models, found {len(trainer.models)}"
    assert trainer.best_score != float("inf"), "Best score was not updated"
    print("Training completed successfully.")

    # 2. Predict on Test Set
    print("Running prediction in debug mode...")
    trainer.predict(debug=True)

    # 3. Validate Submission File
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    sub_df = pd.read_csv(submission_path)
    assert "segment_id" in sub_df.columns, "Submission missing segment_id column"
    assert (
        "time_to_eruption" in sub_df.columns
    ), "Submission missing time_to_eruption column"
    assert len(sub_df) > 0, "Submission file is empty"

    print(f"Submission file verified: {len(sub_df)} rows.")
    print("Pipeline Execution: OK")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)

    try:
        setup_demo_config()
        test_signal_processing()
        test_feature_extraction()
        test_pipeline_execution()

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\nVALIDATION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
