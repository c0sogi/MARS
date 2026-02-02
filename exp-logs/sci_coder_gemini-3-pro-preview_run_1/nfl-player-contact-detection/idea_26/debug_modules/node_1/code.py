import os
import shutil
import numpy as np
import pandas as pd
import joblib
import warnings

# Import provided libraries
import library.config as config
import library.utils as utils
from library.physics_engine import FeatureManager
from library.model_factory import TriEnsemble
from library.training_pipeline import TrainingPipeline
from library.inference_pipeline import InferencePipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def run_demo():
    print("=== Starting VASM-E Library Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("[1] Overriding configuration for fast execution...")

    # Reduce ensemble complexity
    config.N_ESTIMATORS = 10
    config.EARLY_STOPPING_ROUNDS = 5

    # Update model params dictionaries in config
    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["verbose"] = -1
    config.XGB_PARAMS["n_estimators"] = 10
    config.XGB_PARAMS["verbosity"] = 0
    config.HGB_PARAMS["max_iter"] = 10
    config.HGB_PARAMS["verbose"] = 0

    # Use a clean working directory for this demo run
    # We keep the path but ensure we start fresh if needed
    if os.path.exists(config.CACHE_DIR):
        print(f"    Cleaning cache directory: {config.CACHE_DIR}")
        shutil.rmtree(config.CACHE_DIR)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Verify Utilities
    # -------------------------------------------------------------------------
    print("\n[2] Verifying library.utils...")

    # A. Memory Reduction
    df_dummy = pd.DataFrame(
        {
            "a": np.random.rand(100).astype(np.float64),
            "b": np.random.randint(0, 100, 100).astype(np.int64),
        }
    )
    start_dtype_a = df_dummy["a"].dtype
    df_reduced = utils.reduce_mem_usage(df_dummy)
    assert (
        df_reduced["a"].dtype != start_dtype_a
    ), "Float64 should be reduced to Float32"
    print("    Memory reduction verified.")

    # B. Gaussian Smoothing
    # Create a synthetic temporal sequence for a single pair
    df_smooth = pd.DataFrame(
        {
            "game_play": ["game1"] * 10,
            "nfl_player_id_1": [1] * 10,
            "nfl_player_id_2": [2] * 10,
            "step": np.arange(10),
            "contact": [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        }
    )
    df_smooth = utils.gaussian_smooth_labels(df_smooth, sigma=1.0)

    # Check that smoothing happened (values should not be just 0 or 1 anymore)
    # specifically at the edges of the contact event
    assert "contact_smooth" in df_smooth.columns
    assert (
        0.0 < df_smooth.loc[2, "contact_smooth"] < 1.0
    ), "Smoothing failed at rising edge"
    assert (
        0.0 < df_smooth.loc[6, "contact_smooth"] < 1.0
    ), "Smoothing failed at falling edge"
    print("    Gaussian smoothing verified.")

    # -------------------------------------------------------------------------
    # 3. Verify Feature Manager (Physics Engine)
    # -------------------------------------------------------------------------
    print("\n[3] Verifying library.physics_engine.FeatureManager...")

    fm = FeatureManager()

    # Process a small sample of training data
    # This tests loading, merging, gating, and vector feature calculation
    debug_sample_size = 500
    print(f"    Processing {debug_sample_size} rows of training data...")

    df_features = fm.process_data(
        split="train", load_cached_data=False, debug_sample=debug_sample_size
    )

    # Validation
    assert not df_features.empty, "Feature DataFrame is empty"
    assert "v_radial" in df_features.columns, "Vector feature 'v_radial' missing"
    assert (
        "a_radial_energy" in df_features.columns
    ), "Derived feature 'a_radial_energy' missing"

    # Check Gating logic: If distance is large, it should have been filtered out
    # or if we check the logic, the gating function removes rows.
    # We just ensure the pipeline ran through.
    print(f"    Generated features shape: {df_features.shape}")

    # -------------------------------------------------------------------------
    # 4. Verify Model Factory (TriEnsemble)
    # -------------------------------------------------------------------------
    print("\n[4] Verifying library.model_factory.TriEnsemble...")

    # Create synthetic data compatible with the model
    # We need features defined in config.MODEL_FEATURES
    n_samples = 100
    X_synth = pd.DataFrame(
        np.random.rand(n_samples, len(config.MODEL_FEATURES)),
        columns=config.MODEL_FEATURES,
    )
    y_synth = np.random.randint(0, 2, n_samples)

    model = TriEnsemble()

    # Test Fit
    print("    Fitting ensemble on synthetic data...")
    model.fit(X_synth, y_synth)

    # Test Predict
    probs = model.predict_proba(X_synth)
    assert probs.shape == (n_samples,), "Prediction shape mismatch"
    assert probs.min() >= 0.0 and probs.max() <= 1.0, "Probabilities out of range"

    # Test Save/Load
    model.save("demo_model.joblib")
    loaded_model = TriEnsemble.load("demo_model.joblib")
    probs_loaded = loaded_model.predict_proba(X_synth)
    np.testing.assert_allclose(
        probs, probs_loaded, err_msg="Model save/load inconsistency"
    )
    print("    TriEnsemble fit/predict/save/load verified.")

    # -------------------------------------------------------------------------
    # 5. Verify Training Pipeline
    # -------------------------------------------------------------------------
    print("\n[5] Running library.training_pipeline.TrainingPipeline...")

    pipeline = TrainingPipeline()

    # Run the full pipeline with debug sampling
    # This triggers Scout Training -> Mining -> Expert Training -> Evaluation
    mcc_score = pipeline.run(debug_sample=500, load_cached_data=True)

    # Verify outputs
    assert os.path.exists(
        os.path.join(config.MODEL_DIR, "expert_ensemble.joblib")
    ), "Expert model not saved"
    assert os.path.exists(
        os.path.join(config.MODEL_DIR, "best_threshold.npy")
    ), "Threshold file not saved"
    print(f"    Pipeline completed with MCC: {mcc_score:.4f}")

    # -------------------------------------------------------------------------
    # 6. Verify Inference Pipeline
    # -------------------------------------------------------------------------
    print("\n[6] Running library.inference_pipeline.InferencePipeline...")

    inf_pipeline = InferencePipeline()

    # Run inference on a small sample of test data
    submission_path = inf_pipeline.run(load_cached_data=False, debug_sample=200)

    # Verify submission file
    assert os.path.exists(submission_path), "Submission file not created"

    df_sub = pd.read_csv(submission_path)
    expected_cols = ["contact_id", "contact"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Found {df_sub.columns}"
    assert df_sub["contact"].isin([0, 1]).all(), "Submission contains non-binary values"

    print(f"    Submission generated at: {submission_path}")
    print(f"    Submission rows: {len(df_sub)}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Set fixed seeds for reproducibility
    np.random.seed(42)
    run_demo()
