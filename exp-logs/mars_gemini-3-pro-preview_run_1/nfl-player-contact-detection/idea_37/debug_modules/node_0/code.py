import os
import sys
import pandas as pd
import numpy as np
import warnings
import shutil

# Suppress warnings for clean execution
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Import library modules
from library.config import Config
from library.utils import seed_everything, reduce_mem_usage
from library.training import Trainer
from library.inference import InferenceManager


def main():
    print(">>> Starting NFL Contact Detection Demo Script")

    # =========================================================================
    # 1. Configuration Override for Speed & Demo
    # =========================================================================
    print(">>> Configuring environment for rapid demonstration...")

    # Enable Debug mode to use data sampling
    Config.DEBUG = True
    # Use a small sample size to ensure pipeline finishes quickly (< 5 mins)
    # 2000 samples should be enough to get some positive examples and valid plays
    Config.DEBUG_SAMPLE_SIZE = 2000

    # Reduce Model Complexity
    Config.N_ESTIMATORS = 10  # Very few trees for speed
    Config.EARLY_STOPPING_ROUNDS = 5

    # Reduce Feature Complexity
    Config.FEATURE_WINDOW_SIZE = 2  # +/- 2 frames instead of 10

    # Adjust Parallelism
    Config.N_JOBS = 4  # Use 4 cores

    # Clean up previous working directory to ensure fresh run verification
    # (Be careful in production, but safe here for demo logic)
    if os.path.exists(Config.WORKING_DIR):
        try:
            shutil.rmtree(Config.WORKING_DIR)
        except Exception:
            pass
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set Seeds
    seed_everything(Config.SEED)

    # =========================================================================
    # 2. Verify Utility Functions
    # =========================================================================
    print(">>> Verifying Utility Functions...")

    # Test reduce_mem_usage
    df_dummy = pd.DataFrame(
        {
            "float_col": np.random.rand(100).astype(np.float64),
            "int_col": np.random.randint(0, 100, 100).astype(np.int64),
        }
    )

    df_optimized = reduce_mem_usage(df_dummy, verbose=False)

    # Check if types were downcast
    assert (
        df_optimized["float_col"].dtype == np.float32
    ), "Float64 not downcast to Float32"
    assert df_optimized["int_col"].dtype == np.int8, "Int64 not downcast to Int8"

    print("    Utilities verified successfully.")

    # =========================================================================
    # 3. Execute Training Pipeline
    # =========================================================================
    print("\n>>> Initializing Trainer...")
    trainer = Trainer()

    print(">>> Running Training Pipeline (Feature Extraction -> Scouts -> Experts)...")
    # load_cached_data=False forces the Feature Engineering step to run
    trainer.run(load_cached_data=False)

    # Verify Generated Artifacts
    print(">>> Verifying Training Artifacts...")
    models_dir = os.path.join(Config.WORKING_DIR, "models")

    expected_artifacts = [
        os.path.join(models_dir, "scout_lgbm.joblib"),
        os.path.join(models_dir, "scout_xgb.joblib"),
        os.path.join(Config.WORKING_DIR, "hard_negative_indices.npy"),
        os.path.join(models_dir, "expert_lgbm.joblib"),
        os.path.join(models_dir, "expert_xgb.joblib"),
        os.path.join(models_dir, "best_threshold.npy"),
    ]

    for artifact in expected_artifacts:
        assert os.path.exists(artifact), f"Missing artifact: {artifact}"

    # Verify Threshold Validity
    threshold = np.load(os.path.join(models_dir, "best_threshold.npy"))[0]
    print(f"    Optimized Threshold: {threshold:.4f}")
    assert 0.0 < threshold < 1.0, "Threshold is out of valid probability range (0,1)"

    print("    Training Pipeline Verified.")

    # =========================================================================
    # 4. Execute Inference Pipeline
    # =========================================================================
    print("\n>>> Initializing Inference Manager...")
    inference_manager = InferenceManager()

    print(">>> Generating Submission...")
    # load_cached_data=False forces test feature generation
    inference_manager.generate_submission(load_cached_data=False)

    # Verify Submission File
    print(">>> Verifying Submission File...")
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    df_sub = pd.read_csv(submission_path)

    # 1. Check Shape (Should match DEBUG_SAMPLE_SIZE roughly, limited by available test metadata)
    print(f"    Submission Shape: {df_sub.shape}")
    assert not df_sub.empty, "Submission DataFrame is empty"

    # 2. Check Columns
    assert "contact_id" in df_sub.columns, "Missing 'contact_id' column"
    assert "contact" in df_sub.columns, "Missing 'contact' column"

    # 3. Check Data Integrity
    assert df_sub["contact"].isin([0, 1]).all(), "Predictions must be binary (0 or 1)"

    # 4. Check if we actually predicted any contacts (not strictly required, but good for demo)
    n_contacts = df_sub["contact"].sum()
    print(f"    Predicted Contacts: {n_contacts}")

    print("    Inference Pipeline Verified.")

    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    main()
