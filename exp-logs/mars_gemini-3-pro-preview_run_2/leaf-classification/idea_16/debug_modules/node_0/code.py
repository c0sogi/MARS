import os
import sys
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output (e.g. convergence warnings from reduced iterations)
warnings.filterwarnings("ignore")

# 1. Import Config and Patch for Speed
# We import config first to modify hyperparameters before other modules use them.
import library.config as conf

print("Patching configuration for rapid demonstration...")
# Reduce Random Forest complexity
conf.RF_N_ESTIMATORS = 10

# Reduce Logistic Regression grid search and complexity
conf.LR_CS_GRID = np.logspace(-1, 1, 3)  # Only test 3 regularization strengths
conf.LR_CV_FOLDS = 2  # Reduce CV folds from 5 to 2
conf.LR_MAX_ITER = 100  # Limit max iterations for speed

# Reduce Calibration complexity
conf.CALIBRATION_CV = 2

# 2. Import remaining library modules
# These modules will now see the updated configuration values
import library.engine as engine


def run_demonstration():
    print("-" * 40)
    print("Starting End-to-End Pipeline Demonstration")
    print("-" * 40)

    # Instantiate the PhaseManager which orchestrates the whole process
    manager = engine.PhaseManager()

    # Patch the ensemble selector to run fewer iterations (default is 100)
    manager.selector.n_iterations = 5
    print(f"Modified Ensemble Selector iterations to: {manager.selector.n_iterations}")

    # Execute the pipeline:
    # 1. Load Data (Train/Val/Test)
    # 2. Phase A: Train experts on Train, Select best ensemble on Val
    # 3. Phase B: Retrain selected experts on Train + Val
    # 4. Inference: Predict on Test
    # 5. Submission: Save to CSV
    manager.execute()

    # 3. Validate the Output
    print("\n" + "-" * 40)
    print("Validating Submission Output")
    print("-" * 40)

    submission_path = conf.SUBMISSION_PATH

    # Check file existence
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file was not created at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"Loaded submission file with shape: {df_sub.shape}")

    # Expected dimensions: 99 test rows, 100 columns (1 id + 99 species)
    expected_rows = 99
    expected_cols = 100

    if df_sub.shape != (expected_rows, expected_cols):
        raise AssertionError(
            f"Submission shape mismatch. Expected ({expected_rows}, {expected_cols}), "
            f"but got {df_sub.shape}"
        )

    # Check ID column
    if "id" not in df_sub.columns:
        raise AssertionError("Submission file is missing the 'id' column.")

    # Check Species columns (should be 99)
    species_cols = [c for c in df_sub.columns if c != "id"]
    if len(species_cols) != 99:
        raise AssertionError(f"Expected 99 species columns, found {len(species_cols)}.")

    # Validate Probabilities
    probs = df_sub[species_cols].values

    # Check for NaNs
    if np.isnan(probs).any():
        raise AssertionError("Submission contains NaN values.")

    # Check Range [0, 1]
    # Allow for tiny floating point errors slightly outside range, but generally strict
    if probs.min() < -1e-9 or probs.max() > 1.0 + 1e-9:
        raise AssertionError(
            f"Probabilities out of valid range [0, 1]. Range found: [{probs.min()}, {probs.max()}]"
        )

    # Check that predictions are not trivial (all zeros)
    if probs.sum() == 0:
        raise AssertionError("All predicted probabilities are zero.")

    print("Validation successful! The pipeline produced a valid submission file.")


if __name__ == "__main__":
    # Set global seed for reproducibility
    np.random.seed(42)
    run_demonstration()
