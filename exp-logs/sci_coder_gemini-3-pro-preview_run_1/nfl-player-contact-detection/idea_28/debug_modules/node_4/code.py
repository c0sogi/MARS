import os
import shutil
import numpy as np
import pandas as pd
import logging
import sys

# Import from provided library
from library.config import Config
from library.utils import seed_everything, setup_logging
from library.training_manager import TrainingManager
from library.data_manager import DataManager
from library.model_factory import TriModelEnsemble


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print(">>> Setting up configuration for fast demo execution...")

    # Define a separate working directory for the demo to avoid conflicts
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Monkey-patch Config to redirect outputs and ensure speed
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Reduce boosting rounds and early stopping for rapid iteration
    Config.NUM_BOOST_ROUNDS = 5
    Config.EARLY_STOPPING_ROUNDS = 2

    # Update model parameters to reflect reduced complexity
    # Note: We modify the dictionaries in-place
    Config.LGBM_PARAMS["n_estimators"] = 5
    Config.LGBM_PARAMS["verbosity"] = -1

    Config.XGB_PARAMS["n_estimators"] = 5
    Config.XGB_PARAMS["verbosity"] = 0

    Config.SKLEARN_HGB_PARAMS["max_iter"] = 5
    Config.SKLEARN_HGB_PARAMS["verbose"] = 0

    # Setup logging to file and stdout
    logger = setup_logging("demo.log")
    seed_everything(Config.SEED)

    print(f">>> Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Manager & Feature Generation
    # -------------------------------------------------------------------------
    print("\n>>> [Step 1] Testing DataManager with debug=True...")
    # debug=True causes the DataManager to sample a small subset of the metadata (10k rows)
    dm = DataManager(debug=True)

    # Test loading training data
    # This triggers the full feature engineering pipeline:
    # Merge -> Spectral Shock -> Gating -> Dual-Basis Projection
    print(
        "    Generating training features (this may take a moment to load tracking data)..."
    )
    train_df = dm.load_and_merge_data("train", load_cached_data=False)

    print(f"    Generated Train Data Shape: {train_df.shape}")

    # Validations
    expected_cols = ["contact", "distance", "v_comp1_p1", "spectral_energy_p1"]
    for col in expected_cols:
        assert col in train_df.columns, f"Missing expected column: {col}"

    # Ensure data is not empty (Gating shouldn't filter everything)
    assert not train_df.empty, "Training dataframe is empty after processing!"

    # Test loading validation data
    print("    Generating validation features...")
    val_df = dm.load_and_merge_data("val", load_cached_data=False)
    print(f"    Generated Val Data Shape: {val_df.shape}")
    assert not val_df.empty, "Validation dataframe is empty!"

    # -------------------------------------------------------------------------
    # 3. Training Manager: Phase 1 (Scouts)
    # -------------------------------------------------------------------------
    print("\n>>> [Step 2] Testing TrainingManager Phase 1: Train Scouts...")
    # Initialize TrainingManager with debug=True to use the sampled data logic
    tm = TrainingManager(debug=True)

    # Train the Scout models (LGBM, XGB, HGB)
    tm.train_scouts(force_retrain=True)

    # Verify models are saved to disk
    scout_dir = os.path.join(DEMO_DIR, "models", "scouts")
    for model_name in ["lgbm", "xgb", "hgb"]:
        model_path = os.path.join(scout_dir, f"{model_name}_model.joblib")
        assert os.path.exists(
            model_path
        ), f"Scout model {model_name} not saved at {model_path}"
    print("    Scout models trained and saved successfully.")

    # -------------------------------------------------------------------------
    # 4. Training Manager: Phase 2 (Hard Negative Mining)
    # -------------------------------------------------------------------------
    print("\n>>> [Step 3] Testing TrainingManager Phase 2: Mine Hard Negatives...")

    # Use the trained scouts to find hard negatives in the training set
    hard_indices = tm.mine_hard_negatives(force_remine=True)

    print(f"    Mined {len(hard_indices)} hard negatives.")
    assert isinstance(hard_indices, np.ndarray), "Hard indices should be a numpy array"

    indices_path = os.path.join(DEMO_DIR, "hard_negative_indices.npy")
    assert os.path.exists(indices_path), "Hard negative indices file not found."

    # -------------------------------------------------------------------------
    # 5. Training Manager: Phase 3 (Expert Training)
    # -------------------------------------------------------------------------
    print("\n>>> [Step 4] Testing TrainingManager Phase 3: Train Expert...")

    # Train the Expert models on the Anchored Dataset (Positives + Hard Negatives + Anchors)
    # This also applies temporal label smoothing
    tm.train_expert(force_retrain=True)

    # Verify expert models
    expert_dir = os.path.join(DEMO_DIR, "models", "expert")
    for model_name in ["lgbm", "xgb", "hgb"]:
        model_path = os.path.join(expert_dir, f"{model_name}_model.joblib")
        assert os.path.exists(model_path), f"Expert model {model_name} not saved."
    print("    Expert models trained and saved successfully.")

    # -------------------------------------------------------------------------
    # 6. Training Manager: Phase 4 (Threshold Optimization)
    # -------------------------------------------------------------------------
    print("\n>>> [Step 5] Testing TrainingManager Phase 4: Optimize Threshold...")

    # Find the best threshold on the validation set
    best_thresh = tm.optimize_threshold()
    print(f"    Optimal Threshold: {best_thresh}")

    assert 0.0 <= best_thresh <= 1.0, "Threshold must be a probability between 0 and 1"

    thresh_path = os.path.join(DEMO_DIR, "best_threshold.npy")
    assert os.path.exists(thresh_path), "Threshold file not saved."

    # -------------------------------------------------------------------------
    # 7. Training Manager: Phase 5 (Submission Generation)
    # -------------------------------------------------------------------------
    print("\n>>> [Step 6] Testing TrainingManager Phase 5: Generate Submission...")

    # Generate predictions for the test set
    # Note: Since debug=True, this runs on a subset of the test metadata
    tm.generate_submission()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission Rows: {len(df_sub)}")

    # Check submission format
    assert "contact_id" in df_sub.columns
    assert "contact" in df_sub.columns
    assert df_sub["contact"].isin([0, 1]).all(), "Predictions must be binary (0 or 1)"

    print("\n>>> Demo Execution Completed Successfully!")


if __name__ == "__main__":
    run_demo()
