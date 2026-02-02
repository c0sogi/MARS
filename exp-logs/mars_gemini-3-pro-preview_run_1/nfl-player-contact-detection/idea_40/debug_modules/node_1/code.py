import os
import shutil
import numpy as np
import pandas as pd
import warnings
from library.config import Config
from library.utils import seed_everything, setup_logging
from library.data_manager import DataManager
from library.training import Trainer
from library.inference import InferenceManager

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("Initializing Demo Run...")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed & Demo
    # -------------------------------------------------------------------------
    # Modify Config global state to run a fast, small-scale demo
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 500  # Process only 500 rows for speed
    Config.WORKING_DIR = "./working/demo_execution"
    Config.USE_CACHE = False  # Force feature re-calculation to test logic
    Config.N_JOBS = 2

    # Adjust Model Hyperparameters for near-instant training
    Config.LGBM_PARAMS["n_estimators"] = 2
    Config.LGBM_PARAMS["num_leaves"] = 8
    Config.LGBM_PARAMS["max_depth"] = 3

    Config.XGB_PARAMS["n_estimators"] = 2
    Config.XGB_PARAMS["max_depth"] = 3

    # Lower threshold to ensure we pick up some "hard negatives" even with random weights
    Config.HARD_NEGATIVE_THRESHOLD = 0.0001

    # Clean working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds
    seed_everything(Config.SEED)
    setup_logging(level=40)  # ERROR level to suppress verbose info logs

    # -------------------------------------------------------------------------
    # 2. Data Manager & Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[1/4] Testing Feature Engineering Pipeline...")
    dm = DataManager(Config)

    # Test Train Data Loading
    df_train = dm.get_train_data(load_cached_data=False)
    assert not df_train.empty, "Training data should not be empty."
    assert "distance" in df_train.columns, "Feature 'distance' missing from train data."
    assert "ttc" in df_train.columns, "Feature 'ttc' missing from train data."
    # Check that metadata columns were preserved where expected (DataManager.get_X_y strips them later)
    assert "game_play" in df_train.columns
    print(f"  -> Train Data Shape: {df_train.shape} (Success)")

    # Test Validation Data Loading
    df_val = dm.get_val_data(load_cached_data=False)
    assert not df_val.empty, "Validation data should not be empty."
    print(f"  -> Val Data Shape: {df_val.shape} (Success)")

    # -------------------------------------------------------------------------
    # 3. Training Pipeline
    # -------------------------------------------------------------------------
    print("\n[2/4] Testing Training Pipeline...")
    trainer = Trainer(Config)

    # A. Train Scouts
    print("  -> Training Scout Models...")
    scouts = trainer.train_scouts(df_train)
    assert len(scouts) == 2, "Expected 2 scout models (LGBM + XGB)."
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "models", "scout_lgbm.joblib")
    ), "Scout LGBM artifact missing."
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "models", "scout_xgb.joblib")
    ), "Scout XGB artifact missing."

    # B. Mine Hard Negatives
    print("  -> Mining Hard Negatives...")
    hard_neg_indices = trainer.mine_hard_negatives(df_train, scouts, load_cached=False)
    # Note: It's acceptable if this is empty in a tiny random sample, but logic must run.
    print(f"     Found {len(hard_neg_indices)} hard negative candidates.")
    assert isinstance(
        hard_neg_indices, np.ndarray
    ), "Hard negative indices should be a numpy array."

    # C. Train Experts
    print("  -> Training Expert Models...")
    experts = trainer.train_experts(df_train, hard_neg_indices, df_val)
    assert len(experts) == 2, "Expected 2 expert models."
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "models", "expert_lgbm.joblib")
    ), "Expert LGBM artifact missing."

    # D. Optimize Threshold
    print("  -> Optimizing Threshold...")
    best_thresh = trainer.optimize_threshold(experts, df_val)
    assert 0.0 <= best_thresh <= 1.0, "Threshold must be a probability between 0 and 1."
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "models", "best_threshold.npy")
    ), "Threshold artifact missing."
    print(f"     Optimal Threshold: {best_thresh:.4f}")

    # -------------------------------------------------------------------------
    # 4. Inference Pipeline
    # -------------------------------------------------------------------------
    print("\n[3/4] Testing Inference Pipeline...")
    inference_mgr = InferenceManager(Config)

    # Run full prediction flow on Test Set
    # We force load_cached_data=False to ensure the pipeline processes the raw test metadata
    submission = inference_mgr.predict_test_set(load_cached_data=False)

    # Validations
    assert isinstance(submission, pd.DataFrame), "Output must be a DataFrame."
    assert "contact_id" in submission.columns, "Submission missing 'contact_id'."
    assert "contact" in submission.columns, "Submission missing 'contact' column."

    # Check binary predictions
    unique_preds = submission["contact"].unique()
    assert np.all(
        np.isin(unique_preds, [0, 1])
    ), f"Predictions must be binary (0/1). Found: {unique_preds}"

    # Check file output
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not created at {Config.SUBMISSION_PATH}"

    print(f"  -> Submission Generated: {len(submission)} rows.")
    print(f"  -> File saved to: {Config.SUBMISSION_PATH}")

    print("\n[4/4] Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
