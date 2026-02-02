import os
import sys
import pandas as pd
import numpy as np
import logging
import warnings
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Import library components
from library.config import KADM_CONFIG
from library.utils import setup_logger, seed_everything
from library.training_curriculum import TrainingCurriculum
from library.inference_engine import generate_submission
from library.data_loader import DataLoader


def run_demo():
    print("Initializing KADM-AE Demo Script...")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    # Create a local copy of the config to modify for this demo run
    demo_config = KADM_CONFIG.copy()

    # Define working directories for the demo
    demo_work_dir = "./working/demo_run"
    demo_cache_dir = os.path.join(demo_work_dir, "cache")
    demo_model_dir = os.path.join(demo_work_dir, "models")
    demo_submission_path = os.path.join(demo_work_dir, "submission.csv")

    # Override settings for speed and demonstration purposes
    demo_config["paths"]["working_dir"] = demo_work_dir
    demo_config["paths"]["cache_dir"] = demo_cache_dir
    demo_config["paths"]["model_dir"] = demo_model_dir
    demo_config["paths"]["submission_output"] = demo_submission_path

    # Enable debug mode to sample data (e.g., 200 rows)
    demo_config["settings"]["debug"] = True
    demo_config["settings"]["debug_sample_size"] = 500

    # Reduce training iterations for speed
    demo_config["training"]["num_boost_round"] = 10
    demo_config["training"]["early_stopping_rounds"] = 5
    demo_config["training"]["verbose_eval"] = -1  # Silent

    # Ensure directories exist
    os.makedirs(demo_work_dir, exist_ok=True)
    os.makedirs(demo_cache_dir, exist_ok=True)
    os.makedirs(demo_model_dir, exist_ok=True)

    # Setup logger
    logger = setup_logger(name="demo_script", level=logging.INFO)
    logger.info("Configuration configured for fast execution.")

    # Set seeds
    seed_everything(demo_config["settings"]["seed"])

    # -------------------------------------------------------------------------
    # 2. Data Loading & Feature Engineering Verification
    # -------------------------------------------------------------------------
    logger.info("Step 1: Verifying Data Loader and Feature Engineering...")

    # Instantiate DataLoader with demo config
    loader = DataLoader(config=demo_config)

    # Load a sample of training data to verify pipeline mechanics
    # load_cached_data=False forces computation to demonstrate the engine
    X_train, y_train, meta_train = loader.load_dataset(
        "train", apply_gating=True, load_cached_data=False
    )

    # Assertions for Data Loading
    assert not X_train.empty, "Feature matrix X_train should not be empty."
    assert len(X_train) == len(y_train), "X and y must have same length."
    # The feature engine pivots features with suffixes _{offset}, so 'dist' becomes 'dist_0' at t=0.
    assert "dist_0" in X_train.columns, "Feature 'dist_0' missing from feature matrix."
    assert "ttc_0" in X_train.columns, "Feature 'ttc_0' missing from feature matrix."

    logger.info(f"Data Loaded Successfully. Shape: {X_train.shape}")
    logger.info("Feature Engineering Logic Verified.")

    # -------------------------------------------------------------------------
    # 3. Training Curriculum Execution
    # -------------------------------------------------------------------------
    logger.info("Step 2: Executing Training Curriculum...")

    # Instantiate Curriculum
    curriculum = TrainingCurriculum(config=demo_config)

    # Run the full curriculum: Mining -> Expert Training -> Threshold Optimization
    # We allow caching here if available, but since we changed config/paths, it will likely recompute/save to new dir
    ensemble, best_threshold = curriculum.run(load_cached_data=True)

    # Assertions for Model Artifacts
    lgbm_path = os.path.join(demo_model_dir, "expert_lgbm.joblib")
    xgb_path = os.path.join(demo_model_dir, "expert_xgb.joblib")
    thresh_path = os.path.join(demo_model_dir, "best_threshold.npy")

    assert os.path.exists(lgbm_path), f"LGBM model file missing at {lgbm_path}"
    assert os.path.exists(xgb_path), f"XGB model file missing at {xgb_path}"
    assert os.path.exists(thresh_path), f"Threshold file missing at {thresh_path}"

    logger.info(f"Training Complete. Best Threshold: {best_threshold}")

    # -------------------------------------------------------------------------
    # 4. Inference and Submission Generation
    # -------------------------------------------------------------------------
    logger.info("Step 3: Generating Submission...")

    # Generate submission using the trained ensemble
    generate_submission(
        ensemble=ensemble,
        threshold=best_threshold,
        load_cached_data=True,
        config=demo_config,
    )

    # Assertions for Submission
    assert os.path.exists(demo_submission_path), "Submission file was not created."

    df_sub = pd.read_csv(demo_submission_path)
    assert not df_sub.empty, "Submission file is empty."
    assert "contact_id" in df_sub.columns, "contact_id column missing."
    assert "contact" in df_sub.columns, "contact column missing."

    # Check if predictions are binary
    unique_preds = df_sub["contact"].unique()
    assert np.all(np.isin(unique_preds, [0, 1])), "Predictions must be binary (0 or 1)."

    logger.info(f"Submission generated at {demo_submission_path}")
    logger.info(f"Submission Shape: {df_sub.shape}")

    print("\n" + "=" * 40)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 40)


if __name__ == "__main__":
    run_demo()
