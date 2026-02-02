import os
import sys
import shutil
import numpy as np
import pandas as pd
import warnings
import joblib

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.data_processing import load_and_process_data
from library.feature_extraction import FeaturePipeline
from library.model_registry import ModelRegistry
from library.ensemble_engine import DecaViewEnsemble


def run_demo():
    print("Initializing Demo Script...")

    # =========================================================================
    # 1. Configuration Overrides for Speed
    # =========================================================================
    print("Applying configuration overrides for fast execution...")

    # Redirect working directory to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Re-create directories since we changed the path
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce Cross-Validation Folds
    Config.N_FOLDS = 2

    # Reduce Feature Dimensionality
    Config.TFIDF_PARAMS["max_features"] = 100
    Config.TFIDF_PARAMS["min_df"] = 1
    Config.SUBREDDIT_VOCAB_SIZE = 50
    Config.SUBREDDIT_MIN_DF = 1

    # Reduce Model Complexity (Estimators, Iterations)
    # We iterate over the Config attributes to find model parameters
    for attr_name in dir(Config):
        if attr_name.endswith("_PARAMS"):
            params = getattr(Config, attr_name)
            if isinstance(params, dict):
                # Reduce trees for Random Forests / GBMs
                if "n_estimators" in params:
                    params["n_estimators"] = 2
                # Reduce iterations for Linear Models
                if "max_iter" in params:
                    params["max_iter"] = 10
                # Reduce early stopping rounds
                if "early_stopping_rounds" in params:
                    params["early_stopping_rounds"] = 1
                # Reduce depth to speed up
                if "max_depth" in params and params["max_depth"] is not None:
                    params["max_depth"] = 2
                # Reduce leaves for LGBM
                if "num_leaves" in params:
                    params["num_leaves"] = 4

    # =========================================================================
    # 2. Data Loading & Slicing
    # =========================================================================
    print("Loading and preprocessing data...")
    # We force load_cached_data=False to ensure we process the raw metadata
    # However, since we changed WORKING_DIR, it won't find the cache anyway.
    df_train_full, df_test_full = load_and_process_data(load_cached_data=False)

    # Slice data to a tiny subset for demonstration speed
    # Ensure we have at least enough samples for 2-fold CV (min 2 per class per fold)
    # We take 50 samples for train, 20 for test
    subset_size_train = 50
    subset_size_test = 20

    print(f"Slicing data: Train={subset_size_train}, Test={subset_size_test}")
    df_train_small = df_train_full.head(subset_size_train).copy()
    df_test_small = df_test_full.head(subset_size_test).copy()

    # Verify target distribution in small slice to prevent CV errors
    # If purely by chance we get only one class, we force a mix
    if df_train_small[Config.TARGET_COL].nunique() < 2:
        # Manually inject a positive/negative sample if missing
        neg_sample = df_train_full[df_train_full[Config.TARGET_COL] == 0].iloc[0]
        pos_sample = df_train_full[df_train_full[Config.TARGET_COL] == 1].iloc[0]
        df_train_small.iloc[0] = neg_sample
        df_train_small.iloc[1] = pos_sample

    # =========================================================================
    # 3. Pipeline Execution
    # =========================================================================
    print("Instantiating DecaViewEnsemble...")
    ensemble = DecaViewEnsemble()

    # Force the internal pipeline to use the new working directory
    ensemble.pipeline.cache_dir = os.path.join(Config.WORKING_DIR, "cache")
    ensemble.pipeline.models_dir = os.path.join(Config.WORKING_DIR, "models")
    ensemble.models_dir = os.path.join(Config.WORKING_DIR, "models")
    ensemble.submission_dir = Config.SUBMISSION_DIR

    # Ensure subdirs exist
    os.makedirs(ensemble.pipeline.cache_dir, exist_ok=True)
    os.makedirs(ensemble.models_dir, exist_ok=True)

    print("Running Ensemble Pipeline...")
    # This runs Feature Extraction -> CV Training -> Meta Learning -> Inference
    ensemble.run(df_train_small, df_test_small)

    # =========================================================================
    # 4. Validation
    # =========================================================================
    print("Validating outputs...")

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Check 1: File existence
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not generated at {submission_path}")

    # Check 2: Content format
    df_sub = pd.read_csv(submission_path)
    print(f"Submission shape: {df_sub.shape}")

    expected_rows = len(df_test_small)
    if len(df_sub) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} rows in submission, found {len(df_sub)}"
        )

    expected_cols = {Config.ID_COL, Config.TARGET_COL}
    if not expected_cols.issubset(df_sub.columns):
        raise AssertionError(
            f"Missing columns in submission. Expected {expected_cols}, found {set(df_sub.columns)}"
        )

    # Check 3: Probability range
    probs = df_sub[Config.TARGET_COL]
    if probs.min() < 0 or probs.max() > 1:
        raise AssertionError("Predictions contain values outside [0, 1] range.")

    # Check 4: Check if models were saved
    model_files = os.listdir(ensemble.models_dir)
    print(f"Generated {len(model_files)} model files.")
    if len(model_files) == 0:
        raise AssertionError("No model files were saved.")

    print("Demo completed successfully!")


if __name__ == "__main__":
    # Set fixed seed for reproducibility
    np.random.seed(42)
    run_demo()
