import os
import sys
import numpy as np
import pandas as pd
import warnings
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Import from the provided library files
from library.config import Config
from library.features import FeatureEngineer
from library.data import DataManager
from library.model import XGBModelWrapper
from library.workflow import SemiSupervisedPipeline


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    print("Starting Library Demonstration...")
    set_seed(42)

    # =========================================================================
    # 1. Configuration Override for Fast Demonstration
    # =========================================================================
    print("\n[1] Configuring environment for fast execution...")

    # Modify Config global state to run on a small subset with minimal training
    Config.DEBUG_SAMPLES = 2000  # Only load 2000 samples
    Config.CV_FOLDS = 2  # Only 2 folds
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run/submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Update XGBoost params for speed (tiny number of trees)
    Config.XGB_PARAMS["n_estimators"] = 5
    Config.XGB_PARAMS["early_stopping_rounds"] = 1
    Config.XGB_PARAMS["verbosity"] = 0

    # Re-run setup to create new directories
    Config.setup()

    print(f"Debug Samples: {Config.DEBUG_SAMPLES}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # =========================================================================
    # 2. Feature Engineering Demonstration
    # =========================================================================
    print("\n[2] Demonstrating FeatureEngineer...")
    fe = FeatureEngineer()

    # Force processing from scratch (ignore cache) to test logic
    df_train, df_val, df_test = fe.process_data(load_cached_data=False)

    # Validation
    print("Validating Feature Engineering outputs...")

    # Check dimensions
    assert len(df_train) <= Config.DEBUG_SAMPLES, "Train set not subsampled correctly"
    assert len(df_val) <= Config.DEBUG_SAMPLES, "Val set not subsampled correctly"

    # Check Geometric Features
    expected_geo = "Euclidean_Distance_To_Hydrology"
    assert (
        expected_geo in df_train.columns
    ), f"Missing geometric feature: {expected_geo}"

    # Check Dual Representation (Dense Index)
    expected_dense = "Soil_Type_Index"
    assert (
        expected_dense in df_train.columns
    ), f"Missing dense index feature: {expected_dense}"

    # Check Target Mapping
    # Classes should be mapped to 0..5
    unique_targets = df_train[Config.TARGET_COL].unique()
    assert np.all(unique_targets >= 0) and np.all(
        unique_targets < Config.NUM_CLASSES
    ), f"Target mapping failed. Found values: {unique_targets}"

    print("Feature Engineering verification passed.")

    # =========================================================================
    # 3. Data Manager & Pseudo-Labeling Demonstration
    # =========================================================================
    print("\n[3] Demonstrating DataManager...")
    dm = DataManager()

    # Test Fold Generation
    folds = dm.get_folds(df_train, n_splits=Config.CV_FOLDS)
    assert len(folds) == Config.CV_FOLDS, "Incorrect number of folds generated"

    # Test Pseudo-Labeling Logic
    print("Testing Pseudo-Labeling logic...")
    # Create dummy probabilities for test set
    # Make the first sample highly confident (prob=1.0 for class 0)
    # Make others low confidence
    n_test = len(df_test)
    dummy_probs = np.full((n_test, Config.NUM_CLASSES), 0.1)
    dummy_probs[0, 0] = 1.0  # High confidence for class 0

    # Merge
    df_augmented = dm.merge_pseudo_labels(df_train, df_test, dummy_probs, threshold=0.9)

    # Expectation: df_augmented should have 1 more row than df_train
    assert (
        len(df_augmented) == len(df_train) + 1
    ), f"Pseudo-labeling failed. Expected {len(df_train)+1}, got {len(df_augmented)}"

    print("DataManager verification passed.")

    # =========================================================================
    # 4. Model Wrapper Demonstration
    # =========================================================================
    print("\n[4] Demonstrating XGBModelWrapper...")

    # Prepare data for model
    exclude = [Config.ID_COL, Config.TARGET_COL]
    feats = [c for c in df_train.columns if c not in exclude]

    X_sample = df_train[feats]
    y_sample = df_train[Config.TARGET_COL]

    model = XGBModelWrapper()

    # Train on small sample
    print("Training XGBoost model on sample data...")
    model.train(X_sample, y_sample, X_sample, y_sample)

    # Predict
    preds = model.predict(X_sample)
    probs = model.predict_proba(X_sample)

    assert len(preds) == len(X_sample), "Prediction shape mismatch"
    assert probs.shape == (
        len(X_sample),
        Config.NUM_CLASSES,
    ), "Probability shape mismatch"

    print("Model Wrapper verification passed.")

    # =========================================================================
    # 5. Full Pipeline Execution
    # =========================================================================
    print("\n[5] Running Full Semi-Supervised Pipeline...")

    # Instantiate pipeline
    pipeline = SemiSupervisedPipeline()

    # Run pipeline
    # This will use the cached data we generated in step 2 (if path matches)
    # or re-process. Since we set Config.WORKING_DIR, it looks there.
    # Note: FeatureEngineer inside pipeline uses Config.WORKING_DIR.
    # We already saved processed data to Config.WORKING_DIR in Step 2?
    # Actually, FeatureEngineer.process_data saves to cache.
    # Let's ensure the pipeline runs smoothly.

    pipeline.run_pipeline()

    # Verify Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file successfully created at {Config.SUBMISSION_PATH}")

        # Load and check format
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        assert Config.ID_COL in sub_df.columns
        assert Config.TARGET_COL in sub_df.columns
        assert len(sub_df) == len(df_test)
        print("Submission format verified.")
    else:
        raise FileNotFoundError("Pipeline finished but submission file not found.")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
