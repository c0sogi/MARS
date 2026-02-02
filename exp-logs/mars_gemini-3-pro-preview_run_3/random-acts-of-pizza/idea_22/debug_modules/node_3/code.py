import os
import sys
import numpy as np
import pandas as pd
import warnings
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, load_data
from library.feature_engineering import FeaturePipeline
from library.model_definitions import HexEnsemble
from library.retraining_protocol import retrain_final_models


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides for Speed/Demo
    # -------------------------------------------------------------------------
    print("1. Configuring environment for fast demonstration...")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Set seed for reproducibility
    set_seed(42)

    # Override Config for speed and low memory usage
    Config.DEBUG = True  # Forces load_data to sample only ~100 rows
    Config.N_FOLDS = 2  # Minimum folds for CV
    Config.N_JOBS = 1  # Avoid overhead of multiprocessing for small data

    # Reduce dimensionality
    Config.TFIDF_TEXT_PARAMS["max_features"] = 50
    Config.TFIDF_HISTORY_PARAMS["max_features"] = 20
    Config.PCA_N_COMPONENTS = 10  # Must be < n_samples (100 in debug)

    # Reduce Model Complexity
    Config.RF_PARAMS.update({"n_estimators": 5, "n_jobs": 1})
    Config.XGB_PARAMS.update({"n_estimators": 5, "n_jobs": 1, "verbosity": 0})
    Config.KNN_PARAMS.update({"n_neighbors": 5, "n_jobs": 1})
    Config.LR_PARAMS["max_iter"] = 10

    # Use a separate working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_FILE_PATH = "./working/demo_run/submission/submission.csv"

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print("   Configuration updated. Debug mode: ON")

    # -------------------------------------------------------------------------
    # 2. Feature Engineering Pipeline Demonstration
    # -------------------------------------------------------------------------
    print("\n2. Running Feature Engineering Pipeline...")

    # Instantiate pipeline (load_cached_data=False forces re-computation for demo)
    pipeline = FeaturePipeline(load_cached_data=False)

    # Run the pipeline
    # This processes Metadata, Lexical (TF-IDF), Behavioral, Semantic (SBERT), and Manifold (PCA)
    data_dict = pipeline.run()

    # Verify Data Structure
    required_splits = ["train", "val", "test"]
    required_views = ["metadata", "lexical", "behavioral", "semantic", "manifold"]

    for split in required_splits:
        assert split in data_dict, f"Missing split '{split}' in data dictionary"
        for view in required_views:
            assert view in data_dict[split], f"Missing view '{view}' in split '{split}'"

    # Verify Shapes (Debug mode loads ~100 samples)
    n_train = data_dict["train"]["metadata"].shape[0]
    n_val = data_dict["val"]["metadata"].shape[0]
    n_test = data_dict["test"]["metadata"].shape[0]

    print(f"   Data processed successfully.")
    print(f"   Train samples: {n_train}, Val samples: {n_val}, Test samples: {n_test}")

    assert n_train > 0, "Training set is empty"
    assert n_val > 0, "Validation set is empty"
    assert n_test > 0, "Test set is empty"

    # Verify Target Variable
    assert "y" in data_dict["train"], "Target 'y' missing from train data"
    assert len(data_dict["train"]["y"]) == n_train, "Target length mismatch in train"

    # -------------------------------------------------------------------------
    # 3. Model Training Demonstration (HexEnsemble)
    # -------------------------------------------------------------------------
    print("\n3. Training HexEnsemble (Validation-Guided Retraining)...")

    # Initialize Ensemble
    ensemble = HexEnsemble()

    # Verify Base Learners initialization
    expected_models = [
        "lexical_bagger",
        "community_bagger",
        "semantic_booster",
        "semantic_bagger",
        "manifold_neighbor",
        "metadata_anchor",
    ]
    for name in expected_models:
        assert name in ensemble.base_learners, f"Base learner {name} not initialized"

    # Fit the ensemble
    # This runs:
    #   Phase 1: OOF Generation (CV)
    #   Phase 2: Meta-Learner Training
    #   Phase 3: Final Retraining on Train + Val
    ensemble.fit(data_dict["train"], data_dict["val"])

    print("   Ensemble training complete.")

    # Verify Meta-Learner is trained (check if coef_ exists)
    assert hasattr(ensemble.meta_learner, "coef_"), "Meta-learner not trained properly"

    # -------------------------------------------------------------------------
    # 4. Explicit Retraining Protocol Check
    # -------------------------------------------------------------------------
    print("\n4. Verifying standalone Retraining Protocol function...")
    # Although ensemble.fit() calls internal logic, we verify the standalone function
    # from library.retraining_protocol works as expected.
    try:
        retrain_final_models(ensemble, data_dict["train"], data_dict["val"])
        print("   Standalone retraining function executed successfully.")
    except Exception as e:
        raise AssertionError(f"Standalone retraining failed: {e}")

    # -------------------------------------------------------------------------
    # 5. Inference and Submission Generation
    # -------------------------------------------------------------------------
    print("\n5. Generating Predictions and Submission...")

    # Load Test IDs (needed for submission file)
    df_test_raw = load_data("test", debug=Config.DEBUG)
    test_ids = df_test_raw[Config.ID_COL].values

    # Ensure ID count matches feature count
    assert len(test_ids) == n_test, "Mismatch between Test IDs and Test Features"

    # Generate Submission
    ensemble.generate_submission(data_dict["test"], test_ids)

    # Verify Output File
    assert os.path.exists(Config.SUBMISSION_FILE_PATH), "Submission file not created"

    # Verify Content
    submission_df = pd.read_csv(Config.SUBMISSION_FILE_PATH)
    print(f"   Submission file loaded. Shape: {submission_df.shape}")

    assert submission_df.shape == (
        n_test,
        2,
    ), f"Expected shape ({n_test}, 2), got {submission_df.shape}"
    assert list(submission_df.columns) == [
        Config.ID_COL,
        Config.TARGET_COL,
    ], "Incorrect columns in submission"

    # Verify Probability Range
    preds = submission_df[Config.TARGET_COL]
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions out of probability range [0, 1]"

    print("\nAll demonstrations and verifications passed successfully!")


if __name__ == "__main__":
    main()
