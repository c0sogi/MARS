import os
import pandas as pd
import numpy as np
import joblib
import warnings

# Import library components
from library.training_pipeline import TrainingPipeline
from library.inference_pipeline import InferencePipeline
from library.config import PathConfig

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def main():
    print("Initializing Demonstration...")

    # 1. Instantiate the Training Pipeline
    pipeline = TrainingPipeline()

    # 2. Optimize Configuration for Speed
    # We modify the configuration instance within the pipeline to use very few trees.
    # This allows us to test the entire flow without waiting for convergence.
    print("Configuring hyperparameters for fast execution...")

    # Reduce estimators for Scout Models (Stage 1)
    pipeline.model_config.SCOUT_LGBM_PARAMS["n_estimators"] = 10
    pipeline.model_config.SCOUT_XGB_PARAMS["n_estimators"] = 10

    # Reduce estimators for Expert Models (Stage 3)
    pipeline.model_config.EXPERT_LGBM_PARAMS["n_estimators"] = 10
    pipeline.model_config.EXPERT_XGB_PARAMS["n_estimators"] = 10

    # 3. Run the Training Pipeline
    # We use a small sample_fraction (1%) to speed up the data loading and mining steps.
    # The FeatureEngineer will still process the tracking data once to generate the cache.
    print("Running Training Pipeline with sample_fraction=0.01...")
    pipeline.run(sample_fraction=0.01)

    # 4. Validate Training Artifacts
    print("Validating generated artifacts...")
    working_dir = PathConfig.WORKING_DIR

    expected_files = [
        "scout_lgbm.joblib",
        "scout_xgb.joblib",
        "expert_lgbm.joblib",
        "expert_xgb.joblib",
        "best_threshold.npy",
        "hard_negative_indices.npy",
        "train_features.parquet",
        "val_features.parquet",
    ]

    for fname in expected_files:
        fpath = os.path.join(working_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Expected artifact not found: {fpath}")

    print("All training artifacts verified.")

    # 5. Run the Inference Pipeline
    # This simulates the submission environment where we load saved models and predict on test data.
    print("Running Inference Pipeline...")
    inference = InferencePipeline()

    # This will load the models and threshold saved in step 3
    submission_df = inference.predict_test_set(load_cached=True)

    # 6. Validate Submission
    print("Validating submission file...")
    submission_path = PathConfig.SUBMISSION_FILE_PATH

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    # Reload to check content
    df_sub = pd.read_csv(submission_path)

    # Check columns
    expected_cols = ["contact_id", "contact"]
    if list(df_sub.columns) != expected_cols:
        raise ValueError(
            f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"
        )

    # Check values (must be binary 0 or 1)
    if not df_sub["contact"].isin([0, 1]).all():
        raise ValueError("Submission contains non-binary values in 'contact' column.")

    # Check length (should match the dataframe returned by inference)
    if len(df_sub) != len(submission_df):
        raise ValueError("Saved submission length differs from returned dataframe.")

    print(f"Submission verified. Rows: {len(df_sub)}")
    print("Demonstration completed successfully.")


if __name__ == "__main__":
    main()
