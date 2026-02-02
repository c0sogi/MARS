import os
import shutil
import numpy as np
import pandas as pd
import random
import warnings
import lightgbm as lgb
from library.config import Config
from library.utils import count_inversions, kendall_tau, format_submission
from library.train_pipeline import run_training
from library.inference_pipeline import run_inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    print("=== Starting Demonstration of AI4Code Ranking Solution ===")

    # 1. Setup Configuration for Fast Demonstration
    print("\n[Demo] Configuring environment for rapid execution...")

    # Override Config defaults to run a "mini" version of the task
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 notebooks
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run/submission"

    # Reduce Vectorizer complexity
    Config.TFIDF_PARAMS["max_features"] = 1000
    Config.SVD_PARAMS["n_components"] = 16
    Config.SVD_PARAMS["n_iter"] = 2

    # Reduce Model complexity
    Config.N_FOLDS = 2
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["num_leaves"] = 8
    Config.LGBM_EARLY_STOPPING_ROUNDS = 5
    Config.LGBM_VERBOSE_EVAL = False  # Silent mode

    # Ensure clean state
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.setup()

    set_seed(Config.SEED)

    # 2. Verify Utility Functions
    print("\n[Demo] Verifying Utility Logic (utils.py)...")

    # Test count_inversions
    # [2, 1, 0] -> pairs: (2,1), (2,0), (1,0) -> 3 inversions
    inv_count = count_inversions([2, 1, 0])
    assert inv_count == 3, f"Expected 3 inversions, got {inv_count}"

    # Test kendall_tau
    # Perfect match
    gt = [["a", "b", "c"], ["x", "y"]]
    pred_perfect = [["a", "b", "c"], ["x", "y"]]
    score_perfect = kendall_tau(gt, pred_perfect)
    assert np.isclose(score_perfect, 1.0), f"Expected 1.0, got {score_perfect}"

    # Worst case: Reverse order
    # Notebook 1: n=3, pairs=3*2=6. Swaps for reverse [c, b, a] is 3.
    # Notebook 2: n=2, pairs=2*1=2. Swaps for reverse [y, x] is 1.
    # Total Swaps = 4. Total Pairs = 8.
    # K = 1 - 4 * (4 / 8) = 1 - 2 = -1.0
    pred_worst = [["c", "b", "a"], ["y", "x"]]
    score_worst = kendall_tau(gt, pred_worst)
    assert np.isclose(score_worst, -1.0), f"Expected -1.0, got {score_worst}"

    print("Utils verification passed.")

    # 3. Execute Training Pipeline
    # This demonstrates data_loader, vectorizer, feature_extractor, and model_wrapper
    print("\n[Demo] Executing Training Pipeline (train_pipeline.py)...")

    # We set load_cached_data=False to force the pipeline to run logic components
    try:
        run_training(debug=True, load_cached_data=False)
    except Exception as e:
        print(f"Training pipeline failed with error: {e}")
        raise e

    # Validation of Training Artifacts
    expected_artifacts = [
        "text_vectorizer_tfidf.joblib",
        "text_vectorizer_svd.joblib",
        "stage1_ridge_model.joblib",
        "stage2_lgbm.txt",
        "train_debug_dataframe.parquet",
        "val_debug_dataframe.parquet",
        "train_anchor_features.parquet",
        "val_anchor_features.parquet",
    ]

    print("\n[Demo] Validating Training Artifacts...")
    for artifact in expected_artifacts:
        path = os.path.join(Config.WORKING_DIR, artifact)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected training artifact missing: {path}")
    print("All training artifacts generated successfully.")

    # 4. Execute Inference Pipeline
    # This demonstrates end-to-end prediction flow
    print("\n[Demo] Executing Inference Pipeline (inference_pipeline.py)...")

    try:
        run_inference(debug=True, load_cached_data=False)
    except Exception as e:
        print(f"Inference pipeline failed with error: {e}")
        raise e

    # Validation of Submission
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    # Fix: Fill NaNs (from empty strings in debug mode) with empty strings
    df_sub["cell_order"] = df_sub["cell_order"].fillna("")

    print(f"\n[Demo] Submission Generated. Shape: {df_sub.shape}")

    # Basic check on submission content
    assert (
        "id" in df_sub.columns and "cell_order" in df_sub.columns
    ), "Submission columns missing"
    assert len(df_sub) > 0, "Submission file is empty"

    # Check if cell_order contains space-delimited strings
    sample_order = df_sub.iloc[0]["cell_order"]
    assert isinstance(sample_order, str), "cell_order should be a string"
    # Even if empty notebook, it might be empty string, but usually contains IDs

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
