import os
import sys
import shutil
import pandas as pd
import numpy as np
import warnings
import torch

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import kendall_tau_metric, set_seed
from library.pipeline import OrderingPipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def run_demo():
    print("=== Starting Demonstration of AI4Code Notebook Ordering Pipeline ===\n")

    # --------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed and Demonstration
    # --------------------------------------------------------------------------
    print("1. Configuring environment for rapid demonstration...")

    # Set a specific working directory for this demo to avoid overwriting real work
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Update derived paths in Config based on new WORKING_DIR
    Config.TRAIN_CACHE_PATH = os.path.join(
        Config.WORKING_DIR, "train_processed.parquet"
    )
    Config.VAL_CACHE_PATH = os.path.join(Config.WORKING_DIR, "val_processed.parquet")
    Config.TEST_CACHE_PATH = os.path.join(Config.WORKING_DIR, "test_processed.parquet")
    Config.TFIDF_VECTORIZER_PATH = os.path.join(
        Config.WORKING_DIR, "tfidf_vectorizer.joblib"
    )
    Config.LSA_MODEL_PATH = os.path.join(Config.WORKING_DIR, "lsa_model.joblib")
    Config.RIDGE_MODEL_PATH = os.path.join(Config.WORKING_DIR, "ridge_model.joblib")
    Config.LGBM_MODEL_PATH = os.path.join(Config.WORKING_DIR, "lgbm_model.txt")

    # Reduce complexity for speed
    Config.TFIDF_MAX_FEATURES = 1000  # Reduced from 60000
    Config.LSA_COMPONENTS = 16  # Reduced from 128
    Config.NUM_FOLDS = 3  # Reduced from 5

    # Monkeypatch LightGBM parameters to run very few iterations
    original_get_lgbm_params = Config.get_lgbm_params

    def fast_lgbm_params(overrides=None):
        params = original_get_lgbm_params(overrides)
        # Drastically reduce estimators and leaves for demo speed
        params.update({"n_estimators": 10, "num_leaves": 8, "bagging_freq": 1})
        return params

    Config.get_lgbm_params = fast_lgbm_params

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    if os.path.exists(Config.SUBMISSION_DIR):
        shutil.rmtree(Config.SUBMISSION_DIR)

    # Ensure directories exist (Config.setup() does this, but we do it here to be safe after cleanup)
    Config.setup()

    print("   Configuration updated: Working dir set to", Config.WORKING_DIR)
    print("   Hyperparameters reduced for speed.")

    # --------------------------------------------------------------------------
    # 2. Validation of Metric Logic
    # --------------------------------------------------------------------------
    print("\n2. Validating Kendall Tau Metric logic...")

    # Case 1: Perfect Match
    # Notebook with 3 cells: A B C
    df_gt = pd.DataFrame([{"id": "nb1", "cell_order": "A B C"}])
    df_pred_perfect = pd.DataFrame([{"id": "nb1", "cell_order": "A B C"}])

    score_perfect = kendall_tau_metric(df_pred_perfect, df_gt)
    print(f"   Perfect Match Score: {score_perfect:.4f}")
    assert np.isclose(score_perfect, 1.0), "Metric failed: Perfect match should be 1.0"

    # Case 2: One Swap
    # Prediction: A C B (Swapping B and C requires 1 swap)
    # n=3, Total pairs = n(n-1) = 6. Swaps = 1.
    # K = 1 - 4 * (1/6) = 1 - 0.666... = 0.333...
    df_pred_swap = pd.DataFrame([{"id": "nb1", "cell_order": "A C B"}])
    score_swap = kendall_tau_metric(df_pred_swap, df_gt)
    print(f"   One Swap Score: {score_swap:.4f}")
    assert np.isclose(
        score_swap, 1 - 4 * (1 / 6)
    ), "Metric failed: Incorrect score for single swap"

    print("   Metric validation passed.")

    # --------------------------------------------------------------------------
    # 3. Pipeline Execution: Training
    # --------------------------------------------------------------------------
    print("\n3. Executing Training Pipeline (Debug Mode)...")

    pipeline = OrderingPipeline()

    # Run fit with debug=True to process only a small subset (e.g., 50 notebooks)
    # This tests Data Loading, Feature Engineering (TFIDF, LSA, BERT), Stacking, and Model Training
    pipeline.fit(load_cached_data=False, debug=True, num_debug_samples=50)

    # Verify Artifacts
    print("   Verifying training artifacts...")
    assert os.path.exists(Config.RIDGE_MODEL_PATH), "Stage 1 Ridge model not found."
    assert os.path.exists(Config.LGBM_MODEL_PATH), "Stage 2 LGBM model not found."
    assert os.path.exists(Config.TFIDF_VECTORIZER_PATH), "TFIDF Vectorizer not found."
    assert os.path.exists(Config.LSA_MODEL_PATH), "LSA Model not found."

    # Verify Processed Data Cache
    assert os.path.exists(Config.TRAIN_CACHE_PATH), "Train features cache not found."
    assert os.path.exists(Config.VAL_CACHE_PATH), "Val features cache not found."

    print("   Training pipeline completed and artifacts verified.")

    # --------------------------------------------------------------------------
    # 4. Pipeline Execution: Inference
    # --------------------------------------------------------------------------
    print("\n4. Executing Inference Pipeline (Debug Mode)...")

    # Run predict with debug=True
    pipeline.predict(load_cached_data=False, debug=True, num_debug_samples=20)

    # Verify Submission
    print("   Verifying submission file...")
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"   Submission shape: {df_sub.shape}")
    print(f"   Submission columns: {list(df_sub.columns)}")

    assert (
        "id" in df_sub.columns and "cell_order" in df_sub.columns
    ), "Submission missing required columns."
    assert len(df_sub) > 0, "Submission file is empty."

    # Check format of cell_order (should be space-delimited string)
    sample_order = df_sub.iloc[0]["cell_order"]
    assert isinstance(sample_order, str), "cell_order is not a string."
    assert len(sample_order.split()) > 0, "cell_order string is empty/invalid."

    print("   Inference pipeline completed and submission verified.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    try:
        run_demo()
    except AssertionError as e:
        print(f"\n!!! Validation Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
