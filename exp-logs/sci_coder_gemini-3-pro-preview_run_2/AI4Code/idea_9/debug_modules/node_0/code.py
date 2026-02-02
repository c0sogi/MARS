import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import tqdm

# ------------------------------------------------------------------------------
# 1. Setup and Configuration
# ------------------------------------------------------------------------------


# Monkey-patch tqdm to suppress progress bars
def nop(it, *a, **k):
    return it


tqdm.tqdm = nop

# Import library modules
from library.config import Config
from library.utils import seed_everything, compute_kendall_tau, validate_paths
from library.data_loader import NotebookLoader
from library.vectorizers import SemanticSpace
from library.anchor_features import AnchorEngine
from library.models import Stage1Ridge, Stage2LGBM
from library.pipeline import RankingPipeline

# Override Config for rapid demonstration
print("--- Configuring Environment for Demo ---")
Config.DEBUG = True
Config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 notebooks for speed
Config.NUM_WORKERS = 2
Config.WORKING_DIR = "./working/demo_run"
Config.SUBMISSION_DIR = "./working/demo_submission"
Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

# Reduce dimensionality and model complexity for speed
Config.TFIDF_PARAMS["max_features"] = 100
Config.SVD_N_COMPONENTS = 8
Config.N_FOLDS = 2
Config.LGBM_PARAMS["n_estimators"] = 10
Config.LGBM_PARAMS["num_leaves"] = 8
Config.LGBM_EARLY_STOPPING_ROUNDS = 5
Config.LGBM_VERBOSE_EVAL = -1  # Disable logging

# Initialize directories
Config.setup()
seed_everything(Config.RANDOM_SEED)


# ------------------------------------------------------------------------------
# 2. Utility Validation
# ------------------------------------------------------------------------------
def test_metrics():
    print("\n--- Testing Metrics (Kendall Tau) ---")
    # Case: 3 cells. GT: A B C. Pred: A C B.
    # GT Ranks: A=0, B=1, C=2
    # Pred Ranks: 0, 2, 1
    # Inversions: (2, 1) -> 1 swap.
    # Total pairs: n(n-1) = 3*2 = 6.
    # Score = 1 - 4 * (1/6) = 1 - 0.666... = 0.333...

    df_gt = pd.DataFrame({"id": ["nb1"], "cell_order": ["A B C"]})
    df_pred = pd.DataFrame({"id": ["nb1"], "cell_order": ["A C B"]})

    score = compute_kendall_tau(df_pred, df_gt)
    print(f"Calculated Score: {score:.4f}")

    expected_score = 1 - 4 * (1 / 6)
    assert abs(score - expected_score) < 1e-6, "Kendall Tau calculation incorrect"
    print("Metric validation passed.")


# ------------------------------------------------------------------------------
# 3. Data Loader Validation
# ------------------------------------------------------------------------------
def test_data_loader():
    print("\n--- Testing NotebookLoader ---")
    # Load training data (debug mode is on via Config)
    df_train = NotebookLoader.load_dataset(
        Config.TRAIN_METADATA_PATH,
        "debug_train_processed",
        load_cached_data=False,  # Force reload to test logic
        debug=True,
    )

    print(
        f"Loaded {len(df_train)} cells from {df_train['notebook_id'].nunique()} notebooks."
    )

    # Assertions
    required_cols = [
        "notebook_id",
        "cell_id",
        "cell_type",
        "source",
        "rank",
        "norm_rank",
    ]
    for col in required_cols:
        assert col in df_train.columns, f"Missing column: {col}"

    # Check if ranks are present for training data
    assert not df_train["norm_rank"].isnull().all(), "Training data should have ranks"

    return df_train


# ------------------------------------------------------------------------------
# 4. Vectorizer Validation
# ------------------------------------------------------------------------------
def test_vectorizers(df_train):
    print("\n--- Testing SemanticSpace (Vectorizers) ---")
    semantic_space = SemanticSpace()

    # Fit on training data
    semantic_space.fit(df_train, load_cached_models=False)

    # Test Transformation
    sample_text = df_train["source"].iloc[:5]

    tfidf_vec = semantic_space.transform_tfidf(sample_text)
    svd_vec = semantic_space.transform_svd(sample_text)

    print(f"TF-IDF Shape: {tfidf_vec.shape}")
    print(f"SVD Shape: {svd_vec.shape}")

    # Assertions
    assert tfidf_vec.shape[0] == 5
    assert tfidf_vec.shape[1] <= Config.TFIDF_PARAMS["max_features"]
    assert svd_vec.shape[0] == 5
    assert svd_vec.shape[1] == Config.SVD_N_COMPONENTS

    return semantic_space


# ------------------------------------------------------------------------------
# 5. Anchor Features Validation
# ------------------------------------------------------------------------------
def test_anchor_features(df_train, semantic_space):
    print("\n--- Testing AnchorEngine ---")
    anchor_engine = AnchorEngine(semantic_space)

    # Compute features
    # Note: This computes features for markdown cells relative to code cells
    features_df = anchor_engine.compute_features(
        df_train, "debug_train", load_cached_data=False
    )

    print(f"Computed anchor features for {len(features_df)} markdown cells.")
    print("Columns:", features_df.columns.tolist())

    # Assertions
    expected_cols = [
        "cell_id",
        "lexical_anchor_rank",
        "lexical_anchor_sim",
        "latent_anchor_rank",
        "latent_anchor_sim",
    ]
    for col in expected_cols:
        assert col in features_df.columns, f"Missing anchor feature: {col}"

    # Verify we have rows
    assert len(features_df) > 0, "No anchor features returned"

    return features_df


# ------------------------------------------------------------------------------
# 6. Model Validation
# ------------------------------------------------------------------------------
def test_models():
    print("\n--- Testing Models ---")

    # Synthetic Data
    n_samples = 100
    n_features = 10
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)
    groups = np.random.randint(0, 5, n_samples)

    # --- Stage 1: Ridge ---
    print("Testing Stage 1 Ridge...")
    s1 = Stage1Ridge()
    s1.fit(X, y)
    preds_s1 = s1.predict(X)
    assert preds_s1.shape == (n_samples,), "Stage 1 prediction shape mismatch"

    oof_preds = s1.get_oof_predictions(X, y, groups)
    assert oof_preds.shape == (n_samples,), "Stage 1 OOF prediction shape mismatch"

    # --- Stage 2: LightGBM ---
    print("Testing Stage 2 LightGBM...")
    s2 = Stage2LGBM()
    # Split for train/val
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]

    s2.fit(X_train, y_train, X_val, y_val)
    preds_s2 = s2.predict(X_val)
    assert preds_s2.shape == (20,), "Stage 2 prediction shape mismatch"

    print("Model tests passed.")


# ------------------------------------------------------------------------------
# 7. Pipeline Integration Validation
# ------------------------------------------------------------------------------
def test_pipeline():
    print("\n--- Testing Full RankingPipeline ---")

    # Instantiate Pipeline
    pipeline = RankingPipeline()

    # 1. Execute Training
    # This will load data, fit vectorizers, train Stage 1, build Stage 2 dataset, train Stage 2
    pipeline.execute_training()

    # Verify artifacts exist
    assert os.path.exists(os.path.join(Config.WORKING_DIR, "stage1_ridge.joblib"))
    assert os.path.exists(os.path.join(Config.WORKING_DIR, "stage2_lgbm.joblib"))

    # 2. Generate Submission
    pipeline.predict_submission()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH)

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(df_sub)} rows.")
    assert "id" in df_sub.columns and "cell_order" in df_sub.columns

    # Since we used DEBUG_SAMPLE_SIZE=50, submission should reflect test set size clipped by debug logic
    # The loader clips the metadata reading.
    assert len(df_sub) > 0, "Submission file is empty"
    print("Pipeline execution successful.")


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # Validate paths first
        validate_paths()

        # Run component tests
        test_metrics()
        df_train = test_data_loader()
        semantic_space = test_vectorizers(df_train)
        test_anchor_features(df_train, semantic_space)
        test_models()

        # Run integration test
        test_pipeline()

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\nVALIDATION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nEXECUTION ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
