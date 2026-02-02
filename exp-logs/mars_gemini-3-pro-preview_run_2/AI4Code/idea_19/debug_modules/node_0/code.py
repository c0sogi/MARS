import os
import sys
import shutil
import numpy as np
import pandas as pd
import warnings
import random
import joblib

# 1. Import Config first to override settings for a fast demo run
from library.config import Config

# --- Configuration Overrides for Demo ---
# We modify the Config class attributes directly to ensure all modules see these changes.
Config.DEBUG = True
Config.DEBUG_SAMPLE_SIZE = 30  # Small sample for speed
Config.WORKING_DIR = "./working/demo_run"
Config.SUBMISSION_DIR = "./working/demo_run/submission"

# Update paths based on new working dir
Config.PATH_TFIDF_VECTORIZER = os.path.join(
    Config.WORKING_DIR, "code_tfidf_vectorizer.joblib"
)
Config.PATH_SVD_MODEL = os.path.join(Config.WORKING_DIR, "code_svd_model.joblib")
Config.PATH_RIDGE_MODEL = os.path.join(Config.WORKING_DIR, "ridge_model.joblib")
Config.PATH_LGBM_MODEL = os.path.join(Config.WORKING_DIR, "lgbm_model.txt")

Config.CACHE_TRAIN_DATAFRAME = os.path.join(Config.WORKING_DIR, "mini_train.csv")
Config.CACHE_VAL_DATAFRAME = os.path.join(Config.WORKING_DIR, "mini_val.csv")
Config.CACHE_TEST_DATAFRAME = os.path.join(Config.WORKING_DIR, "mini_test.csv")

Config.CACHE_TRAIN_FEATURES = os.path.join(
    Config.WORKING_DIR, "mini_train_feats.parquet"
)
Config.CACHE_VAL_FEATURES = os.path.join(Config.WORKING_DIR, "mini_val_feats.parquet")
Config.CACHE_TEST_FEATURES = os.path.join(Config.WORKING_DIR, "mini_test_feats.parquet")

Config.CACHE_STAGE1_OOF_PREDS = os.path.join(Config.WORKING_DIR, "mini_oof.parquet")
Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

# Reduce Model Complexity for Speed
Config.SVD_COMPONENTS = 10
Config.TFIDF_PARAMS["max_features"] = 500
Config.NUM_BOOST_ROUND = 10
Config.EARLY_STOPPING_ROUNDS = 5
Config.LGBM_PARAMS["num_leaves"] = 8
Config.LGBM_PARAMS["verbosity"] = -1

# Ensure directories exist
os.makedirs(Config.WORKING_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

# --- Import Library Modules ---
from library.utils import preprocess_text, count_inversions, kendall_tau_metric
from library.data_loader import load_data
from library.feature_engineering import MultiViewExtractor
from library.model_zoo import Stage1Ridge, Stage2LGBM
from library.pipeline import RankingPipeline

# Suppress Warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def demo_utils():
    print("\n=== Demonstrating Utils ===")

    # Test Preprocessing
    raw_text = "  Import  NUMPY as np   # comment "
    clean_text = preprocess_text(raw_text)
    print(f"Raw: '{raw_text}' -> Clean: '{clean_text}'")
    assert clean_text == "import numpy as np # comment", "Preprocessing failed"

    # Test Kendall Tau Logic
    # Ground Truth: A B C D (Sorted)
    # Prediction: B A D C (2 swaps needed: B-A, D-C)
    # n=4, pairs = 4*3 = 12. Formula: 1 - 4 * (swaps / (n*(n-1)))
    # Swaps = 2 (A<B but B before A; C<D but D before C)
    # Score = 1 - 4 * (2 / 12) = 1 - 2/3 = 0.3333

    df_gt = pd.DataFrame({"id": ["nb1"], "cell_order": ["A B C D"]})
    df_pred = pd.DataFrame({"id": ["nb1"], "cell_order": ["B A D C"]})

    score = kendall_tau_metric(df_gt, df_pred)
    print(f"Kendall Tau Score (Expected ~0.333): {score:.4f}")
    assert 0.33 < score < 0.34, "Kendall Tau calculation incorrect"
    print("Utils verification passed.")


def demo_data_loader():
    print("\n=== Demonstrating Data Loader ===")

    # Load Train (Debug size)
    # load_cached_data=False ensures we process from scratch for the demo
    df_train = load_data(split="train", load_cached_data=False)
    print(f"Loaded Training Data: {df_train.shape}")

    # Assertions
    required_cols = [
        "id",
        "cell_id",
        "cell_type",
        "source",
        "rank",
        "pct_rank",
        "ancestor_id",
    ]
    for col in required_cols:
        assert col in df_train.columns, f"Missing column {col} in train data"

    # Check rank normalization
    if not df_train.empty:
        assert df_train["pct_rank"].min() >= 0.0, "Rank < 0 found"
        assert df_train["pct_rank"].max() <= 1.0, "Rank > 1 found"

    # Load Val
    df_val = load_data(split="val", load_cached_data=False)
    print(f"Loaded Validation Data: {df_val.shape}")

    return df_train, df_val


def demo_feature_engineering(df_train, df_val):
    print("\n=== Demonstrating Feature Engineering ===")

    extractor = MultiViewExtractor()

    # Generate features for training
    # This fits the vectorizer internally
    feats_train = extractor.generate_features(
        df_train, split="train", load_cached_data=False
    )
    print(f"Generated Train Features: {feats_train.shape}")

    # Generate features for validation
    feats_val = extractor.generate_features(df_val, split="val", load_cached_data=False)
    print(f"Generated Val Features: {feats_val.shape}")

    # Assertions
    assert "id" in feats_train.columns, "Missing 'id' in features"
    assert "cell_id" in feats_train.columns, "Missing 'cell_id' in features"

    # Check for specific feature columns expected from MultiViewExtractor
    # e.g., lex_n1_score, lat_top5_mean_score
    feature_cols = [c for c in feats_train.columns if c not in ["id", "cell_id"]]
    assert len(feature_cols) > 0, "No features generated"
    print(f"Feature columns generated: {len(feature_cols)}")

    return feats_train, feats_val


def demo_modeling(df_train, df_val, feats_train, feats_val):
    print("\n=== Demonstrating Modeling (Stage 1 & 2) ===")

    # --- Stage 1: Ridge ---
    print("Training Stage 1 (Ridge)...")
    stage1 = Stage1Ridge()

    # Fit OOF on train
    oof_preds = stage1.fit_oof(df_train, load_cached_data=False)
    print(f"Stage 1 OOF Predictions: {oof_preds.shape}")

    # Predict on Val
    ridge_val_preds = stage1.predict(df_val)
    print(f"Stage 1 Val Predictions: {ridge_val_preds.shape}")

    assert "pred_ridge" in oof_preds.columns, "Missing pred_ridge in OOF"

    # --- Stage 2: LightGBM ---
    print("Training Stage 2 (LightGBM)...")
    stage2 = Stage2LGBM()

    # Train
    stage2.train(df_train, oof_preds, feats_train, df_val, ridge_val_preds, feats_val)

    # Predict on Val
    lgbm_val_preds = stage2.predict(df_val, ridge_val_preds, feats_val)
    print(f"Stage 2 Val Predictions: {lgbm_val_preds.shape}")

    assert "pred_rank" in lgbm_val_preds.columns, "Missing pred_rank in Stage 2 output"

    return stage1, stage2


def demo_pipeline_inference():
    print("\n=== Demonstrating Full Pipeline Inference ===")

    # We can use the RankingPipeline class to handle the test set and submission generation
    pipeline = RankingPipeline()

    # Since we already trained models in `demo_modeling` and saved them to the paths
    # defined in Config, the pipeline will pick them up.

    # Run inference on Test Set (using debug sampling defined in Config)
    pipeline.run_inference(load_cached_data=False)

    # Verify Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission generated at {Config.SUBMISSION_PATH}")
        print(sub_df.head())
        assert list(sub_df.columns) == [
            "id",
            "cell_order",
        ], "Invalid submission columns"
        assert len(sub_df) > 0, "Submission is empty"
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    set_seed(42)

    print("Starting Demo Script...")
    print(f"Working Directory: {Config.WORKING_DIR}")

    try:
        # 1. Utils
        demo_utils()

        # 2. Data Loading
        df_train, df_val = demo_data_loader()

        # 3. Feature Engineering
        feats_train, feats_val = demo_feature_engineering(df_train, df_val)

        # 4. Modeling
        demo_modeling(df_train, df_val, feats_train, feats_val)

        # 5. Pipeline / Inference
        demo_pipeline_inference()

        print("\nSUCCESS: All demonstrations completed without error.")

    except Exception as e:
        print(f"\nFAILURE: Script failed with error: {e}")
        # Raise to ensure non-zero exit code if run in CI/CD
        raise e
