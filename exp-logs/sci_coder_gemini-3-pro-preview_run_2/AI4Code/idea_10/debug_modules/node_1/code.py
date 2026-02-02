import os
import sys
import random
import numpy as np
import pandas as pd
import warnings

# Ensure the current directory is in the path to import the library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.data_manager import DataManager
from library.feature_engine import FeatureEngineer
from library.modeling import StackedRanker


def set_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def override_config_for_demo():
    """
    Modifies the Config class attributes at runtime to ensure the demo
    runs quickly on a small subset of data.
    """
    print("Overriding configuration for fast demonstration...")

    # Enable Debug mode to sample data
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 50  # Use only 50 notebooks for the demo

    # Reduce Cross-Validation folds
    Config.N_FOLDS = 2

    # Reduce SVD components to fit the small data sample size
    Config.SVD_PARAMS["n_components"] = 10
    Config.SVD_PARAMS["n_iter"] = 2

    # Reduce LightGBM complexity for speed
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["num_leaves"] = 8

    # Ensure silent execution
    Config.LGBM_PARAMS["verbose"] = -1

    # Use a specific cache directory for this demo run to avoid conflicts
    Config.CACHE_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_submission"

    # Update paths dependent on directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    Config.TRAIN_FEATS_PATH = os.path.join(
        Config.CACHE_DIR, "debug_train_processed.parquet"
    )
    Config.VAL_FEATS_PATH = os.path.join(
        Config.CACHE_DIR, "debug_val_processed.parquet"
    )
    Config.TEST_FEATS_PATH = os.path.join(
        Config.CACHE_DIR, "debug_test_processed.parquet"
    )

    Config.TFIDF_MODEL_PATH = os.path.join(Config.CACHE_DIR, "tfidf_vectorizer.joblib")
    Config.SVD_MODEL_PATH = os.path.join(Config.CACHE_DIR, "svd_model.joblib")
    Config.RIDGE_MODEL_PATH = os.path.join(Config.CACHE_DIR, "ridge_model.joblib")
    Config.LGBM_MODEL_PATH = os.path.join(Config.CACHE_DIR, "lgbm_model.txt")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")


def main():
    # 1. Setup
    set_seeds(Config.RANDOM_STATE)
    override_config_for_demo()

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    print("\n=== 1. Data Loading ===")
    dm = DataManager()

    # Load Train, Val, and Test data
    # We force load_cached_data=False to demonstrate the processing logic
    df_train = dm.load_data("train", load_cached_data=False)
    df_val = dm.load_data("val", load_cached_data=False)
    df_test = dm.load_data("test", load_cached_data=False)

    # Validation: Check data loaded correctly
    assert not df_train.empty, "Training dataframe is empty."
    assert not df_val.empty, "Validation dataframe is empty."
    assert not df_test.empty, "Test dataframe is empty."

    # Check if we respected the debug sample limit (approximate, as it samples notebooks, not cells)
    print(f"Loaded {df_train['id'].nunique()} training notebooks.")
    print(f"Loaded {df_val['id'].nunique()} validation notebooks.")
    print(f"Loaded {df_test['id'].nunique()} test notebooks.")

    print("\n=== 2. Feature Engineering ===")
    fe = FeatureEngineer()

    # Generate features for all splits
    # Train split fits the vectorizers
    train_feats = fe.generate_features(df_train, split="train", load_cached_data=False)

    # Val and Test splits use the fitted vectorizers
    val_feats = fe.generate_features(df_val, split="val", load_cached_data=False)
    test_feats = fe.generate_features(df_test, split="test", load_cached_data=False)

    # Validation: Check feature generation
    expected_cols = ["lexical_max_sim", "latent_max_sim", "notebook_code_count"]
    for col in expected_cols:
        assert col in train_feats.columns, f"Missing feature column: {col}"

    assert len(train_feats) > 0, "Train features dataframe is empty."
    assert len(val_feats) > 0, "Val features dataframe is empty."
    assert len(test_feats) > 0, "Test features dataframe is empty."

    print(f"Generated {train_feats.shape[1]} features for training set.")

    print("\n=== 3. Modeling: Stage 1 (Ridge Regression) ===")
    ranker = StackedRanker()

    # Train Ridge OOF
    oof_preds = ranker.train_stage1_ridge_oof(df_train, load_cached_data=False)

    # Validation: Check OOF predictions
    assert (
        "ridge_pred" in oof_preds.columns
    ), "OOF predictions missing 'ridge_pred' column."
    assert len(oof_preds) == len(train_feats), "OOF predictions length mismatch."
    assert os.path.exists(Config.RIDGE_MODEL_PATH), "Ridge model file was not saved."

    print("Stage 1 OOF predictions generated successfully.")

    print("\n=== 4. Modeling: Stage 2 (LightGBM) ===")
    # Train LightGBM using features + OOF predictions
    lgbm_model = ranker.train_stage2_lgbm(train_feats, oof_preds, val_feats, df_val)

    # Validation: Check model file
    assert os.path.exists(Config.LGBM_MODEL_PATH), "LightGBM model file was not saved."
    print("Stage 2 model trained and saved.")

    print("\n=== 5. Inference & Submission ===")
    # Run full inference pipeline
    ranker.predict(df_test, test_feats)

    # Validation: Check submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(df_sub.columns) == [
        "id",
        "cell_order",
    ], "Submission columns are incorrect."
    assert (
        len(df_sub) == df_test["id"].nunique()
    ), "Submission row count does not match test set notebooks."

    # Check content format (space delimited string)
    sample_order = df_sub.iloc[0]["cell_order"]
    assert isinstance(sample_order, str), "cell_order is not a string."
    assert len(sample_order.split()) > 0, "cell_order is empty."

    print(f"Submission generated at {Config.SUBMISSION_PATH}")
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()
