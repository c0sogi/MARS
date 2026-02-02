import os
import shutil
import numpy as np
import pandas as pd
import warnings
import sys
from scipy import sparse

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Set random seeds
np.random.seed(42)

# Import library components
from library.config import Config
from library.data_loader import load_data
from library.feature_engineering import FeatureFactory, get_all_features
from library.model_zoo import (
    LexicalBagger,
    BehavioralBagger,
    SemanticBooster,
    SemanticBagger,
    MetadataAnchor,
)
from library.stacking_manager import StackingManager


def main():
    print("Starting Demo Script...")

    # -------------------------------------------------------------------------
    # 1. Setup Temporary Environment & Configuration Override
    # -------------------------------------------------------------------------
    print("\n[1/5] Setting up temporary environment and overriding Config...")

    # Define temporary directories
    DEMO_BASE = "./demo_run"
    DEMO_INPUT = os.path.join(DEMO_BASE, "input")
    DEMO_WORKING = os.path.join(DEMO_BASE, "working")
    DEMO_SUBMISSION = os.path.join(DEMO_BASE, "submission")

    # Clean up if exists
    if os.path.exists(DEMO_BASE):
        shutil.rmtree(DEMO_BASE)

    os.makedirs(DEMO_INPUT, exist_ok=True)
    os.makedirs(DEMO_WORKING, exist_ok=True)
    os.makedirs(DEMO_SUBMISSION, exist_ok=True)

    # Load a small subset of the original metadata to create demo datasets
    # We read directly from the original paths defined in Config before we patch them
    print("  Creating small dataset subsets (N=50)...")
    orig_train = pd.read_parquet(Config.TRAIN_DATA_PATH).head(50)
    orig_val = pd.read_parquet(Config.VAL_DATA_PATH).head(50)
    orig_test = pd.read_parquet(Config.TEST_DATA_PATH).head(50)

    # Save these subsets to the demo input directory
    demo_train_path = os.path.join(DEMO_INPUT, "train.parquet")
    demo_val_path = os.path.join(DEMO_INPUT, "val.parquet")
    demo_test_path = os.path.join(DEMO_INPUT, "test.parquet")

    orig_train.to_parquet(demo_train_path)
    orig_val.to_parquet(demo_val_path)
    orig_test.to_parquet(demo_test_path)

    # Monkey-patch Config to use demo paths and lightweight parameters
    Config.TRAIN_DATA_PATH = demo_train_path
    Config.VAL_DATA_PATH = demo_val_path
    Config.TEST_DATA_PATH = demo_test_path
    Config.WORKING_DIR = DEMO_WORKING
    Config.SUBMISSION_DIR = DEMO_SUBMISSION
    Config.SUBMISSION_PATH = os.path.join(DEMO_SUBMISSION, "submission.csv")

    # Update Cache Paths in Config to point to new working dir
    Config.CACHE_TRAIN_META = os.path.join(DEMO_WORKING, "X_train_meta.npy")
    Config.CACHE_VAL_META = os.path.join(DEMO_WORKING, "X_val_meta.npy")
    Config.CACHE_TEST_META = os.path.join(DEMO_WORKING, "X_test_meta.npy")
    Config.CACHE_TRAIN_TEXT_TFIDF = os.path.join(DEMO_WORKING, "X_train_lexical.npz")
    Config.CACHE_VAL_TEXT_TFIDF = os.path.join(DEMO_WORKING, "X_val_lexical.npz")
    Config.CACHE_TEST_TEXT_TFIDF = os.path.join(DEMO_WORKING, "X_test_lexical.npz")
    Config.CACHE_TRAIN_HIST_TFIDF = os.path.join(DEMO_WORKING, "X_train_behavioral.npz")
    Config.CACHE_VAL_HIST_TFIDF = os.path.join(DEMO_WORKING, "X_val_behavioral.npz")
    Config.CACHE_TEST_HIST_TFIDF = os.path.join(DEMO_WORKING, "X_test_behavioral.npz")
    Config.CACHE_TRAIN_EMBED = os.path.join(DEMO_WORKING, "X_train_semantic.npy")
    Config.CACHE_VAL_EMBED = os.path.join(DEMO_WORKING, "X_val_semantic.npy")
    Config.CACHE_TEST_EMBED = os.path.join(DEMO_WORKING, "X_test_semantic.npy")
    Config.CACHE_Y_TRAIN = os.path.join(DEMO_WORKING, "y_train.npy")
    Config.CACHE_Y_VAL = os.path.join(DEMO_WORKING, "y_val.npy")

    # Override Model Hyperparameters for Speed
    Config.RF_PARAMS = {
        "n_estimators": 5,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": 1,
        "verbose": 0,
    }
    Config.XGB_PARAMS = {
        "n_estimators": 5,
        "learning_rate": 0.1,
        "max_depth": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1.0,
        "random_state": 42,
        "n_jobs": 1,
        "verbosity": 0,
        "early_stopping_rounds": 5,
    }
    Config.LR_PARAMS = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": 42,
        "max_iter": 100,
    }
    Config.TFIDF_PARAMS["max_features"] = 50  # Reduce vocabulary size

    print("  Configuration patched successfully.")

    # -------------------------------------------------------------------------
    # 2. Data Loader Verification
    # -------------------------------------------------------------------------
    print("\n[2/5] Verifying Data Loader...")
    train_df, val_df, test_df = load_data()

    # Assertions
    assert len(train_df) == 50, f"Expected 50 train samples, got {len(train_df)}"
    assert len(val_df) == 50, f"Expected 50 val samples, got {len(val_df)}"
    assert len(test_df) == 50, f"Expected 50 test samples, got {len(test_df)}"

    # Check if text cleaning worked (no NaNs in text col)
    assert (
        not train_df[Config.TEXT_COL].isnull().any()
    ), "Found NaNs in train text column"
    assert isinstance(
        train_df[Config.HISTORY_COL].iloc[0], str
    ), "History column not serialized to string"

    print("  Data Loader verification passed.")

    # -------------------------------------------------------------------------
    # 3. Feature Engineering Verification
    # -------------------------------------------------------------------------
    print("\n[3/5] Verifying Feature Engineering...")

    # Use the Factory directly first to verify components
    factory = FeatureFactory()

    # Test Metadata
    factory.fit_metadata(train_df)
    X_meta = factory.transform_metadata(train_df)
    assert X_meta.shape == (50, len(Config.NUMERICAL_COLS)), "Metadata shape mismatch"
    assert not np.isnan(X_meta).any(), "Metadata contains NaNs after imputation"

    # Test Lexical
    factory.fit_lexical(train_df)
    X_lex = factory.transform_lexical(train_df)
    assert sparse.issparse(X_lex), "Lexical features should be sparse"
    assert X_lex.shape[0] == 50, "Lexical row count mismatch"

    # Test Semantic (Dense)
    # Note: This might take a few seconds to load the model
    X_sem = factory.create_semantic_view(train_df)
    assert X_sem.shape == (
        50,
        Config.EMBEDDING_DIM,
    ), f"Semantic shape mismatch, expected (50, {Config.EMBEDDING_DIM})"

    # Now verify the full pipeline function `get_all_features`
    # We force computation by setting load_cached_data=False (though cache is empty anyway)
    print("  Running get_all_features (computing and caching)...")
    data_dict = get_all_features(load_cached_data=False)

    required_keys = [
        "X_train_meta",
        "X_val_meta",
        "X_test_meta",
        "X_train_lexical",
        "X_train_behavioral",
        "X_train_semantic",
        "y_train",
        "y_val",
    ]
    for key in required_keys:
        assert key in data_dict, f"Missing key in feature dict: {key}"

    print("  Feature Engineering verification passed.")

    # -------------------------------------------------------------------------
    # 4. Model Zoo Verification
    # -------------------------------------------------------------------------
    print("\n[4/5] Verifying Model Zoo...")

    # Extract data for testing
    X_train_meta = data_dict["X_train_meta"]
    X_train_lex = data_dict["X_train_lexical"]
    X_train_sem = data_dict["X_train_semantic"]
    y_train = data_dict["y_train"]

    # 1. Test LexicalBagger (Sparse + Dense)
    print("  Testing LexicalBagger...")
    model_lex = LexicalBagger()
    model_lex.fit(X_train_lex, X_train_meta, y_train)
    preds = model_lex.predict_proba(X_train_lex, X_train_meta)
    assert preds.shape == (50,), "Prediction shape mismatch"
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Predictions out of probability range"

    # 2. Test SemanticBooster (Dense + Dense + Early Stopping)
    print("  Testing SemanticBooster...")
    model_sem_boost = SemanticBooster()
    # Create fake eval set
    eval_set = (X_train_sem, X_train_meta, y_train)
    model_sem_boost.fit(X_train_sem, X_train_meta, y_train, eval_set=eval_set)
    preds = model_sem_boost.predict_proba(X_train_sem, X_train_meta)
    assert preds.shape == (50,), "Prediction shape mismatch"

    # 3. Test MetadataAnchor (Meta only)
    print("  Testing MetadataAnchor...")
    model_meta = MetadataAnchor()
    model_meta.fit(None, X_train_meta, y_train)  # Should ignore first arg
    preds = model_meta.predict_proba(None, X_train_meta)
    assert preds.shape == (50,), "Prediction shape mismatch"

    print("  Model Zoo verification passed.")

    # -------------------------------------------------------------------------
    # 5. Stacking Manager Verification (End-to-End)
    # -------------------------------------------------------------------------
    print("\n[5/5] Verifying Stacking Manager (End-to-End)...")

    manager = StackingManager()

    # Run the full pipeline
    # This includes: Loading features -> CV (OOF) -> Meta Training -> Retraining -> Prediction -> Submission
    manager.run()

    # Verify Submission
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file not created"

    sub_df = pd.read_csv(submission_path)
    assert (
        len(sub_df) == 50
    ), f"Expected 50 predictions in submission, got {len(sub_df)}"
    assert Config.ID_COL in sub_df.columns, "ID column missing in submission"
    assert Config.TARGET_COL in sub_df.columns, "Target column missing in submission"
    assert not sub_df[Config.TARGET_COL].isnull().any(), "Submission contains NaNs"

    print("  Stacking Manager verification passed.")
    print("\nAll demonstrations completed successfully.")

    # Cleanup
    shutil.rmtree(DEMO_BASE)
    print("Temporary files cleaned up.")


if __name__ == "__main__":
    main()
