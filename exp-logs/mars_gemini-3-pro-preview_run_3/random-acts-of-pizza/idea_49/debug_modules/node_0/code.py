import os
import sys
import shutil
import numpy as np
import pandas as pd
import joblib
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.data_loader import DataLoader
from library.feature_engineering import FeaturePipeline
from library.model_factory import ModelFactory
from library.training_engine import TrainingEngine
from library.inference_engine import InferenceEngine


def setup_demo_environment():
    """
    Sets up a lightweight environment for the demo by creating data subsets
    and overriding Config parameters for speed.
    """
    print("--- Setting up Demo Environment ---")

    # Define paths
    demo_dir = "./working/demo_run"
    demo_cache_dir = os.path.join(demo_dir, "cache")
    demo_model_dir = os.path.join(demo_dir, "models")
    demo_sub_dir = os.path.join(demo_dir, "submission")

    # Clean previous run if exists
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)

    os.makedirs(demo_dir, exist_ok=True)
    os.makedirs(demo_cache_dir, exist_ok=True)
    os.makedirs(demo_model_dir, exist_ok=True)
    os.makedirs(demo_sub_dir, exist_ok=True)

    # 1. Create Data Subsets
    print("Creating data subsets for rapid execution...")
    # Read original metadata
    train_full = pd.read_parquet(Config.TRAIN_DATA_PATH)
    val_full = pd.read_parquet(Config.VAL_DATA_PATH)
    test_full = pd.read_parquet(Config.TEST_DATA_PATH)

    # Sample small subsets (N=20)
    N_SAMPLES = 20
    train_subset = train_full.head(N_SAMPLES).copy()
    val_subset = val_full.head(N_SAMPLES).copy()
    test_subset = test_full.head(N_SAMPLES).copy()

    # Save subsets
    train_subset_path = os.path.join(demo_cache_dir, "demo_train.parquet")
    val_subset_path = os.path.join(demo_cache_dir, "demo_val.parquet")
    test_subset_path = os.path.join(demo_cache_dir, "demo_test.parquet")

    train_subset.to_parquet(train_subset_path, index=False)
    val_subset.to_parquet(val_subset_path, index=False)
    test_subset.to_parquet(test_subset_path, index=False)

    # 2. Override Config
    print("Overriding Config parameters for speed...")

    # Paths
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = demo_cache_dir
    Config.MODEL_DIR = demo_model_dir
    Config.SUBMISSION_DIR = demo_sub_dir
    Config.TRAIN_DATA_PATH = train_subset_path
    Config.VAL_DATA_PATH = val_subset_path
    Config.TEST_DATA_PATH = test_subset_path
    Config.SUBMISSION_PATH = os.path.join(demo_sub_dir, "submission.csv")

    # Model Hyperparameters (Speed Optimization)
    # Reduce estimators to minimum
    Config.PARAMS_LEXICAL_BAGGER["n_estimators"] = 2
    Config.PARAMS_COMMUNITY_BAGGER["n_estimators"] = 2
    Config.PARAMS_SEMANTIC_BAGGER["n_estimators"] = 2

    # XGBoost Speedup
    Config.PARAMS_SEMANTIC_BOOSTER["n_estimators"] = 2
    Config.PARAMS_SEMANTIC_BOOSTER["max_depth"] = 2
    Config.PARAMS_SEMANTIC_BOOSTER["early_stopping_rounds"] = None

    # LightGBM Speedup
    Config.PARAMS_TEMPORAL_BOOSTER["n_estimators"] = 2
    Config.PARAMS_TEMPORAL_BOOSTER["num_leaves"] = 4

    # Anchor Speedup
    Config.PARAMS_METADATA_ANCHOR["max_iter"] = 5

    # Feature Extraction Speedup
    Config.PARAMS_TFIDF_LEXICAL["max_features"] = 50
    Config.PARAMS_TFIDF_COMMUNITY["max_features"] = 20

    # Embedding Model (Use a smaller one if possible, but we stick to config default
    # as we can't easily change the installed package cache.
    # With N=20, inference will be fast anyway.)

    return N_SAMPLES


def test_data_loader():
    print("\n--- Testing DataLoader ---")
    loader = DataLoader()

    # Test raw loading
    tr, va, te = loader.load_raw_data()
    print(f"Raw Data Loaded: Train={tr.shape}, Val={va.shape}, Test={te.shape}")

    assert len(tr) == 20, "Train subset size mismatch"
    assert len(va) == 20, "Val subset size mismatch"

    # Test processing (this triggers text combination and subreddit joining)
    # We force load_cached_data=False to test the logic
    tr_proc, va_proc, te_proc = loader.get_processed_data(load_cached_data=False)

    assert (
        "text_combined" in tr_proc.columns
    ), "text_combined missing from processed train"
    assert (
        "subreddit_text" in tr_proc.columns
    ), "subreddit_text missing from processed train"
    print("DataLoader verification successful.")


def test_feature_pipeline():
    print("\n--- Testing FeaturePipeline ---")
    # Force regeneration of features
    pipeline = FeaturePipeline(load_cached_data=False)

    features = pipeline.get_all_features()

    # Check keys
    expected_keys = [
        "X_train_lexical",
        "X_val_lexical",
        "X_test_lexical",
        "X_train_community",
        "X_val_community",
        "X_test_community",
        "X_train_semantic",
        "X_val_semantic",
        "X_test_semantic",
        "X_train_meta",
        "X_val_meta",
        "X_test_meta",
        "y_train",
        "y_val",
        "test_ids",
    ]

    for key in expected_keys:
        assert key in features, f"Missing feature key: {key}"

    # Check dimensions
    # Lexical (Sparse)
    assert features["X_train_lexical"].shape[0] == 20
    assert (
        features["X_train_lexical"].shape[1]
        <= Config.PARAMS_TFIDF_LEXICAL["max_features"]
    )

    # Semantic (Dense) - Embedding dim is 384 for all-MiniLM-L6-v2
    assert features["X_train_semantic"].shape == (20, 384)

    # Metadata (Dense) - 10 features
    assert features["X_train_meta"].shape == (20, 10)

    print("FeaturePipeline verification successful.")


def test_model_factory():
    print("\n--- Testing ModelFactory ---")

    # Test instantiation of a few types
    rf = ModelFactory.get_base_learner("lexical_bagger")
    assert "RandomForestClassifier" in str(type(rf))
    assert rf.n_estimators == 2

    xgb = ModelFactory.get_base_learner("semantic_booster")
    assert "XGBClassifier" in str(type(xgb))
    assert xgb.n_estimators == 2

    lgbm = ModelFactory.get_base_learner("temporal_booster")
    assert "LGBMClassifier" in str(type(lgbm))

    meta = ModelFactory.get_meta_learner()
    assert "LogisticRegression" in str(type(meta))

    print("ModelFactory verification successful.")


def test_training_engine():
    print("\n--- Testing TrainingEngine ---")
    # We use 2 folds for speed
    engine = TrainingEngine(load_cached_data=True)
    engine.run(n_folds=2)

    # Verify outputs
    expected_models = [
        "lexical_bagger_fold_0.joblib",
        "lexical_bagger_fold_1.joblib",
        "meta_learner.joblib",
    ]

    for model_name in expected_models:
        path = os.path.join(Config.MODEL_DIR, model_name)
        assert os.path.exists(path), f"Model file missing: {model_name}"

    print("TrainingEngine execution successful.")


def test_inference_engine():
    print("\n--- Testing InferenceEngine ---")
    engine = InferenceEngine(load_cached_data=True)
    engine.run_inference(n_folds=2)

    # Verify submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file missing"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert df_sub.shape == (20, 2), f"Submission shape mismatch: {df_sub.shape}"
    assert Config.ID_COL in df_sub.columns
    assert Config.TARGET_COL in df_sub.columns

    # Check probability range
    probs = df_sub[Config.TARGET_COL]
    assert probs.min() >= 0.0 and probs.max() <= 1.0, "Probabilities out of range"

    print("InferenceEngine execution successful.")


if __name__ == "__main__":
    # Set global seed
    np.random.seed(42)

    try:
        # 1. Setup
        setup_demo_environment()

        # 2. Test Components
        test_data_loader()
        test_feature_pipeline()
        test_model_factory()

        # 3. Run Pipeline
        test_training_engine()
        test_inference_engine()

        print("\nAll demonstrations completed successfully!")

    except AssertionError as e:
        print(f"\n[FAILURE] Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAILURE] An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
