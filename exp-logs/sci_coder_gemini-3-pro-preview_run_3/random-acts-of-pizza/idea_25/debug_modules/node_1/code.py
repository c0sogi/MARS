import os
import shutil
import numpy as np
import pandas as pd
import warnings
import sys

# Ensure library can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import clean_text, serialize_subreddits
from library.features import FeatureEngine
from library.model_factory import ModelFactory
from library.stacking_manager import StackingManager


def run_demo():
    print("=== Starting Demonstration of Pizza Request Prediction Pipeline ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed and Demo
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Use a separate cache directory for this demo to ensure we calculate features from scratch
    # and verify the logic, rather than loading potentially stale cache.
    demo_cache_dir = "./working/demo_run_cache/"
    if os.path.exists(demo_cache_dir):
        shutil.rmtree(demo_cache_dir)
    os.makedirs(demo_cache_dir)

    # Modify Config singleton directly to optimize for speed
    Config.CACHE_DIR = demo_cache_dir
    Config.DEBUG_SAMPLE_SIZE = 60  # Small sample for fast execution (approx. 1 min)
    Config.N_FOLDS = 2  # Minimal folds for Cross-Validation
    Config.PCA_COMPONENTS = 10  # Reduced dimensionality for small sample size

    # Reduce ensemble sizes for speed (default is 500-2000)
    Config.RF_LEXICAL_PARAMS["n_estimators"] = 10
    Config.RF_BEHAVIORAL_PARAMS["n_estimators"] = 10
    Config.RF_SEMANTIC_PARAMS["n_estimators"] = 10
    Config.XGB_SEMANTIC_PARAMS["n_estimators"] = 10
    Config.KNN_MANIFOLD_PARAMS["n_neighbors"] = 5

    # Ensure submission directory exists
    Config.SUBMISSION_DIR = "./working/demo_submission/"
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    print(f"    Cache Dir: {Config.CACHE_DIR}")
    print(f"    Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"    Folds: {Config.N_FOLDS}")

    # -------------------------------------------------------------------------
    # 2. Testing Utility Functions
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test clean_text: Should remove "EDIT:" blocks
    raw_edit = "I need pizza. \nEDIT: Got it!"
    cleaned = clean_text(raw_edit)
    assert "EDIT" not in cleaned, "clean_text failed to remove edit block"
    assert cleaned == "I need pizza.", f"clean_text output unexpected: {cleaned}"
    print("    clean_text: OK")

    # Test serialize_subreddits: Should join list into string
    subs_list = ["gaming", "funny", None, ""]
    serialized = serialize_subreddits(subs_list)
    assert (
        serialized == "gaming funny"
    ), f"serialize_subreddits output unexpected: {serialized}"
    print("    serialize_subreddits: OK")

    # -------------------------------------------------------------------------
    # 3. Testing Feature Engineering (Component Level)
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Feature Engine...")

    # Instantiate engine
    fe = FeatureEngine()

    # Run fit_transform on train split
    # Note: This will create cache files in our demo_cache_dir
    print("    Running fit_transform on Train...")
    feats_train, y_train = fe.fit_transform(
        split="train",
        load_cached_data=False,
        debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    )

    # Assertions to verify feature generation
    assert y_train is not None
    assert len(y_train) == Config.DEBUG_SAMPLE_SIZE
    assert "metadata" in feats_train
    assert "lexical" in feats_train
    assert "semantic" in feats_train
    assert feats_train["metadata"].shape[0] == Config.DEBUG_SAMPLE_SIZE
    print("    Train features generated successfully.")

    # Run transform on val split
    print("    Running transform on Val...")
    feats_val, y_val = fe.transform(
        split="val", load_cached_data=False, debug_sample_size=Config.DEBUG_SAMPLE_SIZE
    )
    assert feats_val["metadata"].shape[0] == Config.DEBUG_SAMPLE_SIZE
    print("    Val features generated successfully.")

    # -------------------------------------------------------------------------
    # 4. Testing Model Factory
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Factory...")
    base_models = ModelFactory.get_base_models()
    meta_model = ModelFactory.get_meta_model()

    expected_models = [
        "LexicalBagger",
        "BehavioralBagger",
        "SemanticBooster",
        "SemanticBagger",
        "ManifoldNeighbor",
        "ContextualAnchor",
    ]

    for m in expected_models:
        assert m in base_models, f"Missing base model: {m}"

    assert meta_model is not None
    print(f"    Factory returned {len(base_models)} base models and 1 meta model.")

    # -------------------------------------------------------------------------
    # 5. Testing Stacking Manager (Integration Level)
    # -------------------------------------------------------------------------
    print("\n[5] Running Full Stacking Pipeline (StackingManager)...")

    # Instantiate Manager
    manager = StackingManager()

    # Execute full pipeline: Features -> L1 CV -> L2 Train -> Retrain -> Predict
    # We use the debug_sample_size to keep it fast
    manager.train_and_predict(debug_sample_size=Config.DEBUG_SAMPLE_SIZE)

    # -------------------------------------------------------------------------
    # 6. Final Verification
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Submission...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    print(f"    Submission Shape: {df_sub.shape}")
    print(f"    Columns: {df_sub.columns.tolist()}")

    # Check rows
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission row count mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(df_sub)}"

    # Check columns
    assert Config.ID_COL in df_sub.columns
    assert Config.TARGET_COL in df_sub.columns

    # Check values (probabilities)
    preds = df_sub[Config.TARGET_COL]
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions out of probability range [0, 1]"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    # Fix random seeds for reproducibility
    np.random.seed(42)

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    run_demo()
