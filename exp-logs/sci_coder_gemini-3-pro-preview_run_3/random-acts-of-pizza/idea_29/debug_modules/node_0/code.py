import os
import shutil
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

# Import from the provided library
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import load_and_clean_data, DataLoader
from library.feature_engineering import FeatureEngineer
from library.model_architecture import PentViewEnsemble
from library.workflow_manager import WorkflowManager


def main():
    # =========================================================================
    # 0. SETUP AND CONFIGURATION OVERRIDES
    # =========================================================================
    print(">>> Setting up configuration for fast demonstration...")

    # Set a fixed seed for reproducibility
    set_seed(42)

    # Override Config for speed
    # We use a temporary cache directory for this run
    Config.CACHE_DIR = "./working/demo_run_cache"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_OUTPUT_PATH = os.path.join(
        Config.SUBMISSION_DIR, "submission.csv"
    )

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce data size
    Config.DEBUG_SAMPLE_SIZE = 60  # Small sample for fast execution

    # Reduce CV folds
    Config.N_FOLDS = 2

    # Reduce Model Complexity
    Config.RF_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["early_stopping_rounds"] = (
        None  # Disable for very small data to avoid errors
    )

    # Reduce TF-IDF features for speed
    Config.TFIDF_PARAMS["max_features"] = 100

    # Setup logger
    logger = setup_logger("demo_script")
    logger.info("Configuration overrides applied.")

    # =========================================================================
    # 1. COMPONENT-LEVEL DEMONSTRATION
    # =========================================================================
    print("\n>>> Starting Component-Level Demonstration...")

    # --- A. Data Loading ---
    print("--- Testing DataLoader ---")
    # Load data using the wrapper function
    train_df, val_df, test_df = load_and_clean_data(load_cached_data=False)

    # Verify shapes
    assert (
        len(train_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Train size mismatch: {len(train_df)}"
    assert len(val_df) == Config.DEBUG_SAMPLE_SIZE, f"Val size mismatch: {len(val_df)}"
    assert (
        len(test_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Test size mismatch: {len(test_df)}"

    # Verify columns
    assert Config.TARGET_COL in train_df.columns, "Target column missing in train"
    assert (
        Config.TARGET_COL not in test_df.columns
    ), "Target column present in test (leakage)"
    print("DataLoader verification passed.")

    # --- B. Feature Engineering ---
    print("--- Testing FeatureEngineer ---")
    fe = FeatureEngineer()

    # Fit on training data
    fe.fit(train_df)

    # Transform train and val data
    # We use load_cache=False to force computation
    train_views = fe.transform(train_df, split_name="demo_train", load_cache=False)
    val_views = fe.transform(val_df, split_name="demo_val", load_cache=False)

    # Verify View Structure
    expected_views = ["lexical", "behavioral", "semantic", "metadata"]
    for view in expected_views:
        assert view in train_views, f"Missing view: {view}"
        assert view in val_views, f"Missing view: {view}"

    # Verify Dimensions
    # Metadata view should be (N_samples, N_features)
    n_meta_features = len(Config.NUMERICAL_FEATURES) + 1  # +1 for consistency feature
    assert train_views["metadata"].shape == (Config.DEBUG_SAMPLE_SIZE, n_meta_features)

    # Semantic view should be (N_samples, Embedding_Dim + N_meta_features)
    expected_sem_dim = Config.EMBEDDING_DIM + n_meta_features
    assert train_views["semantic"].shape == (Config.DEBUG_SAMPLE_SIZE, expected_sem_dim)

    # Check sparsity
    assert sp.issparse(train_views["lexical"]), "Lexical view should be sparse"
    assert not sp.issparse(train_views["metadata"]), "Metadata view should be dense"

    print("FeatureEngineer verification passed.")

    # --- C. Model Architecture ---
    print("--- Testing PentViewEnsemble ---")
    model = PentViewEnsemble()

    y_train = train_df[Config.TARGET_COL].values
    y_val = val_df[Config.TARGET_COL].values

    # Fit the ensemble
    # Note: This trains 5 base models and 1 meta-learner
    model.fit(train_views, y_train, val_views, y_val)

    assert model.fitted, "Model should be marked as fitted"

    # Predict
    preds = model.predict_proba(val_views)

    # Verify predictions
    assert len(preds) == Config.DEBUG_SAMPLE_SIZE
    assert np.all((preds >= 0) & (preds <= 1)), "Predictions must be probabilities"

    # Check AUC (just to ensure it runs, value doesn't matter on random/small data)
    try:
        auc = roc_auc_score(y_val, preds)
        print(f"Demo Model AUC: {auc:.4f}")
    except ValueError:
        print("Skipping AUC check (likely only one class in debug sample)")

    print("PentViewEnsemble verification passed.")

    # =========================================================================
    # 2. WORKFLOW-LEVEL DEMONSTRATION
    # =========================================================================
    print("\n>>> Starting Workflow-Level Demonstration...")

    # Instantiate WorkflowManager
    wm = WorkflowManager()

    # --- A. Train CV Bagging ---
    print("--- Running train_cv_bagging ---")
    # This runs the full K-Fold pipeline
    wm.train_cv_bagging(debug_size=Config.DEBUG_SAMPLE_SIZE, load_cached_data=False)

    # Verify artifacts were created
    models_dir = os.path.join(Config.CACHE_DIR, "models")
    for fold in range(Config.N_FOLDS):
        assert os.path.exists(
            os.path.join(models_dir, f"fe_fold_{fold}.joblib")
        ), f"FE artifact missing for fold {fold}"
        assert os.path.exists(
            os.path.join(models_dir, f"model_fold_{fold}.joblib")
        ), f"Model artifact missing for fold {fold}"

    print("Training workflow verification passed.")

    # --- B. Inference ---
    print("--- Running predict_bagged_inference ---")
    wm.predict_bagged_inference(
        debug_size=Config.DEBUG_SAMPLE_SIZE, load_cached_data=False
    )

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_OUTPUT_PATH), "Submission file not found"

    # Check submission content
    sub_df = pd.read_csv(Config.SUBMISSION_OUTPUT_PATH)
    assert (
        len(sub_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission size mismatch: {len(sub_df)}"
    assert list(sub_df.columns) == [
        "request_id",
        Config.TARGET_COL,
    ], "Submission columns mismatch"
    assert (
        sub_df[Config.TARGET_COL].between(0, 1).all()
    ), "Submission probabilities out of range"

    print("Inference workflow verification passed.")
    print(f"Submission generated at: {Config.SUBMISSION_OUTPUT_PATH}")

    # Cleanup (Optional, keeping artifacts for inspection is usually fine in working dir)
    # shutil.rmtree(Config.CACHE_DIR)

    print("\n>>> All demonstrations completed successfully.")


if __name__ == "__main__":
    main()
