import os
import sys
import shutil
import numpy as np
import pandas as pd
import warnings

# Add current directory to path
sys.path.append(os.getcwd())

# Import library components
import library.config as config
from library.data_loader import load_data
from library.feature_pipeline import FeaturePipeline
from library.stacking_engine import TriViewStackingEnsemble
from library.utils import set_seed, save_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Pipeline Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Optimization (Monkey-Patching for Speed)
    # -------------------------------------------------------------------------
    print("[Setup] Optimizing configuration for fast demonstration...")

    # Reduce dataset size
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 samples

    # Reduce Cross-Validation folds
    config.N_FOLDS = 2

    # Reduce Feature Dimensionality
    config.TFIDF_MAX_FEATURES = 100
    config.SUBREDDIT_TFIDF_MAX_FEATURES = 50
    config.SVD_COMPONENTS = 5

    # Reduce Model Complexity (Estimators)
    # Note: We modify the dictionaries in-place so the imported modules see the changes
    config.L1_LEXICAL_PARAMS["n_estimators"] = 5
    config.L1_LEXICAL_PARAMS["n_jobs"] = 1

    config.L1_SEMANTIC_PARAMS["n_estimators"] = 5
    config.L1_SEMANTIC_PARAMS["max_depth"] = 5
    config.L1_SEMANTIC_PARAMS["n_jobs"] = 1

    config.L1_COMMUNITY_PARAMS["n_estimators"] = 5
    config.L1_COMMUNITY_PARAMS["max_depth"] = 3
    config.L1_COMMUNITY_PARAMS["n_jobs"] = 1

    config.L2_META_PARAMS["n_jobs"] = 1

    # Clean working directory to ensure fresh run
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    set_seed(42)

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading Data (Debug Mode)...")
    # load_cached_data=False forces loading from metadata and processing
    X_train, y_train, X_val, y_val, X_test, test_ids = load_data(
        load_cached_data=False, debug=True
    )

    # Validation
    print(f"  Train shape: {X_train.shape}")
    print(f"  Val shape:   {X_val.shape}")
    print(f"  Test shape:  {X_test.shape}")

    assert (
        len(X_train) == config.DEBUG_SAMPLE_SIZE
    ), "Train set size does not match debug sample size"
    assert (
        len(X_val) == config.DEBUG_SAMPLE_SIZE
    ), "Val set size does not match debug sample size"
    assert (
        not X_train.isnull().values.any()
    ), "Found missing values in X_train after cleaning"
    assert config.TARGET_COL not in X_train.columns, "Target column leaked into X_train"

    # -------------------------------------------------------------------------
    # 3. Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[Step 2] Running Feature Pipeline...")
    pipeline = FeaturePipeline()

    # Fit on training data
    pipeline.fit(X_train)

    # Transform all splits
    train_feats = pipeline.transform(X_train, "Train")
    val_feats = pipeline.transform(X_val, "Val")
    test_feats = pipeline.transform(X_test, "Test")

    # Validation of Feature Structure
    required_keys = ["lexical", "semantic", "community"]
    for key in required_keys:
        assert key in train_feats, f"Missing feature view: {key}"
        assert train_feats[key].shape[0] == len(
            X_train
        ), f"Shape mismatch in {key} features"

    print("  Feature dictionaries created successfully.")
    print(f"  Lexical feature shape: {train_feats['lexical'].shape}")
    print(f"  Semantic feature shape: {train_feats['semantic'].shape}")

    # -------------------------------------------------------------------------
    # 4. Model Stacking (Training)
    # -------------------------------------------------------------------------
    print("\n[Step 3] Training Stacking Ensemble...")
    ensemble = TriViewStackingEnsemble()

    # Fit the ensemble (Level 1 CV + Level 2 Meta training + Refitting L1)
    ensemble.fit(train_feats, y_train)

    print("  Ensemble fitted successfully.")

    # -------------------------------------------------------------------------
    # 5. Prediction & Evaluation
    # -------------------------------------------------------------------------
    print("\n[Step 4] Generating Predictions...")

    # Predict on Validation set
    val_probs = ensemble.predict_proba(val_feats)

    # Validate Probabilities
    assert len(val_probs) == len(y_val), "Prediction length mismatch"
    assert np.all(
        (val_probs >= 0) & (val_probs <= 1)
    ), "Probabilities must be between 0 and 1"

    print(f"  Validation Predictions (First 5): {val_probs[:5]}")

    # Predict on Test set
    test_probs = ensemble.predict_proba(test_feats)

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[Step 5] Saving Submission...")
    save_submission(test_ids, test_probs, path=config.SUBMISSION_PATH)

    # Verify file creation
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created"

    # Verify content format
    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    assert df_sub.shape == (len(test_ids), 2), "Submission CSV has incorrect shape"
    assert list(df_sub.columns) == [
        "request_id",
        "requester_received_pizza",
    ], "Incorrect column headers"
    assert (
        df_sub["requester_received_pizza"].between(0, 1).all()
    ), "Submission values out of range"

    print(f"  Submission saved to {config.SUBMISSION_PATH}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
