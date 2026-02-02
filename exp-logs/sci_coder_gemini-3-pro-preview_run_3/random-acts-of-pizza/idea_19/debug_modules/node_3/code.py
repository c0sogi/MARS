import os
import sys
import numpy as np
import pandas as pd
import warnings
from scipy import sparse

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
# We import config directly to monkey-patch parameters for speed
import library.config as config
from library.data_loader import load_datasets
from library.feature_engineering import FeaturePipeline
from library.model_definitions import ModelFactory
from library.ensemble_manager import EnsembleManager


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("\n[Step 1] Overriding configuration for fast execution...")

    # Reduce number of folds
    config.NUM_FOLDS = 2

    # Reduce Random Forest estimators
    config.RF_PARAMS["n_estimators"] = 5
    config.RF_DENSE_PARAMS["n_estimators"] = 5

    # Reduce XGBoost estimators
    config.XGB_PARAMS["n_estimators"] = 5

    # Reduce TF-IDF vocabulary size
    config.TEXT_TFIDF_PARAMS["max_features"] = 50
    config.SUBREDDIT_TFIDF_PARAMS["max_features"] = 20

    # Relax min_df for small sample size (Cite debug_lesson_12)
    config.TEXT_TFIDF_PARAMS["min_df"] = 1
    config.SUBREDDIT_TFIDF_PARAMS["min_df"] = 1

    # Set a small sample size for the demo
    SAMPLE_SIZE = 40

    print(f"  NUM_FOLDS set to {config.NUM_FOLDS}")
    print(f"  n_estimators set to 5")
    print(f"  Sample size set to {SAMPLE_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[Step 2] Loading and cleaning datasets...")

    # We force load_cached_data=False to demonstrate the cleaning logic
    X_train, y_train, X_val, y_val, X_test = load_datasets(
        load_cached_data=False, sample_size=SAMPLE_SIZE
    )

    # Validation
    assert (
        len(X_train) == SAMPLE_SIZE
    ), f"Expected {SAMPLE_SIZE} training samples, got {len(X_train)}"
    assert (
        len(X_val) == SAMPLE_SIZE
    ), f"Expected {SAMPLE_SIZE} validation samples, got {len(X_val)}"
    assert (
        len(X_test) == SAMPLE_SIZE
    ), f"Expected {SAMPLE_SIZE} test samples, got {len(X_test)}"
    assert (
        config.TARGET_COL not in X_train.columns
    ), "Target column should be removed from X_train"

    print("  Data loaded successfully.")
    print(f"  Train shape: {X_train.shape}")

    # -------------------------------------------------------------------------
    # 3. Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[Step 3] Running Feature Pipeline...")

    pipeline = FeaturePipeline()

    # Fit and Transform Train
    # We use a specific prefix to avoid overwriting real work if any
    train_feats = pipeline.fit_transform(
        X_train, prefix="demo_train", load_cached_data=False
    )

    # Validation of Feature Dictionary
    expected_keys = ["meta", "text_sparse", "text_dense", "beh_sparse", "beh_dense"]
    for key in expected_keys:
        assert key in train_feats, f"Missing key '{key}' in feature dictionary"

    # Check shapes
    n_samples = len(X_train)
    assert train_feats["meta"].shape[0] == n_samples
    assert train_feats["text_sparse"].shape[0] == n_samples
    assert train_feats["text_dense"].shape[0] == n_samples

    print("  Feature pipeline fitted and transformed training data.")
    print(f"  Meta features shape: {train_feats['meta'].shape}")
    print(f"  Text sparse shape: {train_feats['text_sparse'].shape}")

    # -------------------------------------------------------------------------
    # 4. Model Definitions & Factory
    # -------------------------------------------------------------------------
    print("\n[Step 4] Verifying Model Factory...")

    models = ModelFactory.get_level_1_models()

    # Check if we have the expected number of base learners (7)
    assert len(models) == 7, f"Expected 7 base models, found {len(models)}"

    # Verify parameter override took effect
    rf_model = models[ModelFactory.KEY_TEXT_SPARSE_RF]
    assert rf_model.n_estimators == 5, "Config override for RF n_estimators failed"

    # Test Feature Preparation Logic
    # Case A: Sparse RF (should return sparse matrix)
    X_sparse_rf = ModelFactory.prepare_features(
        train_feats, ModelFactory.KEY_TEXT_SPARSE_RF
    )
    assert sparse.issparse(X_sparse_rf), "Expected sparse matrix for Sparse RF model"
    assert X_sparse_rf.shape[0] == n_samples

    # Case B: Dense XGB (should return numpy array)
    X_dense_xgb = ModelFactory.prepare_features(
        train_feats, ModelFactory.KEY_TEXT_DENSE_XGB
    )
    assert isinstance(
        X_dense_xgb, np.ndarray
    ), "Expected numpy array for Dense XGB model"
    assert X_dense_xgb.shape[0] == n_samples

    print("  Model Factory instantiated models correctly.")
    print("  Feature preparation logic verified.")

    # -------------------------------------------------------------------------
    # 5. Ensemble Manager (Training & Prediction)
    # -------------------------------------------------------------------------
    print("\n[Step 5] Running Ensemble Manager (Train -> OOF -> Meta -> Predict)...")

    manager = EnsembleManager()

    # Run the full workflow
    # This calls:
    # 1. generate_oof_predictions (CV)
    # 2. train_meta_learner
    # 3. retrain_final_models (Validation Guided)
    # 4. predict_test
    manager.train_and_predict(X_train, y_train, X_val, y_val, X_test, pipeline)

    # Validation of OOF
    assert manager.oof_predictions is not None, "OOF predictions were not generated"
    assert manager.oof_predictions.shape == (
        n_samples,
        7,
    ), "OOF predictions shape mismatch"

    # Validation of Submission File
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created"

    # Load submission and check content
    submission_df = pd.read_csv(config.SUBMISSION_PATH)
    assert len(submission_df) == SAMPLE_SIZE, "Submission row count mismatch"
    assert config.ID_COL in submission_df.columns, "Submission missing ID column"
    assert (
        config.TARGET_COL in submission_df.columns
    ), "Submission missing Target column"

    # Check probability range
    probs = submission_df[config.TARGET_COL]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    print("  Ensemble workflow completed successfully.")
    print(f"  Submission saved to: {config.SUBMISSION_PATH}")
    print(f"  First 3 predictions:\n{submission_df.head(3)}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Ensure reproducibility
    np.random.seed(42)
    run_demo()
