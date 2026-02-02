import os
import sys
import numpy as np
import pandas as pd
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import set_seed, save_submission
from library.data_factory import prepare_datasets
from library.feature_pipeline import FeatureManager
from library.stacking_engine import HybridEnsembleTrainer


def main():
    print("=== Starting Demonstration of Pizza Success Prediction Pipeline ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demonstration
    # -------------------------------------------------------------------------
    print("Configuring pipeline for fast demonstration run...")

    # Set a specific working directory for this demo
    DEMO_DIR = "./working/demo_execution"
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.MODEL_DIR = os.path.join(DEMO_DIR, "models")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories manually since Config code ran on import
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce computational load
    Config.N_FOLDS = 2  # Use 2 folds instead of 5
    Config.EARLY_STOPPING_ROUNDS = 10

    # Patch model parameters for speed (Low estimators, low iterations)
    for key, conf in Config.MODEL_CONFIGS.items():
        params = conf["params"]
        if "n_estimators" in params:
            params["n_estimators"] = 10
        if "max_iter" in params:
            params["max_iter"] = 20
        if "n_jobs" in params:
            params["n_jobs"] = 1  # Reduce thread contention for small data

        # Reduce complexity for tree models
        if "num_leaves" in params:
            params["num_leaves"] = 8
        if "max_depth" in params:
            params["max_depth"] = 3

    # Patch Meta Learner
    Config.META_LEARNER_PARAMS["max_iter"] = 20

    # Set Seed
    set_seed(Config.RANDOM_STATE)

    # -------------------------------------------------------------------------
    # 2. Data Preparation
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading and Preparing Datasets...")
    # Use debug=True to sample only 100 rows for speed
    union_train_df, test_df = prepare_datasets(
        load_cached_data=False,  # Force reload to verify logic
        debug=True,
        debug_size=100,
    )

    # Validation
    assert (
        len(union_train_df) == 100
    ), f"Expected 100 training samples, got {len(union_train_df)}"
    assert len(test_df) == 100, f"Expected 100 test samples, got {len(test_df)}"
    assert (
        Config.TARGET_COL in union_train_df.columns
    ), "Target column missing from training data"
    print("Data loaded successfully.")

    # -------------------------------------------------------------------------
    # 3. Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[Step 2] Generating Features...")
    feature_manager = FeatureManager()

    # Process features (this handles TF-IDF, Embeddings, Metadata scaling)
    # We disable cache loading to force computation
    feature_data = feature_manager.process_features(
        union_train_df, test_df, load_cached_data=False
    )

    # Validation of Feature Dictionary
    expected_keys = [
        "lexical_sparse",
        "community_sparse",
        "semantic_dense",
        "metadata_only",
    ]
    for key in expected_keys:
        assert key in feature_data, f"Missing feature set: {key}"
        train_feat, test_feat = feature_data[key]
        assert train_feat.shape[0] == 100, f"Train feature {key} row count mismatch"
        assert test_feat.shape[0] == 100, f"Test feature {key} row count mismatch"

    print("Features generated and validated.")

    # -------------------------------------------------------------------------
    # 4. Stacking Engine: Training
    # -------------------------------------------------------------------------
    print("\n[Step 3] Initializing Hybrid Ensemble Trainer...")
    trainer = HybridEnsembleTrainer(feature_data, union_train_df, test_df)

    # A. Level 1 Training (CV & OOF)
    print("Running Level 1 Training (Base Learners)...")
    trainer.train_level_1()

    # Validate OOF generation
    oof_path = os.path.join(Config.WORKING_DIR, "oof_predictions.csv")
    assert os.path.exists(oof_path), "OOF predictions file was not created"
    oof_df = pd.read_csv(oof_path)
    assert len(oof_df) == 100, "OOF DataFrame size mismatch"
    # Check for NaNs in OOF (indicates missed folds)
    assert not oof_df.isnull().any().any(), "NaNs found in OOF predictions"

    # B. Retrain Stable Models
    print("Retraining Stable Models on full data...")
    trainer.retrain_stable_models()

    # C. Meta Learner Training
    print("Training Level 2 Meta-Learner...")
    trainer.train_meta_learner()

    # Validate Model Persistence
    assert os.path.exists(
        os.path.join(Config.MODEL_DIR, "meta_learner.joblib")
    ), "Meta learner not saved"
    # Check a random base learner
    assert os.path.exists(
        os.path.join(Config.MODEL_DIR, "lexical_bagger_fold_0.joblib")
    ), "Fold model not saved"
    assert os.path.exists(
        os.path.join(Config.MODEL_DIR, "lexical_bagger.joblib")
    ), "Retrained stable model not saved"

    # -------------------------------------------------------------------------
    # 5. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n[Step 4] Generating Final Predictions...")
    final_preds = trainer.predict()

    # Validation of Predictions
    assert len(final_preds) == 100, "Prediction length mismatch"
    assert np.all(
        (final_preds >= 0) & (final_preds <= 1)
    ), "Predictions out of probability range [0, 1]"

    print("Saving Submission...")
    save_submission(test_df[Config.ID_COL], final_preds, filename="submission.csv")

    # Validate Submission File
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file not found"

    sub_df = pd.read_csv(submission_path)
    assert list(sub_df.columns) == [
        "request_id",
        "requester_received_pizza",
    ], "Submission columns incorrect"
    assert len(sub_df) == 100, "Submission row count incorrect"

    print("\n=== Demonstration Completed Successfully ===")
    print(f"Output Directory: {Config.WORKING_DIR}")


if __name__ == "__main__":
    main()
